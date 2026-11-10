"""Select training horizon, learning rate and update count on other groups."""

from src.paths import OUTPUTS

import hashlib
import numpy as np
import pandas as pd
import torch
from src.training import encounter as core


OUT = OUTPUTS / "neural/staged"
SCREEN = [73, 211]
FINAL = [73, 211, 347, 509, 863]


def grid(family, budget=None):
    width = {"DNN": 16, "LSTM": 8}[family]
    options = [(0.4, 0.003), (1.0, 0.003), (2.0, 0.003), (2.0, 0.001)]
    if budget is not None and budget >= 8:
        options.append((5.0, 0.001))
    return [
        dict(
            id=f"w{width}_h{horizon:g}_lr{lr:g}",
            width=width,
            horizon=horizon,
            lr=lr,
            burnin=1.0,
        )
        for horizon, lr in options
    ]


def initial(data, family, width, seeds):
    net = core.Network(data["n"] * len(seeds), family, width, 601)
    for si, seed in enumerate(seeds):
        for ti, tr in enumerate(data["trials"]):
            stable = int(
                hashlib.sha256(
                    f"{tr.uid}|{family}|{width}|{seed}".encode()
                ).hexdigest()[:12],
                16,
            )
            block = core.Network(1, family, width, stable)
            with torch.no_grad():
                for dst, src in zip(net.parameters(), block.parameters()):
                    dst[si * data["n"] + ti : si * data["n"] + ti + 1].copy_(src)
    return net


def bundle(ds, b, rate, seeds, uids=None):
    core.SEEDS = seeds
    data = core.bundle(ds, b, rate)
    if uids is None:
        return data
    keep = [i for i, t in enumerate(data["trials"]) if t.uid in uids]
    ix = np.array([si * data["n"] + i for si in range(len(seeds)) for i in keep])
    for k in ["z", "u", "mu", "sc"]:
        data[k] = data[k][ix]
    for k in ["x", "leader", "mt", "st"]:
        data[k] = data[k][ix]
    for k in ["trials", "train", "test"]:
        data[k] = [data[k][i] for i in keep]
    data["n"] = len(keep)
    data["m"] = len(ix)
    return data


def read_candidates(base, family, updates):
    rows = []
    budget = int(base.name.split("_B")[1].split("_")[0])
    for cfg in grid(family, budget):
        for u in updates:
            p = base / "screen" / family / cfg["id"] / f"metrics{u}.csv"
            if p.exists():
                rows.append(pd.read_csv(p, dtype={"group": str}))
    return pd.concat(rows, ignore_index=True)


def scores(df, group):
    source = df[df.group != str(group)]
    return (
        source.groupby(["variant", "updates", "trial_uid"])
        .score.mean()
        .groupby(["variant", "updates"])
        .mean()
        .sort_values()
    )


def run(ds, b, rate, family, train):
    base = OUT / f"{ds}_B{b}_Hz{rate}"
    familyout = base / family
    familyout.mkdir(parents=True, exist_ok=True)
    if (familyout / "selected_metrics.csv").exists():
        return
    data = bundle(ds, b, rate, SCREEN)
    groups = sorted({str(t.run) if t.run else t.case for t in data["trials"]})
    for cfg in grid(family, b):
        dest = base / "screen" / family / cfg["id"]
        if not (dest / "metrics1200.csv").exists():
            train(data, family, cfg, SCREEN, dest, 1200, [400, 1200])
    candidates = read_candidates(base, family, [400, 1200])
    traces = []
    for g in groups:
        ss = scores(candidates, g)
        v, u = ss.index[0]
        traces.append(dict(group=g, variant=v))
    chosen = {t["variant"] for t in traces}
    for cfg in grid(family, b):
        if cfg["id"] not in chosen:
            continue
        dest = base / "screen" / family / cfg["id"]
        if not (dest / "metrics3000.csv").exists():
            train(data, family, cfg, SCREEN, dest, 3000, [3000], dest / "update1200.pt")
    candidates = read_candidates(base, family, [400, 1200, 3000])
    for trace in traces:
        ss = scores(candidates[candidates.variant == trace["variant"]], trace["group"])
        s3000 = float(ss.loc[(trace["variant"], 3000)])
        oldbest = min(float(ss.loc[(trace["variant"], u)]) for u in [400, 1200])
        trace["extension_to6000"] = bool(s3000 < 0.99 * oldbest)
    extensions = {t["variant"] for t in traces if t["extension_to6000"]}
    for cfg in grid(family, b):
        if cfg["id"] not in extensions:
            continue
        dest = base / "screen" / family / cfg["id"]
        if not (dest / "metrics6000.csv").exists():
            train(data, family, cfg, SCREEN, dest, 6000, [6000], dest / "update3000.pt")
    candidates = read_candidates(base, family, [400, 1200, 3000, 6000])
    for trace in traces:
        admissible = [400, 1200, 3000] + ([6000] if trace["extension_to6000"] else [])
        df = candidates[
            (candidates.variant == trace["variant"])
            & candidates.updates.isin(admissible)
        ]
        ss = scores(df, trace["group"])
        v, u = ss.index[0]
        trace["locked_updates"] = int(u)
    selected = []
    for cfg in grid(family, b):
        ts = [t for t in traces if t["variant"] == cfg["id"]]
        if not ts:
            continue
        targetgroups = {t["group"] for t in ts}
        uids = {
            t.uid
            for t in data["trials"]
            if (str(t.run) if t.run else t.case) in targetgroups
        }
        finaldata = bundle(ds, b, rate, FINAL, uids)
        wanted = sorted({t["locked_updates"] for t in ts})
        dest = base / "final5" / family / cfg["id"]
        if not all((dest / f"metrics{u}.csv").exists() for u in wanted):
            train(finaldata, family, cfg, FINAL, dest, max(wanted), wanted)
        for trace in ts:
            ff = pd.read_csv(
                dest / f"metrics{trace['locked_updates']}.csv", dtype={"group": str}
            )
            ff = ff[ff.group == trace["group"]].copy()
            ff["variant_source"] = "strong_5seeds"
            selected.append(ff)
    pd.concat(selected).to_csv(familyout / "selected_metrics.csv", index=False)
