"""Grouped configuration selection and paired neural acceleration readouts at 10 Hz."""

from src.paths import OUTPUTS

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.models import neural
from src.training import encounter, staged
from src.data.trials import load_trials
from src.data.openacc import openacc, observation_errors
from src.data.replays import records

SEEDS = (73, 211, 347, 509, 863)
ORIGINS = (15, 20, 24)
NOISE = [("clean", 0)] + [
    (kind, k) for kind in ("white", "correlated") for k in (1101, 1102, 1103)
]
DEEP = "DeepKoopman-projected"


def prepare(folder):
    meta = []
    states = []
    truth = []
    for ds, source in [("local", load_trials(30, 125)[0]), ("openacc", openacc()[0])]:
        for _, obj, tr, a in records(ds, source):
            meta.append((ds, obj, tr.uid, str(tr.run) if ds == "local" else tr.case))
            states.append(tr.z)
            truth.append(np.full(len(tr.t), np.nan) if a is None else a)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta, columns=["dataset", "object", "uid", "group"]).to_csv(
        folder / "records.csv", index=False
    )
    np.savez_compressed(folder / "records.npz", z=states, a=truth)


def configs(family):
    if family == DEEP:
        return [
            dict(id=k, substeps=s, lr=0.003, horizon=0.4)
            for k, s in [("clock", 1), ("continuous0.04", 3)]
        ]
    return staged.grid(family, 7)


def initialize(family, cfg, ids, seeds, device):
    ctor = (
        (
            lambda count, seed: neural.BatchNetwork(
                count, family, torch.Generator().manual_seed(seed)
            )
        )
        if family == DEEP
        else (lambda count, seed: encounter.Network(count, family, cfg["width"], seed))
    )
    net = ctor(len(ids) * len(seeds), 601)
    blocks = {}
    with torch.no_grad():
        for j, uid in enumerate(ids):
            for si, seed in enumerate(seeds):
                key = (
                    seed
                    if family == DEEP
                    else int(
                        hashlib.sha256(
                            f"{uid}|{family}|{cfg['width']}|{seed}".encode()
                        ).hexdigest()[:12],
                        16,
                    )
                )
                if key not in blocks:
                    blocks[key] = ctor(1, key)
                for full, one in zip(net.parameters(), blocks[key].parameters()):
                    full[j * len(seeds) + si].copy_(one[0])
    return net.to(device)


def train(family, cfg, z, ids, seeds, steps, device, evaluate):
    n = len(seeds)
    raw = torch.as_tensor(np.repeat(z, n, axis=0), device=device, dtype=torch.float32)
    u = raw[:, :, 1] + raw[:, :, 2]
    mu = raw.mean(1, keepdim=True)
    sc = raw.std(1, unbiased=False, keepdim=True).maximum(
        raw.new_tensor([1.0, 0.5, 0.5])
    )
    x = (raw - mu) / sc
    net = initialize(family, cfg, ids, seeds, device)
    deep = family == DEEP
    burn = 10 if family == "LSTM" else 0
    horizon = round(cfg["horizon"] * 10)
    starts = np.arange(burn, x.shape[1] - horizon, 4 if deep else 2)
    gpu = device.startswith("cuda")
    ix = torch.tensor(starts if deep else starts[:1], device=device)
    lr = torch.tensor(cfg["lr"], device=device) if gpu else cfg["lr"]
    opt = torch.optim.Adam(
        net.parameters(), lr=lr, capturable=gpu, foreach=False, weight_decay=1e-5
    )

    def step():
        opt.zero_grad(set_to_none=False)
        mem = None
        for j in range(burn):
            _, mem = net.acceleration(x[:, ix - burn + j], mem)
        state = net.initial(x[:, ix]) if deep else x[:, ix]
        loss = 0.0
        for k in range(horizon):
            left = u[:, ix + k]
            right = u[:, ix + k + 1]
            if deep:
                sub = cfg["substeps"]
                for j in range(sub):
                    state = net.step(
                        state,
                        left + (right - left) * j / sub,
                        left + (right - left) * (j + 1) / sub,
                        0.1 / sub,
                        mu,
                        sc,
                    )
                tgt = x[:, ix + k + 1]
                err = (state[:, :, :3] - tgt).square().mean((1, 2)) + 0.1 * (
                    state[:, :, 3:] - net.encode(tgt)[:, :, 3:]
                ).square().mean((1, 2))
                loss += 0.9**k * err / horizon
            else:
                acc, mem = net.acceleration(state, mem)
                state = encounter.advance(state, acc, left, right, 0.1, mu, sc)
                err = (state - x[:, ix + k + 1]) * sc
                loss += ((err[:, :, 0] / 10).square() + err[:, :, 1].square()).mean(
                    1
                ) / (2 * horizon)
        loss.sum().backward()
        norm = sum(
            a.grad.reshape(len(raw), -1).square().sum(1) for a in net.parameters()
        ).sqrt()
        fac = (10 / (norm + 1e-9)).clamp(max=1.0)
        for a in net.parameters():
            a.grad.mul_(fac.reshape(len(raw), *([1] * (a.ndim - 1))))
        opt.step()

    if gpu:
        initial = {k: v.clone() for k, v in net.state_dict().items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        net.load_state_dict(initial)
        for state in opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    for epoch in range(1, max(steps) + 1):
        if not deep:
            ix.fill_(int(starts[(epoch - 1) % len(starts)]))
            value = cfg["lr"] * (
                0.1 + 0.45 * (1 + math.cos(math.pi * min(epoch, 3000) / 3000))
            )
            if gpu:
                lr.fill_(value)
            else:
                opt.param_groups[0]["lr"] = value
        if gpu:
            graph.replay()
        else:
            step()
        if epoch in steps:
            with torch.no_grad():
                yield epoch, evaluate(net, mu, sc, x)


def forecast_score(family, cfg, target, seeds, device):
    n = len(seeds)
    fut = torch.as_tensor(
        np.repeat(target, n, axis=0), device=device, dtype=torch.float32
    )
    u = fut[:, :, 1] + fut[:, :, 2]

    def evaluate(net, mu, sc, past):
        state = (fut[:, :1] - mu) / sc
        mem = None
        if family == DEEP:
            state = net.initial(state)
        if family == "LSTM":
            for j in range(-11, -1):
                _, mem = net.acceleration(past[:, j : j + 1], mem)
        pred = []
        for j in range(50):
            left = u[:, j : j + 1]
            right = u[:, j + 1 : j + 2]
            if family == DEEP:
                sub = cfg["substeps"]
                for k in range(sub):
                    state = net.step(
                        state,
                        left + (right - left) * k / sub,
                        left + (right - left) * (k + 1) / sub,
                        0.1 / sub,
                        mu,
                        sc,
                    )
            else:
                a, mem = net.acceleration(state, mem)
                state = encounter.advance(state, a, left, right, 0.1, mu, sc)
            pred.append((state[:, :, :3] * sc + mu)[:, 0, :2])
        pred = torch.stack(pred, 1)
        rm = (pred - fut[:, 1:, :2]).square().mean(1).sqrt()
        score = rm[:, 0] / 10 + rm[:, 1]
        score = torch.where(
            torch.isfinite(score)
            & (pred[:, :, 0].abs().amax(1) <= 1e4)
            & (pred[:, :, 1].abs().amax(1) <= 1e3),
            score,
            2000.0,
        )
        return score.reshape(-1, n).mean(1).cpu().numpy()

    return evaluate


def screen(family, cfg, idx, steps, seeds, meta, z, device, batch):
    blocks = []
    for chunk in np.array_split(idx, max(1, math.ceil(len(idx) / batch))):
        uids = meta.loc[chunk, "fit_uid"].tolist()
        score = forecast_score(family, cfg, z[chunk, 200:251], seeds, device)
        for epoch, values in train(
            family, cfg, z[chunk, 130:201], uids, seeds, steps, device, score
        ):
            blocks.append(
                pd.DataFrame(
                    dict(
                        object=meta.loc[chunk, "object"],
                        cfg=cfg["id"],
                        steps=epoch,
                        score=values,
                    )
                )
            )
    return (
        pd.concat(blocks)
        .groupby(["object", "cfg", "steps"], as_index=False)
        .score.mean()
    )


def select(family, meta, z, device, batch, out):
    idx = meta.index[meta.object != "recorded"].to_numpy()
    objects = sorted(meta.loc[idx, "object"].unique())
    allowed = configs(family)
    deep = family == DEEP
    seeds = SEEDS if deep else SEEDS[:2]

    def measure(cfg, steps):
        path = out / f"{family}_{cfg['id']}_{max(steps)}.csv"
        if not path.exists():
            print("select", family, cfg["id"], steps, flush=True)
            table = screen(family, cfg, idx, steps, seeds, meta, z, device, batch)
            pending = path.with_suffix(".tmp")
            table.to_csv(pending, index=False)
            pending.replace(path)
        return pd.read_csv(path)

    history = pd.concat(
        [measure(cfg, [150, 400] if deep else [400, 1200]) for cfg in allowed]
    )

    def rank(obj, table):
        return (
            table[table.object != obj]
            .groupby(["cfg", "steps"])
            .score.mean()
            .sort_values(kind="stable")
        )

    picks = {obj: rank(obj, history).index[0] for obj in objects}
    if not deep:
        for cfg in allowed:
            if cfg["id"] not in {c for c, _ in picks.values()}:
                continue
            table = pd.concat([history[history.cfg == cfg["id"]], measure(cfg, [3000])])
            extended = set()
            for obj in objects:
                if picks[obj][0] != cfg["id"]:
                    continue
                scores = rank(obj, table)
                if scores.loc[cfg["id"], 3000] < 0.99 * min(
                    scores.loc[cfg["id"], 400], scores.loc[cfg["id"], 1200]
                ):
                    extended.add(obj)
                picks[obj] = scores.index[0]
            if extended:
                table = pd.concat([table, measure(cfg, [6000])])
                for obj in extended:
                    picks[obj] = rank(obj, table).index[0]
    return picks


def responses(family, clean, device):
    raw = torch.as_tensor(
        np.repeat(clean, len(SEEDS), axis=0), device=device, dtype=torch.float32
    )

    def evaluate(net, mu, sc, past):
        x = (raw - mu) / sc
        if family == DEEP:
            state = net.initial(x[:, 10:])
            du = (
                raw[:, 10:, 1] + raw[:, 10:, 2] - raw[:, 9:-1, 1] - raw[:, 9:-1, 2]
            ) / sc[:, :, 2]
            a = (torch.bmm(state, net.G)[:, :, 1] + 10 * du * net.B[:, :, 1]) * sc[
                :, :, 1
            ]
        elif family == "LSTM":
            mem = None
            for j in range(11):
                a, mem = net.acceleration(x[:, j : j + 61], mem)
        else:
            a, _ = net.acceleration(x[:, 10:], None)
        return a.cpu().numpy().reshape(-1, len(SEEDS), 61)

    return evaluate


def noisy(z, uid, noise, seed):
    if noise == "clean":
        return z
    e = observation_errors(uid, len(z), 0.1, noise, seed)
    v = z[:, 1] + e[:, 1]
    u = z[:, 1] + z[:, 2] + e[:, 2]
    return np.c_[z[:, 0] + e[:, 0], v, u - v]


def run(args):
    folder = args.input
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    meta = pd.read_csv(folder / "records.csv", dtype={"group": str})
    meta["fit_uid"] = np.where(
        meta.object == "recorded", meta.uid, meta.uid + "_" + meta.object
    )
    archive = np.load(folder / "records.npz")
    z = archive["z"]
    truth = archive["a"]
    reference = pd.concat(
        [pd.read_csv(f, dtype={"group": str}) for f in args.profiles]
    ).query("budget_s==7")
    families = args.family or ["DNN", "LSTM", DEEP]
    for family in families:
        choices = select(family, meta, z, args.device, args.batch, out)
        pd.DataFrame(
            [(k, *v) for k, v in choices.items()],
            columns=["object", "variant", "updates"],
        ).to_csv(out / f"choices_{family}.csv", index=False)
        groups = {}
        for i, r in meta.iterrows():
            if r.object == "recorded":
                row = reference[
                    (reference.dataset == r.dataset)
                    & (reference.group == r.group)
                    & (reference.model == family)
                ].iloc[0]
                choice = (
                    (row["variant"], int(row.updates))
                    if family != DEEP
                    else (row.numerical_variant, int(json.loads(row.config)["epoch"]))
                )
            else:
                choice = choices[r.object]
            groups.setdefault(choice, []).append(i)
        for (variant, updates), indices in groups.items():
            cfg = next(c for c in configs(family) if c["id"] == variant)
            for block in range(0, len(indices), args.response_batch):
                group = indices[block : block + args.response_batch]
                path = out / f"field_{family}_{variant}_{updates}_{block}.csv"
                if path.exists():
                    continue
                rows = []
                for origin in ORIGINS:
                    sl = slice((origin - 7) * 10, origin * 10 + 1)
                    clean = z[group, sl]
                    ids = meta.loc[group, "fit_uid"].tolist()
                    uids = meta.loc[group, "uid"].tolist()
                    # Fit clean/noise pairs together using identical initial weights.
                    batch = np.stack(
                        [
                            noisy(z[i], uids[j], noise, seed)[sl]
                            for j, i in enumerate(group)
                            for noise, seed in NOISE
                        ]
                    )
                    fit_ids = [uid for uid in ids for _ in NOISE]
                    eval_fn = responses(
                        family, np.repeat(clean, len(NOISE), axis=0), args.device
                    )
                    _, a = next(
                        train(
                            family,
                            cfg,
                            batch,
                            fit_ids,
                            SEEDS,
                            [updates],
                            args.device,
                            eval_fn,
                        )
                    )
                    a = a.reshape(len(group), len(NOISE), len(SEEDS), 61)
                    if not np.isfinite(a).all():
                        raise ArithmeticError((family, variant, block, origin))
                    for j, i in enumerate(group):
                        exact = truth[i, (origin - 7) * 10 + 10 : origin * 10 + 1]
                        for h, (kind, seed) in enumerate(NOISE):
                            delta = a[j, h] - a[j, 0]
                            rows.append(
                                (
                                    meta.dataset[i],
                                    meta.object[i],
                                    uids[j],
                                    origin,
                                    kind,
                                    seed,
                                    family,
                                    np.sqrt((delta.astype(float) ** 2).mean(1)).mean(),
                                    delta.mean(),
                                    np.sqrt(((a[j, h] - exact) ** 2).mean(1)).mean(),
                                )
                            )
                pd.DataFrame(
                    rows,
                    columns=[
                        "dataset",
                        "object",
                        "uid",
                        "origin",
                        "noise",
                        "noise_seed",
                        "model",
                        "change",
                        "signed_change",
                        "truth_rmse",
                    ],
                ).to_csv(path.with_suffix(".tmp"), index=False)
                path.with_suffix(".tmp").replace(path)
                print(
                    "fields",
                    family,
                    variant,
                    updates,
                    block + len(group),
                    "/",
                    len(indices),
                    flush=True,
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=(OUTPUTS / "noise/neural"))
    ap.add_argument(
        "--profiles",
        type=Path,
        nargs="+",
        default=[
            (OUTPUTS / "summaries/local_neural_settings.csv"),
            (OUTPUTS / "summaries/openacc_neural_settings.csv"),
        ],
    )
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--response-batch", type=int, default=12)
    ap.add_argument("--family", nargs="+", choices=["DNN", "LSTM", DEEP])
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.prepare:
        prepare(args.input)
    else:
        run(args)
