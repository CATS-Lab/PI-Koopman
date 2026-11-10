"""Source-group-selected prediction with fixed neural architectures."""

from src.paths import OUTPUTS

import argparse, copy, json, math
import numpy as np
import pandas as pd
import torch
from src.data.openacc import crop
from src.training.prediction import resample
import src.training.protocol as adapter
import src.training.encounter as core
import src.training.staged as staged
import src.training.compact as compact
import src.models.neural as continuous

OUT = OUTPUTS / "neural"
ORIGIN = 20
RATE = 10
DEVICE = "cpu"
USE_GRAPH = False
BASE_BUNDLE = core.bundle


def clock_union(t, rate):
    native = np.round(np.asarray(t) - t[0], 12)
    control = np.round(np.arange(int(round(native[-1] * rate))) / rate, 12)
    union = np.unique(np.r_[native, control])
    score = np.searchsorted(union, native)
    assert len(control) == 5 * rate and np.array_equal(union[score], native)
    return union, score, np.isin(union, control)


def shifted_crop(tr, a, b, rate=None):
    offset = ORIGIN - 20
    return (
        crop(tr, a + offset, b + offset)
        if rate is None
        else resample(tr, a + offset, b + offset, rate)
    )


def device_bundle(ds, b, rate):
    d = BASE_BUNDLE(ds, b, rate)
    for key in ["x", "leader", "mt", "st"]:
        d[key] = d[key].to(DEVICE)
    return d


def sequences(data, cfg, family):
    H = max(1, int(round(cfg["horizon"] * data["rate"])))
    burn = (
        max(
            0,
            int(
                round(
                    min(cfg["burnin"], data["budget"] - H / data["rate"]) * data["rate"]
                )
            ),
        )
        if family == "LSTM"
        else 0
    )
    starts = np.arange(
        burn, data["x"].shape[1] - H, max(1, int(round(0.2 * data["rate"])))
    )
    assert len(starts) > 0
    return H, burn, starts


def losses(net, data, H, burn, starts):
    ix = (
        starts
        if torch.is_tensor(starts)
        else torch.tensor(starts, dtype=torch.long, device=DEVICE)
    )
    mem = None
    for j in range(burn):
        _, mem = net.acceleration(data["x"][:, ix - burn + j], mem)
    state = data["x"][:, ix]
    result = state.new_zeros(data["m"])
    for k in range(H):
        acc, mem = net.acceleration(state, mem)
        state = core.advance(
            state,
            acc,
            data["leader"][:, ix + k],
            data["leader"][:, ix + k + 1],
            1 / data["rate"],
            data["mt"],
            data["st"],
        )
        delta = (state - data["x"][:, ix + k + 1]) * data["st"]
        result += (
            ((delta[:, :, 0] / 10).square() + delta[:, :, 1].square()).mean(1) / 2 / H
        )
    return result


def cpu_prediction(net, data, burn):
    host = copy.copy(data)
    host["x"] = data["x"].cpu()
    cpu = copy.deepcopy(net).cpu()
    return adapter.physical_rollout(
        cpu, host["test"], host["mu"], host["sc"], host["x"], burn, host["rate"]
    )[0]


def full_loss(net, data, H, burn, starts):
    out = np.zeros(data["m"])
    with torch.no_grad():
        for j in range(0, len(starts), 8):
            out += (
                losses(net, data, H, burn, starts[j : j + 8]).cpu().numpy()
                * len(starts[j : j + 8])
                / len(starts)
            )
    return out


def train(
    data,
    family,
    cfg,
    seeds,
    dest,
    maxupdates,
    saveupdates,
    resume=None,
    eval_forecast=True,
):
    """Fit independent networks, saving the requested update checkpoints."""
    dest.mkdir(parents=True, exist_ok=True)
    core.SEEDS = seeds
    net = staged.initial(data, family, cfg["width"], seeds).to(DEVICE)
    H, burn, starts = sequences(data, cfg, family)
    lr = torch.tensor(cfg["lr"], device=DEVICE) if USE_GRAPH else cfg["lr"]
    opt = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=1e-5, capturable=USE_GRAPH, foreach=False
    )
    failed = torch.zeros(data["m"], dtype=torch.bool, device=DEVICE)
    begin = 0
    if resume:
        arch = torch.load(resume, map_location=DEVICE, weights_only=False)
        net.load_state_dict(arch["state_dict"])
        opt.load_state_dict(arch["optimizer"])
        failed.copy_(torch.as_tensor(arch["training_failed"], device=DEVICE))
        begin = arch["updates"]
        if USE_GRAPH:
            opt.param_groups[0]["lr"] = lr

    chosen = torch.tensor(
        [starts[begin % len(starts)]], dtype=torch.long, device=DEVICE
    )

    def step():
        opt.zero_grad(set_to_none=False)
        per = losses(net, data, H, burn, chosen)
        failed.logical_or_(~torch.isfinite(per))
        loss = torch.where(failed, torch.zeros_like(per), per).sum()
        loss.backward()
        norms = per.new_zeros(data["m"])
        for p in net.parameters():
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            p.grad.mul_((~failed).reshape(data["m"], *([1] * (p.ndim - 1))))
            norms += p.grad.reshape(data["m"], -1).square().sum(1)
        fac = torch.clamp(10 / (norms.sqrt() + 1e-9), max=1.0)
        for p in net.parameters():
            p.grad *= fac.reshape(data["m"], *([1] * (p.ndim - 1)))
        opt.step()
        return per

    graph = None
    if USE_GRAPH:
        state = copy.deepcopy(net.state_dict())
        optim = copy.deepcopy(opt.state_dict())
        fail0 = failed.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        net.load_state_dict(state)
        failed.copy_(fail0)
        if begin:
            opt.load_state_dict(optim)
            opt.param_groups[0]["lr"] = lr
        else:
            for state in opt.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        v.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            per = step()
        torch.cuda.synchronize()
    for update in range(begin + 1, maxupdates + 1):
        rate = cfg["lr"] * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(update, 3000) / 3000))
        )
        if USE_GRAPH:
            lr.fill_(rate)
        else:
            opt.param_groups[0]["lr"] = rate
        chosen.fill_(int(starts[(update - 1) % len(starts)]))
        if graph:
            graph.replay()
        else:
            per = step()
        if update not in saveupdates:
            continue
        fs = failed.cpu().numpy()
        arch = dict(
            state_dict=net.state_dict(),
            optimizer=opt.state_dict(),
            training_failed=fs,
            updates=update,
            config=cfg,
            seeds=seeds,
            uids=[t.uid for t in data["trials"]],
            mu=data["mu"],
            sc=data["sc"],
            family=family,
            device=DEVICE,
        )
        torch.save(arch, dest / f"update{update}.pt")
        if eval_forecast:
            pred = cpu_prediction(net, data, burn)
            pred[fs, 1:] = np.nan
            met = pd.DataFrame(core.metrics(pred, data, family, cfg, update)).rename(
                columns={"epoch": "updates"}
            )
            met.to_csv(dest / f"metrics{update}.csv", index=False)


def configure(ds, b, rate, origin, device="cpu", graph=False):
    global RATE, ORIGIN, DEVICE, USE_GRAPH
    RATE = rate
    ORIGIN = origin
    DEVICE = device
    USE_GRAPH = graph
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device) if ":" in device else 0)
    cell = OUT / "grid" / f"{ds}_B{b}_Hz{rate}_end{origin}"
    cell.mkdir(parents=True, exist_ok=True)
    adapter.RATE = rate
    adapter.clock_union = clock_union
    core.crop = shifted_crop
    core.bundle = device_bundle
    staged.OUT = cell / "staged"
    compact.crop = shifted_crop
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    return cell


def deep_candidates(ds, b, rate, cell, variants):
    tables = []
    for variant, epochs in variants.items():
        compact.OUT = cell / variant
        compact.MAX_STEP = 1 / rate if variant == "clock" else 0.04
        compact.rollout = (
            adapter.compact_generator_rollout
            if variant == "clock"
            else continuous.rollout
        )
        compact.run(ds, b, rate, checkpoints=epochs)
        path = compact.OUT / f"{ds}_B{b}_Hz{rate}_clean_0/candidate_metrics.csv"
        frame = pd.read_csv(path, dtype={"group": str})
        tables.append(frame.assign(numerical_variant=variant))
    return pd.concat(tables, ignore_index=True)


def run(ds, b, rate, origin, device, transfer, reference_origin):
    cell = configure(ds, b, rate, origin, device, device.startswith("cuda"))
    pieces = []
    if transfer:
        anchor = (
            OUT
            / "grid"
            / f"{ds}_B{b}_Hz10_end{reference_origin}"
            / "selected_raw_initializations.csv"
        )
        if not anchor.exists():
            run(ds, b, 10, reference_origin, device, False, reference_origin)
        if rate == 10 and origin == reference_origin:
            return
        cell = configure(ds, b, rate, origin, device, device.startswith("cuda"))
        settings = pd.read_csv(anchor, dtype={"group": str}).drop_duplicates(
            ["group", "model"]
        )
        for family in ("DNN", "LSTM"):
            for (variant, steps), rows in settings[settings.model == family].groupby(
                ["variant", "updates"]
            ):
                cfg = next(c for c in staged.grid(family, b) if c["id"] == variant)
                full = staged.bundle(ds, b, rate, staged.FINAL)
                ids = {t.uid for t in full["trials"] if str(t.run) in set(rows.group)}
                subset = staged.bundle(ds, b, rate, staged.FINAL, ids)
                steps = int(steps)
                dest = cell / "transferred" / family / variant / str(steps)
                train(subset, family, cfg, staged.FINAL, dest, steps, [steps])
                pieces.append(
                    pd.read_csv(dest / f"metrics{steps}.csv", dtype={"group": str})
                )
        deep = settings[settings.model == "DeepKoopman-projected"]
        variants = {
            v: sorted({json.loads(c)["epoch"] for c in g.config})
            for v, g in deep.groupby("numerical_variant")
        }
        pool = deep_candidates(ds, b, rate, cell, variants)
        for row in deep.itertuples():
            epoch = json.loads(row.config)["epoch"]
            pieces.append(
                pool[
                    (pool.group == row.group)
                    & (pool.numerical_variant == row.numerical_variant)
                    & pool.config.map(lambda c: json.loads(c)["epoch"] == epoch)
                ]
            )
    else:
        for family in ("DNN", "LSTM"):
            staged.run(ds, b, rate, family, train)
            pieces.append(
                pd.read_csv(
                    staged.OUT
                    / f"{ds}_B{b}_Hz{rate}"
                    / family
                    / "selected_metrics.csv",
                    dtype={"group": str},
                )
            )
        pool = deep_candidates(
            ds, b, rate, cell, {"clock": [150, 400], "continuous0.04": [150, 400]}
        )
        for group in sorted(pool.group.unique()):
            scores = (
                pool[pool.group != group]
                .groupby(["numerical_variant", "config", "trial_uid"])
                .score.mean()
            )
            variant, cfg = scores.groupby(level=[0, 1]).mean().idxmin()
            pieces.append(
                pool[
                    (pool.group == group)
                    & (pool.config == cfg)
                    & (pool.numerical_variant == variant)
                ]
            )
    result = pd.concat(pieces, ignore_index=True).assign(
        dataset=ds, budget_s=b, rate_hz=rate, origin_s=origin
    )
    keys = [
        "dataset",
        "budget_s",
        "rate_hz",
        "origin_s",
        "trial_uid",
        "case_id",
        "mode",
        "group",
        "model",
    ]
    if (
        result.duplicated(keys + ["neural_seed"]).any()
        or not result.groupby(keys).size().eq(5).all()
    ):
        raise ValueError("Incomplete neural fits")
    result.to_csv(cell / "selected_raw_initializations.csv", index=False)
    result.groupby(keys, as_index=False)[
        ["speed_rmse_mps", "gap_rmse_m", "score"]
    ].mean().to_csv(cell / "selected_encounter_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["local", "openacc"], default="local")
    parser.add_argument("--budget", type=int, default=7)
    parser.add_argument("--rate", type=int, default=10)
    parser.add_argument("--origin", type=int, default=20)
    parser.add_argument("--reference-origin", type=int, default=20)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--device", default="cpu")
    a = parser.parse_args()
    run(a.dataset, a.budget, a.rate, a.origin, a.device, a.transfer, a.reference_origin)


if __name__ == "__main__":
    main()
