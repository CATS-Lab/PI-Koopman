"""Time one target and one initialization per call using locked duration profiles."""

from src.paths import OUTPUTS

from src.data.trials import load_trials
from src.data.openacc import openacc
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from experiments import neural_prediction as exp
from src.training import prediction as phys
from src.models import neural
from src.training import encounter, staged, protocol

SEED = 73
CASES = {
    "local": [
        "ACC1_follows_HV1_run02",
        "ACC2_follows_HV1_run02",
        "ACC2_follows_ACC1_run01",
        "ACC2_follows_HV1_run01",
        "ACC1_follows_HV2_run03",
        "ACC1_follows_ACC1_run03",
    ],
    "openacc": [
        "AstaZero_ASta_050719_platoon2_F5_B0044",
        "Casale_part6_F2_B0000",
        "Casale_part4_F2_B0000",
        "AstaZero_ASta_040719_platoon5_F3_B0016",
        "Casale_part5_F2_B0000",
        "Casale_part10_F2_B0006",
    ],
}
torch.set_num_threads(1)
neural.SEEDS = encounter.SEEDS = [SEED]


def train_neural(trial, train, future, row, device="cpu"):
    z = train.z[None].astype("float32")
    u = train.u[None].astype("float32")
    mean = z.mean(1, keepdims=True)
    scale = np.maximum(z.std(1, keepdims=True), np.array([1.0, 0.5, 0.5], np.float32))
    x = torch.tensor((z - mean) / scale)
    mt, st, leader = map(torch.tensor, (mean, scale, u))
    if device != "cpu":
        x, mt, st, leader = (v.to(device) for v in (x, mt, st, leader))
    family = row.model
    rate = 10
    budget = row.budget_s
    compact = pd.isna(row.get("updates"))
    if compact:
        cfg = json.loads(row.config)
        epochs = cfg["epoch"]
        net = neural.initialize(1, family)
        horizon = 4
        starts = np.arange(0, x.shape[1] - horizon, horizon)
        dt = 0.1
        continuous = row.numerical_variant == "continuous0.04"
        substeps = 3 if continuous else 1
        lr = 0.003
    else:
        cfg = next(c for c in staged.grid(family, budget) if c["id"] == row["variant"])
        epochs = int(row.updates)
        data = dict(
            x=x,
            mt=mt,
            st=st,
            leader=leader,
            rate=rate,
            budget=budget,
            m=1,
            n=1,
            trials=[trial],
        )
        net = staged.initial(data, family, cfg["width"], [SEED])
        horizon, burn, starts = exp.sequences(data, cfg, family)
        lr = cfg["lr"]
    use_cuda = device.startswith("cuda")
    net = net.to(device)
    rate_tensor = torch.tensor(lr, device=device) if use_cuda else lr
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=rate_tensor,
        weight_decay=1e-5,
        foreach=False,
        capturable=use_cuda,
    )
    chosen = torch.tensor([int(starts[0])], device=device)
    if use_cuda and compact:
        starts = torch.tensor(starts, device=device)

    def step():
        optimizer.zero_grad(set_to_none=False)
        if compact:
            memory = None
            state = x[:, starts] if family == "LSTM" else net.initial(x[:, starts])
            loss = 0.0
            for k in range(horizon):
                idx = starts + k
                left = leader[:, idx]
                delta = leader[:, idx + 1] - left
                for j in range(substeps):
                    state = net.step(
                        state,
                        left + delta * j / substeps,
                        left + delta * (j + 1) / substeps,
                        dt / substeps,
                        mt,
                        st,
                    )
                error = (net.physical(state) - x[:, idx + 1]).square().mean()
                if family == "DeepKoopman-projected":
                    error += (
                        0.1
                        * (state[:, :, 3:] - net.encode(x[:, idx + 1])[:, :, 3:])
                        .square()
                        .mean()
                    )
                loss += 0.9**k * error / horizon
        else:
            loss = exp.losses(net, data, horizon, burn, chosen).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()

    if use_cuda:
        initial = {k: v.clone() for k, v in net.state_dict().items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        net.load_state_dict(initial)
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    for epoch in range(1, epochs + 1):
        if not compact:
            chosen.fill_(int(starts[(epoch - 1) % len(starts)]))
            current_lr = lr * (
                0.1 + 0.45 * (1 + math.cos(math.pi * min(epoch, 3000) / 3000))
            )
            if use_cuda:
                rate_tensor.fill_(current_lr)
            else:
                optimizer.param_groups[0]["lr"] = current_lr
        if use_cuda:
            graph.replay()
        else:
            step()
    if use_cuda:
        torch.cuda.synchronize()
    fit_end = time.perf_counter()
    if use_cuda:
        net = net.cpu()
        x = x.cpu()
    if not compact:
        pred = protocol.physical_rollout(net, [future], mean, scale, x, burn)
    elif continuous:
        pred = neural.rollout(net, [future], mean, scale)
    else:
        pred = protocol.compact_generator_rollout(net, [future], mean, scale)
    return pred[0][0], epochs, fit_end


def run(dataset, budget, device="cpu", output=None):
    cell = exp.OUT / "grid" / f"{dataset}_B{budget}_Hz10_end20"
    profiles = pd.read_csv(cell / "selected_raw_initializations.csv")
    profiles = profiles[profiles.neural_seed.eq(SEED)]
    pi = pd.read_csv(
        OUTPUTS / f"summaries/{dataset}_pi_settings.csv",
        dtype={"heldout_group": str},
    )
    pi = pi[pi.budget_s == budget]
    uids = CASES[dataset]
    trials = load_trials(30, 125)[0] if dataset == "local" else openacc()[0]
    rows = []
    for uid in uids:
        trial = next(t for t in trials if t.uid == uid)
        train = phys.resample(trial, 20 - budget, 20, 10)
        future = exp.crop(trial, 20, 25)
        methods = ["DeepKoopman-projected", "DNN", "LSTM"]
        if device == "cpu":
            methods = ["CTH", "IDM", *phys.PI, *methods]
        for name in methods:
            selected = profiles[(profiles.trial_uid == uid) & (profiles.model == name)]
            updates = 0
            if name in phys.PI:
                group = str(trial.run) if dataset == "local" else trial.case
                row = pi[(pi.heldout_group == group) & (pi.model == name)].iloc[0]
                cfg = dict(T=row["T"], ridge=row.ridge, kappa=row.kappa)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            if name in ["CTH", "IDM", *phys.PI]:
                family = name.split("-")[-1]
                p, audit = phys.physics.fit(train.t, train.z, train.u, family)
                model = dict(
                    kind="physical", family=family.lower(), coef=p, audit=audit
                )
                if name in phys.PI:
                    dictionary, prior = phys.PI[name]
                    model = phys.koop.fit(
                        train.t,
                        train.z,
                        train.u,
                        "expweak",
                        dict(
                            **cfg, dictionary=dictionary, prior=prior, prior_model=model
                        ),
                    )
                fit_end = time.perf_counter()
                forecast = phys.physics.predict_field(
                    model, future.t, future.u, future.z[0, :2]
                )
            else:
                forecast, updates, fit_end = train_neural(
                    trial, train, future, selected.iloc[0], device
                )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            rows.append(
                dict(
                    dataset=dataset,
                    budget_s=budget,
                    trial_uid=uid,
                    model=name,
                    updates=updates,
                    fit_s=fit_end - start,
                    forecast_s=elapsed - (fit_end - start),
                    single_fit_and_forecast_s=elapsed,
                    finite=np.isfinite(forecast).all(),
                )
            )
        print(dataset, budget, uid, flush=True)
    path = output or OUTPUTS / ("timing.csv" if device == "cpu" else "timing_cuda.csv")
    result = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_csv(path)
        retained = previous[
            (previous.dataset != dataset) | (previous.budget_s != budget)
        ]
        result = pd.concat([retained, result], ignore_index=True)
    result.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["local", "openacc"])
    parser.add_argument("budget", type=int, choices=[5, 7, 11])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, device=args.device))], lr=0.003)
    run(args.dataset, args.budget, args.device, args.output)
