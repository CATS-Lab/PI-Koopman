#!/usr/bin/env python3
"""Projected Deep Koopman training on 30-s encounters."""

from src.paths import OUTPUTS

from src.paths import OPENACC_RAW
import json
import math
import time
import pandas as pd
import numpy as np
import torch
from src.data.trials import load_trials, write_csv
from src.data.openacc import crop, openacc
from src.models.neural import (
    DEFAULT_CHECKPOINTS,
    SEEDS,
    MAX_STEP,
    initialize,
    metric,
    rollout,
)

OUT = OUTPUTS / "neural/compact/neural"


def run(dataset, budget, rate, checkpoints=DEFAULT_CHECKPOINTS):
    family = "DeepKoopman-projected"
    scenario, corruption_seed = "clean", 0
    tag = f"{dataset}_B{budget}_Hz{rate}_{scenario}_{corruption_seed}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    candidate_file = out / "candidate_metrics.csv"
    if candidate_file.exists():
        saved = pd.read_csv(candidate_file, usecols=["config"])
        done = {json.loads(c)["epoch"] for c in saved.config.unique()}
        if set(checkpoints).issubset(done):
            return
    if dataset == "local":
        trials, path = load_trials(30, 125)
        meta = None
        end = 20
    else:
        trials, meta = openacc()
        path = OPENACC_RAW
        end = 20
    train = [crop(tr, end - budget, end, rate) for tr in trials]
    test = [crop(tr, end, end + 5) for tr in trials]
    n = len(train)
    raw = np.array([tr.z for seed in SEEDS for tr in train], np.float32)
    observed_u = np.array([tr.u for seed in SEEDS for tr in train], np.float32)
    mu = raw.mean(1, keepdims=True)
    scale = np.maximum(raw.std(1, keepdims=True), np.array([1.0, 0.5, 0.5], np.float32))
    X = (raw - mu) / scale
    # Fixed .4 s physical training horizon where observable. At1Hz, the minimum is1s.
    H = max(1, int(round(0.4 * rate)))
    stride = max(1, int(round(0.4 * rate)))
    starts = np.arange(0, raw.shape[1] - H, stride)
    target = np.stack([X[:, j : j + H + 1] for j in starts], axis=1)
    inputs = np.stack([observed_u[:, j : j + H + 1] for j in starts], axis=1)
    tar = torch.tensor(target)
    inp = torch.tensor(inputs)
    mt = torch.tensor(mu)
    st = torch.tensor(scale)
    dt = float(np.median(np.diff(train[0].t)))
    substeps = max(1, int(math.ceil(dt / MAX_STEP - 1e-8)))
    h = dt / substeps
    rows = []
    net = initialize(n, family)
    opt = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-5)
    started = time.perf_counter()
    m = n * len(SEEDS)
    npar = sum(p.numel() for p in net.parameters()) // m
    for epoch in range(1, max(checkpoints) + 1):
        opt.zero_grad()
        state = net.initial(tar[:, :, 0])
        loss_per = torch.zeros(m)
        for k in range(H):
            for sub in range(substeps):
                ul = inp[:, :, k] + (inp[:, :, k + 1] - inp[:, :, k]) * sub / substeps
                ur = (
                    inp[:, :, k]
                    + (inp[:, :, k + 1] - inp[:, :, k]) * (sub + 1) / substeps
                )
                state = net.step(state, ul, ur, h, mt, st)
            err = (net.physical(state) - tar[:, :, k + 1]).square().mean((1, 2))
            if family == "DeepKoopman-projected":
                reference = net.encode(tar[:, :, k + 1])
                err = err + 0.1 * (state[:, :, 3:] - reference[:, :, 3:]).square().mean(
                    (1, 2)
                )
            loss_per += 0.9 ** (k * 10 / rate) * err
        loss = loss_per.sum() / H
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{tag} {family} nonfinite calibration loss at epoch{epoch}"
            )
        loss.backward()
        norms = torch.zeros(m)
        for p in net.parameters():
            norms += p.grad.reshape(m, -1).square().sum(1)
        fac = torch.clamp(10 / (norms.sqrt() + 1e-9), max=1.0)
        for p in net.parameters():
            p.grad *= fac.reshape(m, *([1] * (p.ndim - 1)))
        opt.step()
        if epoch % 50 == 0:
            print(
                tag,
                family,
                "epoch",
                epoch,
                "train_loss",
                float(loss.detach() / m),
                "seconds",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
        if epoch not in checkpoints:
            continue
        fit_seconds = time.perf_counter() - started

        pred, audit = rollout(net, test, mu, scale)
        cfg = json.dumps(
            dict(
                epoch=epoch,
                width=16,
                latent=6 if family == "DeepKoopman-projected" else 0,
                max_internal_step_s=MAX_STEP,
                projection="d=u-v,gap_trapezoid",
                training_horizon_s=H * dt,
            ),
            sort_keys=True,
        )
        for init_index, seed in enumerate(SEEDS):
            for i, tr in enumerate(trials):
                base = dict(
                    dataset=dataset,
                    trial_uid=tr.uid,
                    case_id=tr.case,
                    mode=tr.mode,
                    group=str(tr.run) if dataset == "local" else tr.case,
                    budget_s=budget,
                    rate_hz=rate,
                    scenario=scenario,
                    seed=corruption_seed,
                    neural_seed=seed,
                    model=family,
                    config=cfg,
                    parameters=npar,
                    fit_seconds=fit_seconds / (n * len(SEEDS)),
                    batch_fit_seconds=fit_seconds,
                    optimizer_success=True,
                )
                if meta is not None:
                    base.update(
                        campaign=str(meta.iloc[i].campaign),
                        vehicle=str(meta.iloc[i].vehicle),
                    )
                rows.append(dict(**base, **metric(pred[init_index * n + i], test[i])))
        print(
            tag,
            family,
            "saved epoch",
            epoch,
            "prediction failures",
            audit["failures"],
            flush=True,
        )
    write_csv(out / "candidate_metrics.csv", rows)
