"""Independent fixed-width networks for encounter-level prediction."""

import math
import numpy as np
import torch
from torch import nn
from src.data.trials import load_trials
from src.data.openacc import openacc, crop
from src.paths import OPENACC_RAW

SEEDS = [73, 211, 347, 509, 863]


class Linear(nn.Module):
    def __init__(self, m, a, b, g, scale=None):
        super().__init__()
        self.w = nn.Parameter(
            torch.randn(m, a, b, generator=g)
            * (1 / math.sqrt(a) if scale is None else scale)
        )
        self.b = nn.Parameter(torch.zeros(m, 1, b))

    def forward(self, x):
        return torch.bmm(x, self.w) + self.b


class Network(nn.Module):
    def __init__(self, m, family, width, seed):
        super().__init__()
        self.family = family
        self.width = width
        g = torch.Generator().manual_seed(seed)
        if family == "DNN":
            self.f1 = Linear(m, 3, width, g)
            self.f2 = Linear(m, width, width, g)
        else:
            self.gates = Linear(m, 3 + width, 4 * width, g)
        self.out = Linear(m, width, 1, g, 0.02)

    def acceleration(self, x, mem=None):
        if self.family == "DNN":
            return self.out(torch.tanh(self.f2(torch.tanh(self.f1(x)))))[:, :, 0], None
        if mem is None:
            mem = (
                x.new_zeros((*x.shape[:2], self.width)),
                x.new_zeros((*x.shape[:2], self.width)),
            )
        hh, cc = mem
        i, f, o, j = self.gates(torch.cat([x, hh], -1)).chunk(4, -1)
        cc = torch.sigmoid(f) * cc + torch.sigmoid(i) * torch.tanh(j)
        hh = torch.sigmoid(o) * torch.tanh(cc)
        return self.out(hh)[:, :, 0], (hh, cc)


def advance(x, acc, u0, u1, dt, mu, sc):
    z = x * sc + mu
    s, v, d = z.unbind(-1)
    vn = v + dt * acc
    sn = s + 0.5 * dt * (u0 + u1 - v - vn)
    return (torch.stack([sn, vn, u1 - vn], -1) - mu) / sc


def bundle(dataset, budget, rate):
    trials, path = (
        load_trials(30, 125) if dataset == "local" else (openacc()[0], OPENACC_RAW)
    )
    train = [crop(t, 20 - budget, 20, rate) for t in trials]
    test = [crop(t, 20, 25) for t in trials]
    z = np.array([t.z for seed in SEEDS for t in train], dtype=np.float32)
    u = np.array([t.u for seed in SEEDS for t in train], dtype=np.float32)
    mu = z.mean(1, keepdims=True)
    sc = np.maximum(z.std(1, keepdims=True), np.array([1.0, 0.5, 0.5], np.float32))
    return dict(
        trials=trials,
        path=path,
        train=train,
        test=test,
        z=z,
        u=u,
        mu=mu,
        sc=sc,
        x=torch.tensor((z - mu) / sc),
        leader=torch.tensor(u),
        mt=torch.tensor(mu),
        st=torch.tensor(sc),
        n=len(trials),
        m=len(z),
        rate=rate,
        budget=budget,
    )


def metrics(pred, data, family, cfg, epoch):
    rows = []
    n = data["n"]
    for si, seed in enumerate(SEEDS):
        for i, tr in enumerate(data["trials"]):
            p = pred[si * n + i]
            fail = not np.isfinite(p).all()
            rm = (
                np.array([1e4, 1e3])
                if fail
                else np.sqrt(np.mean((p[1:] - data["test"][i].z[1:, :2]) ** 2, axis=0))
            )
            rows.append(
                dict(
                    trial_uid=tr.uid,
                    case_id=tr.case,
                    mode=tr.mode,
                    group=str(tr.run) if tr.run else tr.case,
                    model=family,
                    variant=cfg["id"],
                    epoch=epoch,
                    neural_seed=seed,
                    gap_rmse_m=rm[0],
                    speed_rmse_mps=rm[1],
                    score=rm[1] + rm[0] / 10,
                    failed=int(fail),
                    negative_gap=int(np.any(p[:, 0] < 0)),
                    negative_speed=int(np.any(p[:, 1] < 0)),
                )
            )
    return rows
