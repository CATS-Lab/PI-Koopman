"""Projected neural generator and native-grid evaluation."""

from __future__ import annotations

from src.paths import OUTPUTS
import math
import numpy as np
import torch
from torch import nn

OUT = OUTPUTS / "models/neural"
torch.set_num_threads(2)
SEEDS = [73, 211, 347, 509, 863]
DEFAULT_CHECKPOINTS = [150, 400]
MAX_STEP = 0.04


class BatchLinear(nn.Module):
    def __init__(self, m, a, b, generator, scale=None):
        super().__init__()
        scale = 1 / math.sqrt(a) if scale is None else scale
        self.w = nn.Parameter(torch.randn(m, a, b, generator=generator) * scale)
        self.b = nn.Parameter(torch.zeros(m, 1, b))

    def forward(self, x):
        return torch.bmm(x, self.w) + self.b


class BatchNetwork(nn.Module):
    def __init__(self, m, family, generator):
        super().__init__()
        self.family = family
        if family != "DeepKoopman-projected":
            raise ValueError(f"Unknown generator {family}")
        self.latent = 6
        self.f1 = BatchLinear(m, 3, 16, generator)
        self.f2 = BatchLinear(m, 16, 16, generator)
        self.out = BatchLinear(m, 16, 6, generator, scale=0.02)
        self.G = nn.Parameter(torch.randn(m, 9, 9, generator=generator) * 0.001)
        self.B = nn.Parameter(torch.zeros(m, 1, 9))
        self.B.data[:, :, 2] = 1.0

    def encode(self, x):
        h = self.out(torch.tanh(self.f2(torch.tanh(self.f1(x)))))
        return torch.cat([x, h], dim=-1)

    def initial(self, x):
        return self.encode(x)

    def physical(self, state):
        return state[:, :, :3]

    def step(self, state, u0, u1, dt, mu, scale):
        """One small generator step with exact relative-speed projection and trapezoid gap."""
        x = self.physical(state)
        raw = x * scale + mu
        s, v, d = raw.unbind(-1)
        du = (u1 - u0) / scale[:, :, 2]
        candidate = state + dt * torch.bmm(state, self.G) + du[:, :, None] * self.B
        vn = candidate[:, :, 1] * scale[:, :, 1] + mu[:, :, 1]
        h = candidate[:, :, 3:]
        sn = s + 0.5 * dt * ((u0 - v) + (u1 - vn))
        dn = u1 - vn
        physical = (torch.stack([sn, vn, dn], dim=-1) - mu) / scale
        return torch.cat([physical, h], dim=-1)


def initialize(n, family):
    net = BatchNetwork(n * len(SEEDS), family, torch.Generator().manual_seed(601))
    for j, seed in enumerate(SEEDS):
        block = BatchNetwork(n, family, torch.Generator().manual_seed(seed))
        with torch.no_grad():
            for full, piece in zip(net.parameters(), block.parameters()):
                full[j * n : (j + 1) * n].copy_(piece)
    return net


def rollout(net, tests, mu, scale):
    n = len(tests)
    m = n * len(SEEDS)
    t = tests[0].t
    assert all(np.array_equal(tr.t, t) for tr in tests)
    u = torch.tensor(np.array([tr.u for seed in SEEDS for tr in tests], np.float32))
    raw0 = torch.tensor(
        np.array([tr.z[0] for seed in SEEDS for tr in tests], np.float32)
    )
    mt = torch.tensor(mu)
    st = torch.tensor(scale)
    state = net.initial((raw0[:, None, :] - mt) / st)
    preds = [raw0[:, :2].numpy()]
    fail = torch.zeros(m, dtype=torch.bool)
    consistency = []
    with torch.no_grad():
        for i in range(1, len(t)):
            dt = float(t[i] - t[i - 1])
            substeps = max(1, int(math.ceil(dt / MAX_STEP - 1e-8)))
            h = dt / substeps
            for sub in range(substeps):
                ul = (
                    u[:, i - 1 : i]
                    + (u[:, i : i + 1] - u[:, i - 1 : i]) * sub / substeps
                )
                ur = (
                    u[:, i - 1 : i]
                    + (u[:, i : i + 1] - u[:, i - 1 : i]) * (sub + 1) / substeps
                )
                state = net.step(state, ul, ur, h, mt, st)
            raw = net.physical(state) * st + mt
            pred = raw[:, 0, :2]
            bad = (
                (~torch.isfinite(pred).all(1))
                | (pred[:, 0].abs() > 1e4)
                | (pred[:, 1].abs() > 1e3)
            )
            fail |= bad
            value = pred.numpy().copy()
            value[fail.numpy()] = np.nan
            preds.append(value)
            # Diagnostic only: independent d drift is prevented at every accepted state.
            valid = ~fail
            if valid.any():
                consistency.append(
                    float(
                        (raw[valid, 0, 2] - (u[valid, i] - raw[valid, 0, 1]))
                        .abs()
                        .max()
                    )
                )
    return np.stack(preds, axis=1), dict(
        max_relative_speed_identity_error=max(consistency, default=0.0),
        failures=int(fail.sum()),
    )


def metric(pred, truth):
    fail = not np.isfinite(pred).all()
    rm = (
        np.array([1e4, 1e3])
        if fail
        else np.sqrt(np.mean((pred[1:] - truth.z[1:, :2]) ** 2, axis=0))
    )
    return dict(
        gap_rmse_m=float(rm[0]),
        speed_rmse_mps=float(rm[1]),
        score=float(rm[1] + rm[0] / 10),
        failed=int(fail),
        negative_gap=int(np.any(pred[:, 0] < 0)),
        negative_speed=int(np.any(pred[:, 1] < 0)),
    )
