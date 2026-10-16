#!/usr/bin/env python3
"""Frozen matched encounter-level LSTM baseline on the primary clean grid.

The recurrent clock follows acquisition intervals. Dense scoring only refines
physical integration; it never adds recurrent updates or calibration targets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.trials import ROOT, Trial, load_trials
from src.data.openacc import crop, openacc

OUT = ROOT/'outputs/models/lstm'
SEEDS = [73, 211]
CHECKPOINTS = [150, 400]
CONDITIONS = [('local', 8, 25), ('openacc', 3, 10), ('openacc', 8, 10), ('openacc', 15, 10)]
torch.set_num_threads(2)


class BatchLinear(nn.Module):
    def __init__(self, m, a, b, generator, scale=None):
        super().__init__()
        scale = 1/math.sqrt(a) if scale is None else scale
        self.w = nn.Parameter(torch.randn(m, a, b, generator=generator)*scale)
        self.b = nn.Parameter(torch.zeros(m, 1, b))

    def forward(self, x):
        return torch.bmm(x, self.w)+self.b


class BatchLSTM(nn.Module):
    """Same eight-unit cell and 393 parameters as the existing short baseline."""
    def __init__(self, m, generator):
        super().__init__()
        self.gates = BatchLinear(m, 11, 32, generator)
        self.out = BatchLinear(m, 8, 1, generator, scale=.02)

    def acceleration(self, x, memory=None):
        if memory is None:
            memory = (x.new_zeros((*x.shape[:2], 8)), x.new_zeros((*x.shape[:2], 8)))
        hh, cc = memory
        i, f, o, j = self.gates(torch.cat([x, hh], dim=-1)).chunk(4, dim=-1)
        cc = torch.sigmoid(f)*cc+torch.sigmoid(i)*torch.tanh(j)
        hh = torch.sigmoid(o)*torch.tanh(cc)
        return self.out(hh)[:, :, 0], (hh, cc)


def initialize(n):
    net = BatchLSTM(n*len(SEEDS), torch.Generator().manual_seed(601))
    for j, seed in enumerate(SEEDS):
        block = BatchLSTM(n, torch.Generator().manual_seed(seed))
        with torch.no_grad():
            for full, piece in zip(net.parameters(), block.parameters()):
                full[j*n:(j+1)*n].copy_(piece)
    assert sum(p.numel() for p in net.parameters())//(n*len(SEEDS))==393
    return net


def advance(state, acceleration, u0, u1, dt, mu, scale):
    raw = state*scale+mu
    s, v, _ = raw.unbind(-1)
    vn = v+dt*acceleration
    sn = s+.5*dt*(u0+u1-v-vn)
    dn = u1-vn
    return (torch.stack([sn, vn, dn], dim=-1)-mu)/scale


def rollout(net, trains, tests, mu, scale, training_failed=None):
    """Past-only warm start; one LSTM update per acquisition interval.

    Acceleration is held over an observation interval. Gap/speed are integrated
    at every reference timestamp using the common dense leader observations.
    """
    n = len(tests); m = n*len(SEEDS)
    t = tests[0].t
    assert all(np.allclose(tr.t, t, atol=1e-9) for tr in tests)
    dt_obs = float(np.median(np.diff(trains[0].t)))
    assert max(np.diff(t)) <= dt_obs+1e-7
    past = np.array([tr.z for seed in SEEDS for tr in trains], dtype=np.float32)
    mt, st = torch.tensor(mu), torch.tensor(scale)
    normalized = torch.tensor((past-mu)/scale)
    u = np.array([tr.u for seed in SEEDS for tr in tests], dtype=np.float64)
    # The physical integration accumulator is float64; network/cell remain
    # float32, matching training. This avoids a dense-grid rounding drift.
    raw = np.array([tr.z[0, :2] for seed in SEEDS for tr in tests], dtype=np.float64)
    fail = np.zeros(m, dtype=bool) if training_failed is None else np.array(training_failed, dtype=bool)
    predictions = [raw.copy()]
    memory = None; tick_count = 0; projection_error = 0.
    with torch.no_grad():
        for j in range(past.shape[1]-1):
            _, memory = net.acceleration(normalized[:, j:j+1], memory)
        for i in range(1, len(t)):
            tick = (t[i-1]-t[0])/dt_obs
            if i==1 or abs(tick-round(tick))<1e-6:
                z = np.c_[raw, u[:, i-1]-raw[:, 1]]
                x = torch.tensor((z[:, None, :]-mu)/scale, dtype=torch.float32)
                acceleration, memory = net.acceleration(x, memory)
                acc = acceleration[:, 0].numpy().astype(np.float64)
                tick_count += 1
            dt = float(t[i]-t[i-1])
            vn = raw[:, 1]+dt*acc
            sn = raw[:, 0]+.5*dt*(u[:, i-1]+u[:, i]-raw[:, 1]-vn)
            raw = np.c_[sn, vn]
            fail |= (~np.isfinite(raw).all(axis=1)) | (np.abs(raw[:, 0])>1e4) | (np.abs(raw[:, 1])>1e3)
            value = raw.copy(); value[fail] = np.nan
            predictions.append(value)
            raw[fail] = 0.  # already failed; prevent meaningless numerical propagation
    prediction = np.stack(predictions, axis=1)
    prediction[fail, 1:] = np.nan
    expected_ticks = int(np.ceil((t[-1]-t[0])/dt_obs-1e-8))
    assert tick_count==expected_ticks
    return prediction, dict(recurrent_interval_s=dt_obs, recurrent_forecast_updates=tick_count,
                            expected_recurrent_forecast_updates=expected_ticks,
                            past_warmup_updates=past.shape[1]-1, future_teacher_forcing=False,
                            reference_samples=len(t), failures=int(fail.sum()),
                            max_relative_speed_identity_error=projection_error)


def metric(pred, truth):
    failed = not np.isfinite(pred).all()
    rm = np.array([1e4, 1e3]) if failed else np.sqrt(np.mean((pred[1:]-truth.z[1:, :2])**2, axis=0))
    return dict(gap_rmse_m=float(rm[0]), speed_rmse_mps=float(rm[1]), score=float(rm[1]+rm[0]/10),
                failed=int(failed), negative_gap=int(np.any(pred[:, 0]<0)), negative_speed=int(np.any(pred[:, 1]<0)))


def select(df):
    selected, choices = [], []
    for group in df.group.unique():
        source = df[df.group!=group]
        target = df[df.group==group]
        scores = source.groupby(['config', 'trial_uid']).score.mean().groupby('config').mean().sort_values()
        cfg = scores.index[0]
        selected.extend(target[target.config==cfg].index.to_list())
        source_groups = sorted(source.group.unique().tolist())
        assert group not in source_groups
        choices.append(dict(group=group, model='LSTM', config=cfg, source_mean_score=float(scores.iloc[0]),
                            source_groups=source_groups, n_source_trials=int(source.trial_uid.nunique()),
                            selection_from='other source groups in this condition'))
    return selected, choices


def tests():
    net = initialize(1)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.out.b.fill_(1.7)
    t = np.arange(126)/125; v = 20+1.7*t; u = np.full(len(t), 22.)
    tr = Trial('unit', 'unit', 'ACC', 1, t, np.c_[30+2*t-.85*t*t, v, u-v], u, np.zeros(len(t)))
    train = crop(tr, 0, .4, 25)
    mu = np.zeros((2, 1, 3), np.float32); scale = np.ones_like(mu)
    pred, audit = rollout(net, [train], [tr], mu, scale)
    expected = tr.z[:, :2]
    error = float(np.max(np.abs(pred-expected)))
    assert error<1e-6
    assert audit['recurrent_forecast_updates']==25 and audit['past_warmup_updates']==10
    poisoned = Trial(tr.uid, tr.case, tr.mode, tr.run, tr.t.copy(), tr.z.copy(), tr.u.copy(), tr.corr.copy())
    poisoned.z[1:] = 999.
    poisoned_pred, _ = rollout(net, [train], [poisoned], mu, scale)
    assert np.array_equal(pred, poisoned_pred)
    normal = initialize(1)
    x = torch.tensor([[[1., 2., 3.]], [[4., 5., 6.]]])
    a, mem = normal.acceleration(x)
    xx = x.clone(); xx[1] += 100
    b, _ = normal.acceleration(xx)
    assert torch.equal(a[0], b[0])
    # Checkpoint selection for a target is unchanged by arbitrary target scores.
    df = pd.DataFrame([dict(group=g, trial_uid=g, config=str(ep), score=float(ep==400))
                       for g in ['a', 'b', 'c'] for ep in CHECKPOINTS])
    _, before = select(df)
    df.loc[df.group=='a', 'score'] = [1e9, -1e9]
    _, after = select(df)
    assert before[0]==after[0]
    x = torch.randn(2, 3, 3); u0 = torch.full((2, 3), 21.); u1 = torch.full((2, 3), 21.2)
    mt = torch.zeros((2, 1, 3)); st = torch.ones((2, 1, 3)); a = torch.full((2, 3), .7)
    nxt = advance(x, a, u0, u1, .1, mt, st)
    assert torch.allclose(nxt[:, :, 2], u1-nxt[:, :, 1], atol=1e-6)
    (OUT/'implementation_tests.json').write_text(json.dumps(dict(
        parameter_count=393, constant_acceleration_rollout_error=error,
        recurrent_clock_matches_acquisition=True, future_target_poison_invariant=True,
        independent_model_blocks=True, target_score_poison_selection_invariant=True,
        exact_relative_speed_projection=True), indent=2))


