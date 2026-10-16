#!/usr/bin/env python3
"""Frozen run-wise benchmark of effective follower-speed fields.
Data are read-only. No pointwise acceleration ground truth is assumed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import minimize, least_squares
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
RIDGES = [1e-6, 1e-3, .1, 10.]
MODELS = ['CV','Affine-integral','Quadratic-integral','FD-quadratic','Weak-quadratic','ExpWeak1-quadratic','ExpWeak3-quadratic','FiniteExp-quadratic','ConstrainedWeak-quadratic','Delay-affine','MLP8-FD','CTH-RV-OE','IDM-OE']
FAIL_SCORE = 1e6

@dataclass
class Trial:
    uid: str
    case: str
    mode: str
    run: int
    t: np.ndarray
    z: np.ndarray
    u: np.ndarray
    corr: np.ndarray
    top2: bool = False

    def prefix(self, seconds):
        keep = self.t <= seconds + 1e-9
        return Trial(self.uid,self.case,self.mode,self.run,self.t[keep],self.z[keep],self.u[keep],self.corr[keep],self.top2)


def load_trials(duration=15, rate=25):
    path = ROOT/f'data/release/{duration}s/koopman_ready/koopman_{rate}hz.csv'
    manifest = {}
    with (path.parents[1]/'metadata/cv_fold_manifest.csv').open() as f:
        manifest = {r['trial_uid']:r['include_in_top2_quality_subset']=='True' for r in csv.DictReader(f)}
    groups = {}
    with path.open() as f:
        for r in csv.DictReader(f): groups.setdefault(r['trial_uid'],[]).append(r)
    trials=[]
    for uid,rows in groups.items():
        arr=lambda k:np.array([float(r[k]) for r in rows])
        t=arr('t_rel_s'); s=arr('gap_m'); v=arr('follower_longitudinal_velocity_mps'); u=arr('leader_velocity_in_follower_longitudinal_mps')
        z=np.column_stack([s,v,u-v]); corr=arr('follower_yaw_rate_corrimu_radps')*arr('lateral_offset_left_m')
        trials.append(Trial(uid,rows[0]['case_id'],rows[0]['follower_mode'],int(rows[0]['run_index']),t,z,u,corr,manifest[uid]))
    assert len(trials)==42 and len({x.case for x in trials})==14
    assert all(np.all(np.diff(x.t)>0) and np.isfinite(x.z).all() for x in trials)
    return trials,path


def safejson(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,MLPRegressor): return {'coefs':[x.tolist() for x in value.coefs_],'intercepts':[x.tolist() for x in value.intercepts_]}
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value))


def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


