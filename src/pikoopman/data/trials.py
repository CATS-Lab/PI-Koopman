"""CATS-CF encounter records and release-table loading."""
from __future__ import annotations
import csv
from dataclasses import dataclass
import numpy as np

from pikoopman.paths import ROOT

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


def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


