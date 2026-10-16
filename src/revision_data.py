"""Transparent, non-overlapping official OpenACC subset and paired corruption."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from empirical_benchmark import Trial
ROOT=Path(__file__).resolve().parents[1]
def openacc(build=False):
    out=ROOT/'data/external/openacc';records=[];trials=[]
    for path in sorted((out/'raw').glob('*/*.csv')):
        if 'specifications' in path.name:continue
        lines=path.read_text().splitlines();header=next(i for i,l in enumerate(lines) if l.startswith('Time,'))
        meta={l.split(',')[0]:[x for x in l.split(',')[1:] if x] for l in lines[:header]}
        df=pd.read_csv(path,skiprows=header);n=int(meta['Number_of_vehicles'][0])
        for follower in range(2,n+1):
            candidates=[]
            for start in range(0,len(df)-300+1,300):
                block=df.iloc[start:start+300];t=block.Time.to_numpy();t=t-t[0]
                v=block[f'Speed{follower}'].to_numpy();u=block[f'Speed{follower-1}'].to_numpy();s=block[f'IVS{follower-1}'].to_numpy()
                a=np.c_[s,v,u-v];modes=block[f'Driver{follower}'].astype(str).str.strip().unique()
                good=len(modes)==1 and modes[0] in ['ACC','Human'] and np.isfinite(a).all() and np.all(s>0) and np.all(v>=0) and np.all(u>=0) and np.max(abs(np.diff(t)-.1))<1e-4
                if not good:continue
                uid=f'{path.parent.name}_{path.stem}_F{follower}_B{start//300:04d}'
                tr=Trial(uid,path.stem,modes[0],0,np.arange(300)/10,a,u,np.zeros(300))
                vehicle=meta['Vehicle_order'][follower-1]
                rec=dict(uid=uid,campaign=path.parent.name,source_file=path.name,source_row_start=start,source_time_start=float(block.Time.iloc[0]),follower_index=follower,vehicle=vehicle,mode=modes[0],distance_setting=meta.get('Distance_setting',['unknown'])[0],gap_min=float(s.min()),leader_speed_range_mps=float(np.ptp(u)),follower_speed_range_mps=float(np.ptp(v)))
                candidates.append((tr,rec))
            if not candidates:continue
            for j in np.unique(np.rint(np.linspace(0,len(candidates)-1,min(6,len(candidates)))).astype(int)):
                tr,rec=candidates[j];rec['qualifying_blocks_for_file_pair']=len(candidates);trials.append(tr);records.append(rec)
    manifest=pd.DataFrame(records)
    if build:
        manifest.to_csv(out/'episode_manifest.csv',index=False)
        np.savez_compressed(out/'episodes.npz',uid=np.array([tr.uid for tr in trials]),t=trials[0].t,z=np.array([tr.z for tr in trials]),u=np.array([tr.u for tr in trials]))
    return trials,manifest
def crop(tr,a,b,rate=None):
    mask=(tr.t>=a-1e-6)&(tr.t<=b+1e-6);idx=np.flatnonzero(mask)
    if rate is not None:
        native=round(1/np.median(np.diff(tr.t)));stride=int(round(native/rate));assert abs(native/stride-rate)<1e-6
        idx=idx[::stride]
    t=tr.t[idx].copy();t-=t[0]
    return Trial(tr.uid,tr.case,tr.mode,tr.run,t,tr.z[idx].copy(),tr.u[idx].copy(),tr.corr[idx].copy(),tr.top2)
def corrupt(tr,scenario,seed):
    if scenario=='clean':return tr
    stable=int(hashlib.sha256(tr.uid.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(np.random.SeedSequence([seed,stable]));n=len(tr.t)
    errors=rng.normal(size=(n,3));sigma=np.array([2.,.5,.5]) if scenario.endswith('_high') else np.array([.5,.1,.1])
    base_scenario=scenario.replace('_high','')
    if base_scenario=='correlated':
        rho=np.exp(-np.median(np.diff(tr.t))/.3)
        for i in range(1,n):errors[i]=rho*errors[i-1]+np.sqrt(1-rho*rho)*errors[i]
    elif base_scenario!='white':raise ValueError(scenario)
    noise=errors*sigma;s=tr.z[:,0]+noise[:,0];v=tr.z[:,1]+noise[:,1];u=tr.u+noise[:,2]
    return Trial(tr.uid,tr.case,tr.mode,tr.run,tr.t.copy(),np.c_[s,v,u-v],u,tr.corr.copy(),tr.top2)
