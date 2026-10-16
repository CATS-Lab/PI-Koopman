"""Matched10-Hz neural experiment with exact acquisition/evaluation clock union.

Uses unchanged individual-object networks and source-only staged search. New
artifacts do not overwrite25-Hz archives. No future follower labels are inputs.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from src.data.trials import ROOT, Trial, load_trials
from src.data.openacc import crop
from src.training.prediction import resample
import src.training.encounter as core
import src.training.staged as staged
import src.training.compact as compact
import src.training.lstm as recurrent

OUT=ROOT/'outputs/neural/protocol'
RATE=10


def observed_crop(tr,a,b,rate=None):
    return crop(tr,a,b) if rate is None else resample(tr,a,b,rate)


def clock_union(t,rate):
    native=np.round(np.asarray(t)-t[0],12)
    control=np.round(np.arange(int(np.ceil(native[-1]*rate-1e-8)))/rate,12)
    union=np.unique(np.r_[native,control])
    score=np.searchsorted(union,native)
    is_control=np.isin(union,control)
    assert len(control)==50 and np.array_equal(union[score],native)
    return union,score,is_control


def physical_rollout(net,tests,mu,sc,past,burn,rate=10,training_failed=None):
    n=len(tests);m=len(mu);ns=m//n;t=tests[0].t;grid,score,ticks=clock_union(t,rate)
    u=np.array([np.interp(grid,t,tr.u) for seed in range(ns) for tr in tests])
    raw=np.array([tr.z[0,:2] for seed in range(ns) for tr in tests]);out=[raw.copy()]
    fail=np.zeros(m,dtype=bool) if training_failed is None else np.array(training_failed,dtype=bool)
    memory=None;tick_count=0
    with torch.no_grad():
        for j in range(past.shape[1]-1-burn,past.shape[1]-1):
            if burn:_,memory=net.acceleration(past[:,j:j+1],memory)
        for k in range(1,len(grid)):
            if ticks[k-1]:
                z=np.c_[raw,u[:,k-1]-raw[:,1]]
                x=torch.tensor((z[:,None,:]-mu)/sc,dtype=torch.float32)
                acc,memory=net.acceleration(x,memory);a=acc[:,0].numpy().astype(float);tick_count+=1
            h=grid[k]-grid[k-1];vn=raw[:,1]+h*a
            raw=np.c_[raw[:,0]+.5*h*(u[:,k-1]+u[:,k]-raw[:,1]-vn),vn]
            fail|=(~np.isfinite(raw).all(1))|(np.abs(raw[:,0])>1e4)|(np.abs(raw[:,1])>1e3)
            value=raw.copy();value[fail]=np.nan;out.append(value);raw[fail]=0
    assert tick_count==50
    pred=np.stack(out,1)[:,score];pred[fail,1:]=np.nan
    return pred,dict(recurrent_interval_s=1/rate,recurrent_forecast_updates=tick_count,expected_recurrent_forecast_updates=50,
        past_warmup_updates=burn,reference_samples=len(t),integration_union_samples=len(grid),future_teacher_forcing=False,
        failures=int(fail.sum()),max_relative_speed_identity_error=0.)


def expanded_predictions(net,data,burn):
    return physical_rollout(net,data['test'],data['mu'],data['sc'],data['x'],burn,data['rate'])[0]


def compact_recurrent_rollout(net,trains,tests,mu,sc,training_failed=None):
    ns=len(mu)//len(tests)
    past=np.array([tr.z for seed in range(ns) for tr in trains],np.float32)
    normalized=torch.tensor((past-mu)/sc)
    return physical_rollout(net,tests,mu,sc,normalized,past.shape[1]-1,RATE,training_failed)


def compact_generator_rollout(net,tests,mu,sc):
    n=len(tests);m=len(mu);ns=m//n;t=tests[0].t;grid,score,ticks=clock_union(t,RATE)
    u=np.array([np.interp(grid,t,tr.u) for seed in range(ns) for tr in tests])
    raw=np.array([tr.z[0,:2] for seed in range(ns) for tr in tests]);out=[raw.copy()];fail=np.zeros(m,dtype=bool)
    x0=torch.tensor((np.c_[raw,u[:,0]-raw[:,1]][:,None,:]-mu)/sc,dtype=torch.float32)
    with torch.no_grad():
        state=net.initial(x0);latent=state[:,0,3:].numpy().astype(float) if net.family=='DeepKoopman-projected' else None
        B=net.B[:,0].numpy().astype(float) if latent is not None else None
        tick_count=0
        for k in range(1,len(grid)):
            if ticks[k-1]:
                z=np.c_[raw,u[:,k-1]-raw[:,1]];x=(z[:,None,:]-mu)/sc
                if latent is None:
                    xx=torch.tensor(x,dtype=torch.float32);a=net.out(torch.tanh(net.f2(torch.tanh(net.f1(xx)))))[:,0,0].numpy().astype(float)
                else:
                    xx=torch.tensor(np.concatenate([x,latent[:,None,:]],axis=2),dtype=torch.float32)
                    field=torch.bmm(xx,net.G)[:,0].numpy().astype(float)
                tick_count+=1
            h=grid[k]-grid[k-1]
            if latent is None:vn=raw[:,1]+h*a
            else:
                du=(u[:,k]-u[:,k-1])/sc[:,0,2]
                vn=raw[:,1]+(h*field[:,1]+du*B[:,1])*sc[:,0,1]
                latent=latent+h*field[:,3:]+du[:,None]*B[:,3:]
            raw=np.c_[raw[:,0]+.5*h*(u[:,k-1]+u[:,k]-raw[:,1]-vn),vn]
            fail|=(~np.isfinite(raw).all(1))|(np.abs(raw[:,0])>1e4)|(np.abs(raw[:,1])>1e3)
            value=raw.copy();value[fail]=np.nan;out.append(value);raw[fail]=0
            if latent is not None:latent[fail]=0
    assert tick_count==50
    pred=np.stack(out,1)[:,score];pred[fail,1:]=np.nan
    return pred,dict(acquisition_updates=tick_count,reference_samples=len(t),integration_union_samples=len(grid),failures=int(fail.sum()),max_relative_speed_identity_error=0.)


