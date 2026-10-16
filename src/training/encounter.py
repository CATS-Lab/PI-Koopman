#!/usr/bin/env python3
"""Frozen stronger, independent-encounter neural identification experiment."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.data.trials import ROOT, load_trials
from src.data.openacc import crop, openacc

OUT=ROOT/'outputs/neural/encounter'
SEEDS=[73,211,347,509,863]
CHECKPOINTS=[400,1200,3000]
CONDITIONS=[(ds,b,25 if ds=='local' else 10) for ds in ['local','openacc'] for b in [3,8,15]]
torch.set_num_threads(2)

def configs(family):
    small,large=(16,32) if family=='DNN' else (8,32)
    return [dict(id='small_short',width=small,horizon=.4,lr=.003,burnin=1.),
            dict(id='small_1s',width=small,horizon=1.,lr=.003,burnin=1.),
            dict(id='large_2s',width=large,horizon=2.,lr=.003,burnin=1.),
            dict(id='large_2s_slow',width=large,horizon=2.,lr=.001,burnin=3.)]

class Linear(nn.Module):
    def __init__(self,m,a,b,g,scale=None):
        super().__init__(); self.w=nn.Parameter(torch.randn(m,a,b,generator=g)*(1/math.sqrt(a) if scale is None else scale)); self.b=nn.Parameter(torch.zeros(m,1,b))
    def forward(self,x):return torch.bmm(x,self.w)+self.b

class Network(nn.Module):
    def __init__(self,m,family,width,seed):
        super().__init__();self.family=family;self.width=width;g=torch.Generator().manual_seed(seed)
        if family=='DNN':self.f1=Linear(m,3,width,g);self.f2=Linear(m,width,width,g)
        else:self.gates=Linear(m,3+width,4*width,g)
        self.out=Linear(m,width,1,g,.02)
    def acceleration(self,x,mem=None):
        if self.family=='DNN':return self.out(torch.tanh(self.f2(torch.tanh(self.f1(x)))))[:,:,0],None
        if mem is None:mem=(x.new_zeros((*x.shape[:2],self.width)),x.new_zeros((*x.shape[:2],self.width)))
        hh,cc=mem;i,f,o,j=self.gates(torch.cat([x,hh],-1)).chunk(4,-1)
        cc=torch.sigmoid(f)*cc+torch.sigmoid(i)*torch.tanh(j);hh=torch.sigmoid(o)*torch.tanh(cc)
        return self.out(hh)[:,:,0],(hh,cc)

def initialize(n,family,width):
    net=Network(n*len(SEEDS),family,width,601)
    for j,seed in enumerate(SEEDS):
        block=Network(n,family,width,seed)
        with torch.no_grad():
            for dest,src in zip(net.parameters(),block.parameters()):dest[j*n:(j+1)*n].copy_(src)
    return net

def advance(x,acc,u0,u1,dt,mu,sc):
    z=x*sc+mu;s,v,d=z.unbind(-1);vn=v+dt*acc;sn=s+.5*dt*(u0+u1-v-vn)
    return (torch.stack([sn,vn,u1-vn],-1)-mu)/sc

def bundle(dataset,budget,rate):
    trials,path=load_trials(30,125) if dataset=='local' else (openacc()[0],ROOT/'data/external/openacc/episode_manifest.csv')
    train=[crop(t,20-budget,20,rate) for t in trials];test=[crop(t,20,25) for t in trials]
    z=np.array([t.z for seed in SEEDS for t in train],dtype=np.float32);u=np.array([t.u for seed in SEEDS for t in train],dtype=np.float32)
    mu=z.mean(1,keepdims=True);sc=np.maximum(z.std(1,keepdims=True),np.array([1.,.5,.5],np.float32))
    return dict(trials=trials,path=path,train=train,test=test,z=z,u=u,mu=mu,sc=sc,x=torch.tensor((z-mu)/sc),leader=torch.tensor(u),mt=torch.tensor(mu),st=torch.tensor(sc),n=len(trials),m=len(z),rate=rate,budget=budget)

def sequences(data,cfg,family):
    H=int(round(cfg['horizon']*data['rate']));burn=int(round(min(cfg['burnin'],data['budget']-cfg['horizon'])*data['rate'])) if family=='LSTM' else 0
    starts=np.arange(burn,data['x'].shape[1]-H,max(1,int(round(.2*data['rate']))))
    if len(starts)==0:raise ValueError('No accessible training window')
    return H,burn,starts

def losses(net,data,H,burn,starts):
    ix=torch.tensor(starts,dtype=torch.long);mem=None
    for j in range(burn):_,mem=net.acceleration(data['x'][:,ix-burn+j],mem)
    state=data['x'][:,ix];result=torch.zeros(data['m'])
    for k in range(H):
        acc,mem=net.acceleration(state,mem)
        state=advance(state,acc,data['leader'][:,ix+k],data['leader'][:,ix+k+1],1/data['rate'],data['mt'],data['st'])
        delta=(state-data['x'][:,ix+k+1])*data['st']
        result+=((delta[:,:,0]/10).square()+delta[:,:,1].square()).mean(1)/2/H
    return result

def predictions(net,data,burn):
    tests=data['test'];t=tests[0].t;u=np.array([tr.u for seed in SEEDS for tr in tests]);raw=np.array([tr.z[0,:2] for seed in SEEDS for tr in tests]);m=data['m'];mem=None;failed=np.zeros(m,dtype=bool);pred=[raw.copy()]
    with torch.no_grad():
        for j in range(data['x'].shape[1]-1-burn,data['x'].shape[1]-1):
            if burn:_,mem=net.acceleration(data['x'][:,j:j+1],mem)
        for k in range(1,len(t)):
            tick=(t[k-1]-t[0])*data['rate']
            if k==1 or abs(tick-round(tick))<1e-6:
                z=np.c_[raw,u[:,k-1]-raw[:,1]];x=torch.tensor((z[:,None,:]-data['mu'])/data['sc'],dtype=torch.float32)
                acc,mem=net.acceleration(x,mem);a=acc[:,0].numpy().astype(np.float64)
            dt=t[k]-t[k-1];vn=raw[:,1]+dt*a;sn=raw[:,0]+.5*dt*(u[:,k-1]+u[:,k]-raw[:,1]-vn);raw=np.c_[sn,vn]
            failed|=(~np.isfinite(raw).all(1))|(np.abs(raw[:,0])>1e4)|(np.abs(raw[:,1])>1e3)
            val=raw.copy();val[failed]=np.nan;pred.append(val);raw[failed]=0
    return np.stack(pred,1)

def metrics(pred,data,family,cfg,epoch):
    rows=[];n=data['n']
    for si,seed in enumerate(SEEDS):
        for i,tr in enumerate(data['trials']):
            p=pred[si*n+i];fail=not np.isfinite(p).all();rm=np.array([1e4,1e3]) if fail else np.sqrt(np.mean((p[1:]-data['test'][i].z[1:,:2])**2,axis=0))
            rows.append(dict(trial_uid=tr.uid,case_id=tr.case,mode=tr.mode,group=str(tr.run) if tr.run else tr.case,model=family,variant=cfg['id'],epoch=epoch,neural_seed=seed,gap_rmse_m=rm[0],speed_rmse_mps=rm[1],score=rm[1]+rm[0]/10,failed=int(fail),negative_gap=int(np.any(p[:,0]<0)),negative_speed=int(np.any(p[:,1]<0))))
    return rows


