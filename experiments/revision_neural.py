#!/usr/bin/env python3
"""Matched individual-object neural robustness/external validation.
DNN and projected DeepKoopman are new rate-aware adaptations, not paper reproductions.
No shared learned weights or normalization across different local objects.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from empirical_benchmark import ROOT,load_trials,write_csv
from revision_data import openacc,crop,corrupt
OUT=ROOT/'results/revision/neural';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
SEEDS=[73,211];FAMILIES=['DNN','DeepKoopman-projected'];DEFAULT_CHECKPOINTS=[150,400]
MAX_STEP=.04

class BatchLinear(nn.Module):
    def __init__(self,m,a,b,generator,scale=None):
        super().__init__();scale=1/math.sqrt(a) if scale is None else scale
        self.w=nn.Parameter(torch.randn(m,a,b,generator=generator)*scale);self.b=nn.Parameter(torch.zeros(m,1,b))
    def forward(self,x):return torch.bmm(x,self.w)+self.b

class BatchNetwork(nn.Module):
    def __init__(self,m,family,generator):
        super().__init__();self.family=family;self.latent=6
        self.f1=BatchLinear(m,3,16,generator);self.f2=BatchLinear(m,16,16,generator)
        self.out=BatchLinear(m,16,6 if family=='DeepKoopman-projected' else 1,generator,scale=.02)
        if family=='DeepKoopman-projected':
            self.G=nn.Parameter(torch.randn(m,9,9,generator=generator)*.001)
            self.B=nn.Parameter(torch.zeros(m,1,9));self.B.data[:,:,2]=1.
    def encode(self,x):
        h=self.out(torch.tanh(self.f2(torch.tanh(self.f1(x)))))
        return torch.cat([x,h],dim=-1)
    def initial(self,x):return self.encode(x) if self.family=='DeepKoopman-projected' else x
    def physical(self,state):return state[:,:,:3] if self.family=='DeepKoopman-projected' else state
    def step(self,state,u0,u1,dt,mu,scale):
        """One small generator step with exact relative-speed projection and trapezoid gap."""
        x=self.physical(state);raw=x*scale+mu;s,v,d=raw.unbind(-1)
        if self.family=='DeepKoopman-projected':
            du=(u1-u0)/scale[:,:,2]
            candidate=state+dt*torch.bmm(state,self.G)+du[:,:,None]*self.B
            vn=candidate[:,:,1]*scale[:,:,1]+mu[:,:,1]
            h=candidate[:,:,3:]
        else:
            acc=self.out(torch.tanh(self.f2(torch.tanh(self.f1(x)))))[:,:,0];vn=v+dt*acc
        sn=s+.5*dt*((u0-v)+(u1-vn));dn=u1-vn
        physical=(torch.stack([sn,vn,dn],dim=-1)-mu)/scale
        return torch.cat([physical,h],dim=-1) if self.family=='DeepKoopman-projected' else physical


def initialize(n,family):
    net=BatchNetwork(n*len(SEEDS),family,torch.Generator().manual_seed(601))
    for j,seed in enumerate(SEEDS):
        block=BatchNetwork(n,family,torch.Generator().manual_seed(seed))
        with torch.no_grad():
            for full,piece in zip(net.parameters(),block.parameters()):full[j*n:(j+1)*n].copy_(piece)
    return net


def rollout(net,tests,mu,scale):
    n=len(tests);m=n*len(SEEDS);t=tests[0].t
    assert all(np.array_equal(tr.t,t) for tr in tests)
    u=torch.tensor(np.array([tr.u for seed in SEEDS for tr in tests],np.float32))
    raw0=torch.tensor(np.array([tr.z[0] for seed in SEEDS for tr in tests],np.float32))
    mt=torch.tensor(mu);st=torch.tensor(scale);state=net.initial((raw0[:,None,:]-mt)/st)
    preds=[raw0[:,:2].numpy()];fail=torch.zeros(m,dtype=torch.bool)
    consistency=[]
    with torch.no_grad():
        for i in range(1,len(t)):
            dt=float(t[i]-t[i-1]);substeps=max(1,int(math.ceil(dt/MAX_STEP-1e-8)));h=dt/substeps
            for sub in range(substeps):
                ul=u[:,i-1:i]+(u[:,i:i+1]-u[:,i-1:i])*sub/substeps
                ur=u[:,i-1:i]+(u[:,i:i+1]-u[:,i-1:i])*(sub+1)/substeps
                state=net.step(state,ul,ur,h,mt,st)
            raw=net.physical(state)*st+mt
            pred=raw[:,0,:2];bad=(~torch.isfinite(pred).all(1))|(pred[:,0].abs()>1e4)|(pred[:,1].abs()>1e3);fail|=bad
            value=pred.numpy().copy();value[fail.numpy()]=np.nan;preds.append(value)
            # Diagnostic only: independent d drift is prevented at every accepted state.
            valid=~fail
            if valid.any():consistency.append(float((raw[valid,0,2]-(u[valid,i]-raw[valid,0,1])).abs().max()))
    return np.stack(preds,axis=1),dict(max_relative_speed_identity_error=max(consistency,default=0.),failures=int(fail.sum()))


def metric(pred,truth):
    fail=not np.isfinite(pred).all();rm=np.array([1e4,1e3]) if fail else np.sqrt(np.mean((pred[1:]-truth.z[1:,:2])**2,axis=0))
    return dict(gap_rmse_m=float(rm[0]),speed_rmse_mps=float(rm[1]),score=float(rm[1]+rm[0]/10),failed=int(fail),negative_gap=int(np.any(pred[:,0]<0)),negative_speed=int(np.any(pred[:,1]<0)))


def tests():
    torch.manual_seed(1);n=2;mu=np.zeros((4,1,3),np.float32);scale=np.ones((4,1,3),np.float32);x=torch.tensor([[[30.,18.,2.]],[[40.,17.,1.]],[[30.,18.,2.]],[[40.,17.,1.]]]);u0=x[:,:,1]+x[:,:,2];u1=u0+.04
    result=[]
    for family in FAMILIES:
        net=initialize(n,family);state=net.initial(x);out=net.step(state,u0,u1,.04,torch.tensor(mu),torch.tensor(scale));physical=net.physical(out)
        assert torch.allclose(physical[:,:,2],u1-physical[:,:,1],atol=2e-6)
        expected=x[:,:,0]+.5*.04*((u0-x[:,:,1])+(u1-physical[:,:,1]));assert torch.allclose(physical[:,:,0],expected,atol=2e-6)
        # Each model block is independent of the inputs to another object.
        poison=x.clone();poison[1]+=100.;state2=net.initial(poison);out2=net.step(state2,u0,u1,.04,torch.tensor(mu),torch.tensor(scale));assert torch.equal(out[0],out2[0])
        result.append(dict(family=family,relative_speed_identity=True,trapezoid_spacing_identity=True,independent_object_parameter_blocks=True))
    write_csv(OUT/'implementation_tests.csv',result);print(result,flush=True)


