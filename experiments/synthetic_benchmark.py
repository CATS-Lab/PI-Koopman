"""Fixed synthetic tests; no experimental dataset is used for design or tuning.

Compare the same quadratic dictionary across differentiation/integration choices.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.linalg import expm,logm
from scipy.integrate import cumulative_trapezoid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/synthetic';OUT.mkdir(parents=True,exist_ok=True)

def accel(q,nonlinear):
    s,v,d=np.moveaxis(np.asarray(q),-1,0);e=s-4-1.3*v
    return .12*e+.55*d+(0.002*e**2-0.008*d**2 if nonlinear else 0)

def simulate(seed,n,nonlinear):
    rng=np.random.default_rng(seed);dt=.008;t=np.arange(2501)*dt
    base=rng.uniform(14,24,n);phase=rng.uniform(0,2*np.pi,(2,n))
    def leader(tt):return base+1.5*np.sin(.45*tt+phase[0])+.5*np.sin(1.2*tt+phase[1])
    v0=leader(0)+rng.uniform(-3,3,n);s0=4+1.3*v0+rng.uniform(-7,7,n)
    x=np.empty((len(t),n,2));x[0]=np.c_[s0,v0]
    def f(tt,xx):
        d=leader(tt)-xx[:,1]
        return np.c_[d,accel(np.c_[xx,d],nonlinear)]
    for i,tt in enumerate(t[:-1]):
        z=x[i];k1=f(tt,z);k2=f(tt+dt/2,z+dt*k1/2);k3=f(tt+dt/2,z+dt*k2/2);k4=f(tt+dt,z+dt*k3)
        x[i+1]=z+dt*(k1+2*k2+2*k3+k4)/6
    u=np.array([leader(tt) for tt in t]);return np.concatenate([x,(u-x[:,:,1])[:,:,None]],axis=2)

def phi(q,mu,sd):
    a,b,c=np.moveaxis((q-mu)/sd,-1,0)
    return np.stack([np.ones_like(a),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c],axis=-1)

def regress(tracks,rate,method):
    allq=tracks.reshape(-1,3);mu=allq.mean(0);sd=np.maximum(allq.std(0),1e-6)
    feat=phi(tracks,mu,sd);dt=1/rate;As=[];ys=[]
    if method=='FD':
        As=[(feat[1:]+feat[:-1]).reshape(-1,10)/2]
        ys=[np.diff(tracks[:,:,1],axis=0).reshape(-1)/dt]
    else:
        m=int(4*rate);r=np.arange(m+1)/m
        w=np.ones(m+1) if method=='Integral' else 16*r*r*(1-r)**2
        if method=='ExpWeak':w*=np.exp(-3*r)
        w/=np.trapezoid(w,dx=dt)
        for i in range(0,len(tracks)-m,max(1,m//2)):
            As.append(np.trapezoid(w[:,None,None]*feat[i:i+m+1],dx=dt,axis=0))
            # Discrete summation by parts: constant signals annihilated exactly.
            ys.append(np.sum((w[1:]+w[:-1])[:,None]/2*np.diff(tracks[i:i+m+1,:,1],axis=0),axis=0))
    A=np.concatenate(As);y=np.concatenate(ys);pen=np.eye(10)*1e-6;pen[0,0]=0
    coef=np.linalg.solve(A.T@A/len(A)+pen,A.T@y/len(A))
    return mu,sd,coef


def main():
    start=time.perf_counter();rows=[]
    for nonlinear in [False,True]:
        truth=simulate(109+int(nonlinear),18,nonlinear)
        test=simulate(909+int(nonlinear),12,nonlinear)[::25].reshape(-1,3)
        for rate in [1,5,25]:
            base=truth[::int(125/rate)]
            for noise in [0,.05,.15]:
                for seed in range(8):
                    rng=np.random.default_rng(2000+seed)
                    noisy=base.copy()
                    # Independent sensor errors in s, v and u; reconstruct d=u-v.
                    noisy[:,:,0]+=rng.normal(0,2*noise,base.shape[:2])
                    ev=rng.normal(0,noise,base.shape[:2]);eu=rng.normal(0,noise,base.shape[:2])
                    noisy[:,:,1]+=ev;noisy[:,:,2]+=eu-ev
                    for method in ['FD','Integral','Weak','ExpWeak']:
                        mu,sd,c=regress(noisy,rate,method);pred=phi(test,mu,sd)@c
                        rows.append(dict(system='quadratic' if nonlinear else 'affine',rate_hz=rate,speed_noise_sd_mps=noise,seed=seed,
                          method=method,field_rmse_mps2=np.sqrt(np.mean((pred-accel(test,nonlinear))**2))))
    d=pd.DataFrame(rows);d.to_csv(OUT/'field_error.csv',index=False)
    d.groupby(['system','rate_hz','speed_noise_sd_mps','method']).field_rmse_mps2.agg(['mean','std']).reset_index().to_csv(OUT/'summary.csv',index=False)
    (OUT/'config.json').write_text(json.dumps(dict(seed_initial_train=[109,110],seed_initial_test=[909,910],noise_seeds=list(range(2000,2008)),
       training_trajectories=18,test_trajectories=12,duration_s=20,integration_dt_s=.008,rate_hz=[1,5,25],window_s=4,
       ridge=1e-6,degree=2,noise_note='iid sensor measurement error; s SD twice speed SD; d computed from noisy u-v',
       evaluation='clean independent trajectory support; acceleration-vector-field RMSE, not rollout',
       python=sys.version,elapsed_s=time.perf_counter()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(d.groupby(['method']).field_rmse_mps2.agg(['mean','max']).to_string())

if __name__=='__main__':main()
