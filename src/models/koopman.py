"""Input-explicit physics-library generator regression and autonomous legacy tests.

Public API: fit(t,z,u,method,config) -> dict, predict(model,t,u,x0,corr=None).
z columns are [gap, follower speed, leader minus follower speed].
No autonomous Koopman interpretation is asserted for the arbitrary-input fit.
"""
from pathlib import Path
import argparse, hashlib, json, time, warnings
import numpy as np
import pandas as pd
from scipy.linalg import logm, expm
from scipy.optimize import least_squares

from src.paths import ROOT
OUT=ROOT/'outputs/models/koopman'


def physical(z,p,family):
    s,v,d=np.asarray(z).T
    if family=='cth':
        a,tau,s0,b=p
        return a*(s-s0-tau*v)+b*d
    a,b,tau,s0,v0=p
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        star=s0+np.maximum(0,v*tau-v*d/(2*np.sqrt(a*b)))
        return a*(1-(v/v0)**4-(star/s)**2)


def raw_features(z,dictionary):
    z=np.atleast_2d(z);s,v,d=z.T
    base=[np.ones(len(z)),s,v,d]
    if dictionary=='affine': pass
    elif dictionary=='quadratic': base += [s*s,v*v,d*d,s*v,s*d,v*d]
    elif dictionary=='tensor27':
        # Exact old tensor exponents, expressed in [s,v,u] as in the paper.
        u=d  # tensor27 callers pass [s,v,u], not [s,v,d]
        return np.column_stack([s**i*v**j*u**k for k in range(3) for j in range(3) for i in range(3)])
    elif dictionary=='cth':
        e=(s-3-1.5*v)/10
        base += [e*e,e*d,d*d,np.tanh(d),e*np.abs(e)]
    elif dictionary in ['idm','hybrid']:
        # On the active desired-headway branch this includes the exact IDM
        # monomials after expanding (s0+tau*v-v*d/(2sqrt(ab)))^2/s^2.
        # Clipped desired-headway switching is NOT exactly spanned.
        si=1/s**2
        base += [(v/30)**4,si,v*si,v*v*si,v*d*si,v*v*d*si,v*v*d*d*si]
        if dictionary=='hybrid':
            e=(s-3-1.5*v)/10
            base += [e*e,e*d,np.tanh(d)]
    else: raise ValueError(dictionary)
    return np.column_stack(base)


def features(z,model):
    # Scaling physical input before polynomial expansion avoids overflow and
    # improves old tensor conditioning, while preserving its polynomial span.
    zz=np.atleast_2d(z)
    if model['dictionary']=='tensor27':
        zz=np.column_stack([zz[:,0],zz[:,1],zz[:,1]+zz[:,2]])
        zz=(zz-model['zmean'])/model['zscale']
    return (raw_features(zz,model['dictionary'])-model['feature_mean'])/model['feature_scale']


def _prepare(z,dictionary):
    zz=np.column_stack([z[:,0],z[:,1],z[:,1]+z[:,2]]) if dictionary=='tensor27' else z
    model={'dictionary':dictionary,'zmean':zz.mean(0),'zscale':np.maximum(zz.std(0),[1,.5,.5])}
    zz=(zz-model['zmean'])/model['zscale'] if dictionary=='tensor27' else zz
    F=raw_features(zz,dictionary); mean=F.mean(0); scale=F.std(0)
    mean[0]=0;scale[0]=1
    scale=np.where(scale>1e-8,scale,1.)
    model.update(feature_mean=mean,feature_scale=scale)
    return model


def acceleration(model,z):
    z=np.atleast_2d(z)
    if model['kind']=='physical':return physical(z,model['coef'],model['family'])
    ans=features(z,model)@model['coef']
    if model.get('prior') is not None:ans+=physical(z,model['prior']['coef'],model['prior']['family'])
    return ans


def predict(model,t,u,x0,corr=None,max_dt=.1):
    """RK4 conditional free rollout; no speed/gap/acceleration clipping.

    Future input interpolates supplied u only; initial measured x0 is common.
    Failed trajectories are returned as NaN and must remain in evaluation.
    """
    t=np.asarray(t);u=np.asarray(u);corr=np.zeros_like(t) if corr is None else np.asarray(corr)
    p=np.full((len(t),2),np.nan);p[0]=np.asarray(x0)[:2]
    def rhs(x,uu,cc):
        if model['kind']=='physical':a=physical(np.array([[x[0],x[1],uu-x[1]]]),model['coef'],model['family'])[0]
        else:a=acceleration(model,[[x[0],x[1],uu-x[1]]])[0]
        return np.array([uu-x[1]+cc,a])
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        for j in range(1,len(t)):
            n=max(1,int(np.ceil((t[j]-t[j-1])/max_dt)));h=(t[j]-t[j-1])/n;x=p[j-1].copy()
            for k in range(n):
                q=k/n;uu=u[j-1]*(1-q)+u[j]*q;cc=corr[j-1]*(1-q)+corr[j]*q
                un=u[j-1]*(1-(k+1)/n)+u[j]*(k+1)/n;cn=corr[j-1]*(1-(k+1)/n)+corr[j]*(k+1)/n
                a=rhs(x,uu,cc);b=rhs(x+h*a/2,(uu+un)/2,(cc+cn)/2);c=rhs(x+h*b/2,(uu+un)/2,(cc+cn)/2);d=rhs(x+h*c,un,cn)
                x=x+h*(a+2*b+2*c+d)/6
                if not np.isfinite(x).all() or abs(x[0])>1e4 or abs(x[1])>1e3:return p
            p[j]=x
    return p


def weak_rows(t,F,v,method,T=1.,kappa=1.):
    """Discrete SBP response exactly annihilates constants and reproduces a
    constant acceleration on EVERY grid. All finite endpoints are retained.
    """
    t=np.asarray(t);A=[];y=[]
    if method=='fd':return (F[1:]+F[:-1])/2,np.diff(v)/np.diff(t)
    n=max(2,int(round(T/np.median(np.diff(t))))+1);n=min(n,len(t))
    if method in ['weak','expweak'] and n<3:
        # Compact support cannot exist with endpoints only; return explicit
        # unavailable status via error rather than silently relabeling integral.
        raise ValueError('Compact weak kernel requires >=3 samples')
    for j in range(0,len(t)-n+1,max(1,(n-1)//2)):
        tt=t[j:j+n];q=(tt-tt[0])/(tt[-1]-tt[0])
        w=np.ones(n)
        if method in ['weak','expweak']:w=16*q*q*(1-q)**2
        if method in ['expweak','finiteexp']:w=w*np.exp(-kappa*q)
        w=w/np.trapezoid(w,tt)
        A.append(np.trapezoid(w[:,None]*F[j:j+n],tt,axis=0))
        y.append(np.sum((w[1:]+w[:-1])/2*np.diff(v[j:j+n])))
    return np.asarray(A),np.asarray(y)


def _physical_fit(t,z,u,family,config):
    if family=='cth':bounds=([.001,.1,0,0],[2,5,20,3]);starts=[[.12,1.5,3,.5],[.05,2.5,5,.3]]
    else:bounds=([.05,.05,.1,.1,15],[5,8,5,20,45]);starts=[[1,2,1.5,2,30],[2,3,2,5,40]]
    # Fast integral initializer, followed by a true free output-error fit.
    A=np.empty((0,0));dt=np.diff(t)
    def integ_res(p):return dt*(physical(z[:-1],p,family)+physical(z[1:],p,family))/2-np.diff(z[:,1])
    initials=[least_squares(integ_res,p,bounds=bounds,max_nfev=70) for p in starts]
    init=min(initials,key=lambda r:r.cost)
    m={'kind':'physical','family':family,'coef':init.x,'prior':None}
    def oe_res(p):
        m['coef']=p;P=predict(m,t,u,z[0,:2],corr=config.get('corr'),max_dt=.2)
        if not np.isfinite(P).all():return np.full(len(t)*2,1e4)
        return ((P-z[:,:2])/np.array([10.,1.])).ravel()
    opt=least_squares(oe_res,init.x,bounds=bounds,max_nfev=config.get('max_nfev',35),ftol=1e-5,xtol=1e-5,gtol=1e-5)
    m.update(coef=opt.x.copy(),optimizer_success=bool(opt.success),optimizer_nfev=int(opt.nfev),cost=float(opt.cost))
    return m


def fit(t,z,u,method='weak',config=None):
    """Fit only supplied calibration data; all dictionary scalers are local.

    methods: fd, integral, weak, expweak, finiteexp, CTH-OE, IDM-OE.
    config: dictionary (affine/quadratic/tensor27/cth/idm/hybrid), ridge,
    T, kappa, prior ('cth'/'idm'/None), prior_model (optional fitted train-only
    model), corr (optional known training-coordinate correction).
    Penalizing residual coefficients shrinks toward a fitted physical prior;
    this is physical-prior regularization, not evidence from an external run.
    """
    config={} if config is None else config.copy();t=np.asarray(t,float);z=np.asarray(z,float);u=np.asarray(u,float)
    assert len(t)==len(z)==len(u) and z.shape[1]==3
    assert np.all(np.diff(t)>0) and np.isfinite(z).all()
    assert np.allclose(z[:,2],u-z[:,1],atol=1e-5)
    start=time.perf_counter()
    if method in ['CTH-OE','IDM-OE']:m=_physical_fit(t,z,u,'cth' if method=='CTH-OE' else 'idm',config)
    else:
        m=_prepare(z,config.get('dictionary','cth'));m.update(kind='field',prior=None)
        if config.get('prior'):
            m['prior']=config.get('prior_model')
            if m['prior'] is None:m['prior']=_physical_fit(t,z,u,config['prior'],config)
        F=features(z,m);A,y=weak_rows(t,F,z[:,1],method,config.get('T',1.),config.get('kappa',1.))
        if m['prior'] is not None:
            AF,_=weak_rows(t,physical(z,m['prior']['coef'],m['prior']['family'])[:,None],z[:,1],method,config.get('T',1.),config.get('kappa',1.));y=y-AF[:,0]
        ridge=config.get('ridge',.1);H=A.T@A/len(y)+ridge*np.eye(F.shape[1]);b=A.T@y/len(y)
        m['coef']=np.linalg.lstsq(H,b,rcond=1e-12)[0]
        ss=np.linalg.svd(A/np.sqrt(len(A)),compute_uv=False)
        m.update(rows=len(y),singular_values=ss,rank=int(np.linalg.matrix_rank(A)),effective_rank_1e3=int(np.sum(ss>ss[0]*1e-3)),condition=float(ss[0]/ss[-1]) if ss[-1]>0 else np.inf)
    m.update(method=method,config={k:v for k,v in config.items() if k not in ['prior_model','corr']},fit_seconds=time.perf_counter()-start,training_min=z.min(0),training_max=z.max(0))
    return m


def unit_checks():
    rows=[]
    for rate in [1,5,25]:
        for T in [1,2,4]:
            t=np.arange(int(T*rate)+1)/rate;F=np.ones((len(t),1))
            for method in ['fd','integral','weak','expweak','finiteexp']:
                try:
                    A,y=weak_rows(t,F,20+1.7*t,method,T,3.)
                    _,c=weak_rows(t,F,np.full(len(t),20.),method,T,3.)
                except ValueError:continue
                rows.append(dict(rate_hz=rate,T_s=T,method=method,constant_error=float(np.max(abs(c))),acceleration_error=float(np.max(abs(y/A[:,0]-1.7)))))
    assert max(r['acceleration_error'] for r in rows)<1e-12
    assert max(r['constant_error'] for r in rows)==0
    return rows


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(unit_checks()).to_csv(OUT/'sbp_checks.csv',index=False)
    print('SBP checks passed; API ready')
