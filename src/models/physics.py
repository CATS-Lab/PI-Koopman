"""Accelerated physical output-error baselines, independently checked against RK4."""
from pathlib import Path
import ctypes, time
import numpy as np
from scipy.optimize import least_squares
from src.paths import ROOT
import sys
_native = ROOT/'build'/('revision_native.dylib' if sys.platform == 'darwin' else 'revision_native.so')
if not _native.exists():
    raise RuntimeError('Build the native library first: python -m scripts.build_native')
LIB=ctypes.CDLL(str(_native))
PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
LIB.replay.argtypes=[ctypes.c_int,PTR,ctypes.c_int,PTR,PTR,ctypes.c_double,ctypes.c_double,ctypes.c_double,PTR]
LIB.replay.restype=ctypes.c_int
BOUNDS={'CTH':([.0001,.1,0,0],[2,5,30,3]),'IDM':([.05,.05,.1,.1,15],[5,8,5,20,45])}
STARTS={'CTH':[[.15,1.5,3,.5],[.05,2.5,5,.3]],'IDM':[[1,2,1.5,2,30],[2,3,2,5,25]]}
def predict(p,t,u,x0,method,maxdt=.04):
    t,u,p=map(lambda a:np.ascontiguousarray(a,dtype=np.float64),(t,u,p))
    out=np.empty((len(t),2));LIB.replay(0 if method=='CTH' else 1,p,len(t),t,u,float(x0[0]),float(x0[1]),maxdt,out)
    return out
def predict_field(model,t,u,x0,maxdt=.04):
    if model['kind']=='physical':return predict(model['coef'],t,u,x0,'CTH' if model['family']=='cth' else 'IDM',maxdt)
    dict_id=['affine','quadratic','cth','idm','hybrid','tensor27'].index(model['dictionary'])
    prior=model.get('prior');pc=np.zeros(5)
    if prior:pc[:len(prior['coef'])]=prior['coef']
    pid=-1 if not prior else (0 if prior['family']=='cth' else 1)
    packed=np.ascontiguousarray(np.r_[pid,pc,model['zmean'],model['zscale'],model['coef'],model['feature_mean'],model['feature_scale']],dtype=np.float64)
    t,u=map(lambda a:np.ascontiguousarray(a,dtype=np.float64),(t,u));out=np.empty((len(t),2))
    LIB.replay(dict_id+2,packed,len(t),t,u,float(x0[0]),float(x0[1]),maxdt,out)
    return out
def fit(t,z,u,method,max_nfev=400):
    start=time.perf_counter(); audits=[]; models=[]
    def res(p):
        pred=predict(p,t,u,z[0,:2],method)
        if not np.isfinite(pred).all():return np.full(2*len(t),1000.)
        return ((pred-z[:,:2])/[10,1]).ravel()
    for p0 in STARTS[method]:
        opt=least_squares(res,p0,bounds=BOUNDS[method],max_nfev=max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5)
        rss=float(np.sum(res(opt.x)**2));models.append(opt.x)
        finite_calibration=bool(np.isfinite(predict(opt.x,t,u,z[0,:2],method)).all())
        audits.append(dict(rss=rss,success=bool(opt.success) and finite_calibration,optimizer_terminated=bool(opt.success),finite_calibration=finite_calibration,nfev=int(opt.nfev),message=opt.message))
    k=int(np.argmin([a['rss'] for a in audits]))
    return models[k],dict(starts=audits,selected_start=k,success=audits[k]['success'],fit_seconds=time.perf_counter()-start)
