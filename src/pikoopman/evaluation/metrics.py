"""Prediction error metrics and weak-support candidate grid."""
import numpy as np

def grid(method,rate,budget):
    Ts=[max(2/rate,x) for x in [.5,1,2]];Ts=sorted({x for x in Ts if x<=budget})
    if method=='fd':Ts=[1]
    return [dict(T=T,ridge=r,kappa=1) for T in Ts for r in [.001,.1,10]]
def metric(pred,truth):
    fail=not np.isfinite(pred).all()
    rmse=np.array([1e4,1e3]) if fail else np.sqrt(np.mean((pred[1:]-truth.z[1:,:2])**2,axis=0))
    return dict(gap_rmse_m=float(rmse[0]),speed_rmse_mps=float(rmse[1]),score=float(rmse[1]+rmse[0]/10),failed=int(fail),negative_gap=int(np.any(pred[:,0]<0)),negative_speed=int(np.any(pred[:,1]<0)))
