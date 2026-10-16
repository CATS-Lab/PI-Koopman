"""Fixed-work single-target DNN/LSTM CPU/CUDA latency, never batched models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import argparse,json,time,hashlib,platform,sys
import numpy as np
import pandas as pd
import torch
from timing_neural_core import Network,advance
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def fit(case,arrays,family,device,updates):
    sync(device);start=time.perf_counter_ns()
    z=arrays[case['id']+'_z'].astype(np.float32)[None]
    u=arrays[case['id']+'_u'].astype(np.float32)[None]
    mu=z.mean(1,keepdims=True);sc=np.maximum(z.std(1,keepdims=True),np.array([1.,.5,.5],np.float32))
    model=Network(1,family,16 if family=='DNN' else 8,73)
    host={'x':torch.tensor((z-mu)/sc),'u':torch.tensor(u),'mu':torch.tensor(mu),'sc':torch.tensor(sc),
          'future_u':torch.tensor(arrays[case['id']+'_future_u'].astype(np.float32)),
          'x0':torch.tensor(arrays[case['id']+'_x0'].astype(np.float32))}
    t0=time.perf_counter_ns();data={k:v.to(device) for k,v in host.items()};model=model.to(device);sync(device)
    transfer_ms=(time.perf_counter_ns()-t0)/1e6
    opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-5)
    rate=case['rate_hz'];H=int(rate);burn=int(rate) if family=='LSTM' else 0
    starts=list(range(burn,data['x'].shape[1]-H,max(1,int(round(.2*rate)))))
    for update in range(1,updates+1):
        idx=starts[(update-1)%len(starts)];mem=None
        for j in range(burn):_,mem=model.acceleration(data['x'][:,idx-burn+j:idx-burn+j+1],mem)
        state=data['x'][:,idx:idx+1];loss=state.new_zeros(())
        for k in range(H):
            acc,mem=model.acceleration(state,mem)
            state=advance(state,acc,data['u'][:,idx+k:idx+k+1],data['u'][:,idx+k+1:idx+k+2],1./rate,data['mu'],data['sc'])
            delta=(state-data['x'][:,idx+k+1:idx+k+2])*data['sc']
            loss=loss+((delta[:,:,0]/10).square()+delta[:,:,1].square()).mean()/2/H
        for pg in opt.param_groups:pg['lr']=.003*(.1+.9*.5*(1+np.cos(np.pi*update/3000)))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
    sync(device);fit_ms=(time.perf_counter_ns()-start)/1e6
    t0=time.perf_counter_ns();mem=None
    with torch.no_grad():
        for j in range(data['x'].shape[1]-1-burn,data['x'].shape[1]-1):
            if burn:_,mem=model.acceleration(data['x'][:,j:j+1],mem)
        fu=data['future_u'];raw=torch.stack([data['x0'][0],data['x0'][1],fu[0]-data['x0'][1]]).reshape(1,1,3)
        state=(raw-data['mu'])/data['sc'];prediction=[raw[0,0,:2]]
        for k in range(len(fu)-1):
            acc,mem=model.acceleration(state,mem)
            state=advance(state,acc,fu[k],fu[k+1],1./rate,data['mu'],data['sc'])
            prediction.append((state*data['sc']+data['mu'])[0,0,:2])
        prediction=torch.stack(prediction).cpu().numpy()
    sync(device);rollout_ms=(time.perf_counter_ns()-t0)/1e6;total_ms=(time.perf_counter_ns()-start)/1e6
    return dict(fit_including_initialization_transfer_ms=fit_ms,input_and_model_transfer_ms=transfer_ms,
                rollout_5s_including_download_ms=rollout_ms,total_fit_rollout_ms=total_ms,
                parameter_count=sum(p.numel() for p in model.parameters()),finite=bool(np.isfinite(prediction).all()),
                final_training_loss=float(loss.detach().cpu())),prediction

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--label',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cases=json.loads((root/'data/timing/cases.json').read_text());arrays=np.load(root/'data/timing/state_arrays.npz');device=torch.device(a.device)
    chosen=[next(c for c in cases if c['dataset']=='openacc' and c['budget_s']==b) for b in [3,8,15]]
    meta=dict(device=str(device),hardware_label=a.label,python=sys.version,platform=platform.platform(),
              torch=torch.__version__,torch_cuda_version=torch.version.cuda,torch_threads=torch.get_num_threads(),
              torch_interop_threads=torch.get_num_interop_threads(),dtype='float32',models_per_fit=1,
              case_ids=[c['id'] for c in chosen],budgets_s=[3,8,15],rate_hz=10,updates=400,repetitions=3,
              warmup_updates=20,seed=73,DNN_width=16,LSTM_width=8,training_horizon_s=1,LSTM_warmup_s=1,
              optimizer='Adam lr0.003 cosine decay over3000 updates, weight_decay1e-5, gradient norm cap10',
              loss='0.5*((gap_error/10)^2+speed_error^2), mean over1s rollout',
              context='Frozen 400-update workload measures computation; it is not a convergence or accuracy comparison.',
              includes='Each fit is independent and includes train-only normalization, model creation, model/input transfer, Adam initialization,400updates and5srollout with output transfer; device synchronized at boundaries.',
              excludes='One-time framework import/CUDA context startup and20-update warmup; offline configuration search/validation excluded.',
              input_sha256=hashlib.sha256((root/'data/timing/state_arrays.npz').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    (out/'protocol_runtime.json').write_text(json.dumps(meta,indent=2)+'\n')
    rows=[];preds={};start=time.perf_counter()
    for case in chosen:
        for family in ['DNN','LSTM']:
            fit(case,arrays,family,device,20)
            for repeat in range(3):
                row,pred=fit(case,arrays,family,device,400)
                rows.append(dict(case=case['id'],budget_s=case['budget_s'],model=family,repeat=repeat,**row))
                preds[f"{case['id']}_{family}_{repeat}"]=pred
            pd.DataFrame(rows).to_csv(out/'samples.csv',index=False)
            print(a.label,case['budget_s'],family,round(time.perf_counter()-start,2),flush=True)
    np.savez_compressed(out/'predictions.npz',**preds)
    df=pd.DataFrame(rows);s=[]
    for (b,m),g in df.groupby(['budget_s','model']):
        d=dict(hardware=a.label,device=str(device),budget_s=b,model=m,repeats=len(g),updates=400)
        for col in ['fit_including_initialization_transfer_ms','input_and_model_transfer_ms','rollout_5s_including_download_ms','total_fit_rollout_ms']:
            d[col+'_median']=g[col].median();d[col+'_min']=g[col].min();d[col+'_max']=g[col].max()
        s.append(d)
    pd.DataFrame(s).to_csv(out/'summary.csv',index=False)
    (out/'completion.json').write_text(json.dumps(dict(completed=True,elapsed_s=time.perf_counter()-start),indent=2)+'\n')

if __name__=='__main__':main()
