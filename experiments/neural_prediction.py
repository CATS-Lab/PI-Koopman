"""Complete matched neural grid; unchanged source-only staged search.

Each declared cell independently repeats compact solver/checkpoint candidates,
expanded staged search and target-group-excluded selection. CUDA graph replay
changes execution, not the optimizer updates, starts, loss, or admissibility.
"""

import sys
from pathlib import Path
import argparse,copy,hashlib,inspect,json,math,time
import numpy as np
import pandas as pd
import torch
from src.data.trials import ROOT, load_trials
from src.data.openacc import crop, openacc
from src.training.prediction import resample
import src.training.protocol as adapter
import src.training.encounter as core
import src.training.staged as staged
import src.training.compact as compact
import src.training.lstm as recurrent
import src.models.neural as continuous

OUT=ROOT/'outputs/neural'
ORIGIN=20; RATE=10; DEVICE='cpu'; USE_GRAPH=False
BASE_BUNDLE=core.bundle
BASE_TRAIN=staged.train
BASE_PHYSICAL=inspect.getsource(adapter.physical_rollout)
BASE_GENERATOR=inspect.getsource(adapter.compact_generator_rollout)
BASE_COMPACT_RUN=inspect.getsource(compact.run)
BASE_RECURRENT_RUN=inspect.getsource(recurrent.run)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    p=dict(origins_s=[15,20,24],histories_s=[3,8,15],history_axis_rate_hz=10,
        rate_axis_history_s=8,rate_axis_hz=[1,5,10,25,125],rate_axis_dataset='local',
        future_s=5,cohorts='42 local and284 OpenACC including existing source modes; primary ACC24/236.',
        old_cell_reuse='Exact existing source-selected origin20,10Hz results. New cells use same full selection procedure.',
        selection='Each target run/file excluded entirely. Every new cell repeats compact150/400, both valid numerical solvers, expanded400/1200 screening,source-only configuration,3000 and own-source conditional6000, lockedfive-seedrefit. No target error chooses origin/rate/configuration/checkpoint/seed.',
        expanded_configs={str(b):{f:staged.grid(f,b) for f in ['DNN','LSTM']} for b in [3,8,15]},
        seeds_screen=[73,211],seeds_final=[73,211,347,509,863],
        physical_time='Expanded rollout horizons0.4/1/2/5s and LSTM1s burnin;0.2s startspacing. Minimum1observed interval when0.4s inaccessible at1Hz. Compact0.4s horizon/stride,minimumoneobservedinterval; compact DNN/DeepKoopman/LSTM discount0.9**(k*10/rate), preserving0.1s physical-time discount at every rate and exactly unchanged10Hz.',
        observations='Same crop/resample as physical grid, within history only; native125Hz local and10Hz OpenACC scoretimes. Future follower labels never enter rollout.',
        numerics='Compact both held acquisition-clock field and continuous<=0.04s substeps; source chooses. Expanded/LSTM exact acquisition updates and union-grid kinematics. No hidden native calibration labels.',
        failures='Retain every candidate and all seeds, fixed1000m/s and10000m errors; no failed row dropped.',
        cuda_graph='Optional exact repeated optimizer-step graph, same deterministic starts, Adam,cosineLR and per-modelclip. Float32,TF32off. CPU/GPU numericalcheck archived.',
        inputs={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/training/staged.py',ROOT/'src/training/encounter.py',ROOT/'src/training/protocol.py',ROOT/'src/training/compact.py',ROOT/'src/training/lstm.py']})
    path=OUT/'protocol_frozen.json'
    if path.exists():
        old=json.loads(path.read_text());assert {k:v for k,v in old.items() if k!='inputs'}=={k:v for k,v in p.items() if k!='inputs'}
    else:path.write_text(json.dumps(p,indent=2))


def clock_union(t,rate):
    native=np.round(np.asarray(t)-t[0],12)
    control=np.round(np.arange(int(round(native[-1]*rate)))/rate,12)
    union=np.unique(np.r_[native,control]);score=np.searchsorted(union,native)
    assert len(control)==5*rate and np.array_equal(union[score],native)
    return union,score,np.isin(union,control)


def shifted_crop(tr,a,b,rate=None):
    offset=ORIGIN-20
    return crop(tr,a+offset,b+offset) if rate is None else resample(tr,a+offset,b+offset,rate)


def device_bundle(ds,b,rate):
    d=BASE_BUNDLE(ds,b,rate)
    for key in ['x','leader','mt','st']:d[key]=d[key].to(DEVICE)
    return d


def sequences(data,cfg,family):
    H=max(1,int(round(cfg['horizon']*data['rate'])))
    burn=max(0,int(round(min(cfg['burnin'],data['budget']-H/data['rate'])*data['rate']))) if family=='LSTM' else 0
    starts=np.arange(burn,data['x'].shape[1]-H,max(1,int(round(.2*data['rate']))))
    assert len(starts)>0
    return H,burn,starts


def losses(net,data,H,burn,starts):
    ix=starts if torch.is_tensor(starts) else torch.tensor(starts,dtype=torch.long,device=DEVICE)
    mem=None
    for j in range(burn):_,mem=net.acceleration(data['x'][:,ix-burn+j],mem)
    state=data['x'][:,ix];result=state.new_zeros(data['m'])
    for k in range(H):
        acc,mem=net.acceleration(state,mem)
        state=core.advance(state,acc,data['leader'][:,ix+k],data['leader'][:,ix+k+1],1/data['rate'],data['mt'],data['st'])
        delta=(state-data['x'][:,ix+k+1])*data['st']
        result+=((delta[:,:,0]/10).square()+delta[:,:,1].square()).mean(1)/2/H
    return result


def cpu_prediction(net,data,burn):
    host=copy.copy(data);host['x']=data['x'].cpu()
    cpu=copy.deepcopy(net).cpu()
    return adapter.physical_rollout(cpu,host['test'],host['mu'],host['sc'],host['x'],burn,host['rate'])[0]


def full_loss(net,data,H,burn,starts):
    out=np.zeros(data['m'])
    with torch.no_grad():
        for j in range(0,len(starts),8):out+=losses(net,data,H,burn,starts[j:j+8]).cpu().numpy()*len(starts[j:j+8])/len(starts)
    return out


def train(data,family,cfg,seeds,dest,maxupdates,saveupdates,resume=None,eval_forecast=True):
    """Original staged training with device support and optional CUDA graphs."""
    dest.mkdir(parents=True,exist_ok=True);core.SEEDS=seeds
    net=staged.initial(data,family,cfg['width'],seeds).to(DEVICE)
    H,burn,starts=sequences(data,cfg,family)
    lr=torch.tensor(cfg['lr'],device=DEVICE) if USE_GRAPH else cfg['lr']
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-5,capturable=USE_GRAPH,foreach=False)
    failed=torch.zeros(data['m'],dtype=torch.bool,device=DEVICE);begin=0
    if resume:
        arch=torch.load(resume,map_location=DEVICE,weights_only=False);net.load_state_dict(arch['state_dict']);opt.load_state_dict(arch['optimizer']);failed.copy_(torch.as_tensor(arch['training_failed'],device=DEVICE));begin=arch['updates']
        if USE_GRAPH:opt.param_groups[0]['lr']=lr
    elapsed=time.perf_counter();curves=[]
    chosen=torch.tensor([starts[begin%len(starts)]],dtype=torch.long,device=DEVICE)
    def step():
        opt.zero_grad(set_to_none=False);per=losses(net,data,H,burn,chosen)
        failed.logical_or_(~torch.isfinite(per));loss=torch.where(failed,torch.zeros_like(per),per).sum();loss.backward()
        norms=per.new_zeros(data['m'])
        for p in net.parameters():
            p.grad.nan_to_num_(nan=0.,posinf=0.,neginf=0.);p.grad.mul_((~failed).reshape(data['m'],*([1]*(p.ndim-1))));norms+=p.grad.reshape(data['m'],-1).square().sum(1)
        fac=torch.clamp(10/(norms.sqrt()+1e-9),max=1.)
        for p in net.parameters():p.grad*=fac.reshape(data['m'],*([1]*(p.ndim-1)))
        opt.step();return per
    graph=None
    if USE_GRAPH:
        state=copy.deepcopy(net.state_dict());optim=copy.deepcopy(opt.state_dict());fail0=failed.clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream().wait_stream(stream);net.load_state_dict(state);failed.copy_(fail0)
        if begin:opt.load_state_dict(optim);opt.param_groups[0]['lr']=lr
        else:
            for state in opt.state.values():
                for v in state.values():
                    if torch.is_tensor(v):v.zero_()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):per=step()
        torch.cuda.synchronize()
    for update in range(begin+1,maxupdates+1):
        rate=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*min(update,3000)/3000)))
        if USE_GRAPH:lr.fill_(rate)
        else:opt.param_groups[0]['lr']=rate
        chosen.fill_(int(starts[(update-1)%len(starts)]))
        if graph:graph.replay()
        else:per=step()
        if update%100==0:
            vals=per.detach().cpu().numpy();fs=failed.cpu().numpy()
            for i,v in enumerate(vals):curves.append(dict(updates=update,trial_uid=data['trials'][i%data['n']].uid,neural_seed=seeds[i//data['n']],minibatch_loss=float(v),training_failed=int(fs[i]),window_start_s=float(starts[(update-1)%len(starts)]/data['rate'])))
        if update%400==0:print(dest.relative_to(OUT),update,'seconds',round(time.perf_counter()-elapsed,1),flush=True)
        if update not in saveupdates:continue
        fs=failed.cpu().numpy();arch=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),training_failed=fs,updates=update,config=cfg,seeds=seeds,uids=[t.uid for t in data['trials']],mu=data['mu'],sc=data['sc'],family=family,seconds_this_stage=time.perf_counter()-elapsed)
        torch.save(arch,dest/f'update{update}.pt')
        ls=full_loss(net,data,H,burn,starts)
        pd.DataFrame([dict(updates=update,trial_uid=data['trials'][i%data['n']].uid,neural_seed=seeds[i//data['n']],loss=float(v)) for i,v in enumerate(ls)]).to_csv(dest/f'calibration_loss{update}.csv',index=False)
        if eval_forecast:
            pred=cpu_prediction(net,data,burn);pred[fs,1:]=np.nan
            met=pd.DataFrame(core.metrics(pred,data,family,cfg,update)).rename(columns={'epoch':'updates'});met.to_csv(dest/f'metrics{update}.csv',index=False)
            np.savez_compressed(dest/f'predictions{update}.npz',predictions=pred,trial_uid=np.array([t.uid for seed in seeds for t in data['trials']]),seed=np.repeat(seeds,data['n']))
        pd.DataFrame(curves).to_csv(dest/f'training_curve_from{begin}.csv',index=False)
    (dest/f'fit_stage_from{begin}_to{maxupdates}.json').write_text(json.dumps(dict(updates=maxupdates,begin=begin,seeds=seeds,n_trials=data['n'],n_windows=len(starts),training_horizon_s=H/data['rate'],burnin_s=burn/data['rate'],seconds=time.perf_counter()-elapsed,device=DEVICE,cuda_graph=USE_GRAPH),indent=2))


def configure(ds,b,rate,origin,device='cpu',graph=False):
    global RATE,ORIGIN,DEVICE,USE_GRAPH
    RATE=rate;ORIGIN=origin;DEVICE=device;USE_GRAPH=graph
    if str(device).startswith('cuda'):torch.cuda.set_device(torch.device(device))
    cell=OUT/'grid'/f'{ds}_B{b}_Hz{rate}_end{origin}';cell.mkdir(parents=True,exist_ok=True)
    adapter.OUT=cell;adapter.RATE=rate;adapter.clock_union=clock_union
    # Generalize only the fixed10Hz assertions/defaults of the audited union integrator.
    namespace=adapter.__dict__
    exec(BASE_PHYSICAL.replace('rate=10','rate=None').replace('assert tick_count==50','assert tick_count==5*rate').replace('expected_recurrent_forecast_updates=50','expected_recurrent_forecast_updates=5*rate'),namespace)
    exec(BASE_GENERATOR.replace('assert tick_count==50','assert tick_count==5*RATE'),namespace)
    core.OUT=cell;core.crop=shifted_crop;core.bundle=device_bundle;core.losses=losses;core.sequences=sequences;core.predictions=cpu_prediction
    core.CONDITIONS=[(ds,b,rate)];staged.OUT=cell/'staged';staged.OUT.mkdir(exist_ok=True);staged.train=train;staged.full_loss=full_loss
    staged.old_candidates=lambda ds,b,rate,family:pd.read_csv(cell/'compact'/('neural' if family=='DNN' else 'lstm')/f'{ds}_B{b}_Hz{rate}_clean_0/candidate_metrics.csv',dtype={'group':str}).query('model == @family')
    compact.OUT=cell/'compact/neural';compact.crop=shifted_crop;compact.rollout=adapter.compact_generator_rollout;compact.MAX_STEP=1/rate
    recurrent.OUT=cell/'compact/lstm';recurrent.crop=shifted_crop;recurrent.rollout=adapter.compact_recurrent_rollout
    # Retain nonfinite compact objects independently, exactly like recurrent/expanded fits.
    source=BASE_COMPACT_RUN.replace("npar=sum(p.numel() for p in net.parameters())//m", "npar=sum(p.numel() for p in net.parameters())//m;training_failed=torch.zeros(m,dtype=torch.bool)").replace('.9**k','.9**(k*10/rate)')
    source=source.replace("loss=loss_per.sum()/H\n            if not torch.isfinite(loss):raise RuntimeError(f'{tag} {family} nonfinite calibration loss at epoch{epoch}')", "training_failed|=~torch.isfinite(loss_per)\n            loss=torch.where(training_failed,torch.zeros_like(loss_per),loss_per).sum()/H")
    source=source.replace("for p in net.parameters():norms+=p.grad.reshape(m,-1).square().sum(1)", "for p in net.parameters():\n                p.grad.nan_to_num_(nan=0.,posinf=0.,neginf=0.);p.grad[training_failed]=0.;norms+=p.grad.reshape(m,-1).square().sum(1)")
    source=source.replace("pred,audit=rollout(net,test,mu,scale);consistency.append", "pred,audit=rollout(net,test,mu,scale);pred[training_failed.numpy(),1:]=np.nan;audit['failures']=int((~np.isfinite(pred).all((1,2))).sum());consistency.append")
    exec(source,compact.__dict__)
    # At1Hz the shortest supported training rollout is one observed interval.
    exec(BASE_RECURRENT_RUN.replace('horizon = int(round(.4*rate)); stride = horizon','horizon = max(1, int(round(.4*rate))); stride = horizon').replace('.9**k','.9**(k*10/rate)'),recurrent.__dict__)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return cell


def report_cell(ds,b,rate,origin,cell):
    pieces=[];choices=[]
    candidates=[]
    for folder,variant in [('compact/neural','clock'),('continuous','continuous0.04')]:
        c=pd.read_csv(cell/folder/f'{ds}_B{b}_Hz{rate}_clean_0/candidate_metrics.csv',dtype={'group':str});c['numerical_variant']=variant;candidates.append(c)
    c=pd.concat(candidates,ignore_index=True)
    trace=json.loads((cell/'staged'/f'{ds}_B{b}_Hz{rate}'/'DNN/final_selection_locked_before_five_seed_test.json').read_text())
    strong=pd.read_csv(cell/'staged'/f'{ds}_B{b}_Hz{rate}'/'DNN/selected_strong5_metrics.csv',dtype={'group':str})
    for group in sorted(c.group.unique()):
        for family in ['DNN','DeepKoopman-projected']:
            src=c[c.group.ne(group)&c.model.eq(family)]
            ss=src.groupby(['numerical_variant','config','trial_uid']).score.mean().groupby(['numerical_variant','config']).mean().sort_values(kind='stable')
            numerical,cfg=ss.index[0];score=float(ss.iloc[0]);expanded=np.inf
            if family=='DNN':expanded=float(next(t for t in trace if str(t['group'])==group)['locked_source_score'])
            if score<=expanded:
                chosen=c[c.group.eq(group)&c.model.eq(family)&c.numerical_variant.eq(numerical)&c.config.eq(cfg)].copy();branch='compact'
            else:chosen=strong[strong.group.eq(group)].copy();branch='expanded';chosen['numerical_variant']='clock'
            chosen['variant_source']=branch;pieces.append(chosen)
            choices.append(dict(group=group,model=family,branch=branch,compact_variant=numerical,compact_config=cfg,compact_source_score=score,expanded_source_score=expanded if np.isfinite(expanded) else None,source_groups=sorted(src.group.unique()),target_excluded=group not in src.group.unique()))
    pieces.append(pd.read_csv(cell/'staged'/f'{ds}_B{b}_Hz{rate}'/'LSTM/selected_fallback_metrics.csv',dtype={'group':str}))
    d=pd.concat(pieces,ignore_index=True);d['dataset']=ds;d['budget_s']=b;d['rate_hz']=rate;d['origin_s']=origin
    d.to_csv(cell/'selected_raw_initializations.csv',index=False)
    enc=d.groupby(['dataset','budget_s','rate_hz','origin_s','trial_uid','case_id','mode','group','model'],as_index=False).agg(speed_rmse_mps=('speed_rmse_mps','mean'),gap_rmse_m=('gap_rmse_m','mean'),score=('score','mean'),failed=('failed','max'),n_seeds=('neural_seed','size'))
    enc.to_csv(cell/'selected_encounter_metrics.csv',index=False);(cell/'source_choices.json').write_text(json.dumps(choices,indent=2))
    (cell/'complete.json').write_text(json.dumps(dict(complete=True,n_encounter_models=len(enc),n_raw_initializations=len(d)),indent=2))


def fix_compact_metadata(cell,b,rate,origin):
    for folder in ['compact/neural','compact/lstm','continuous']:
        for p in (cell/folder).glob('*/config.json'):
            d=json.loads(p.read_text());audit=d.get('training_audit',d)
            audit.update(train_start_s=origin-b,train_end_s=origin,training_horizon_s=max(1,round(.4*rate))/rate)
            d.update(prediction_origin_s=origin,future_interval_s=[origin,origin+5],source_protocol=str(OUT/'protocol_frozen.json'))
            p.write_text(json.dumps(d,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['local','openacc'],default='local');p.add_argument('--budget',type=int,default=8);p.add_argument('--rate',type=int,default=10);p.add_argument('--origin',type=int,default=15);p.add_argument('--device',default='cpu');p.add_argument('--graph',action='store_true');p.add_argument('--stage',choices=['all','compact','expanded','report','pilot'],default='all');p.add_argument('--family',choices=['DNN','LSTM']);p.add_argument('--pilot-updates',type=int,default=50);a=p.parse_args()
    frozen_protocol();cell=configure(a.dataset,a.budget,a.rate,a.origin,a.device,a.graph)
    if a.stage=='pilot':
        data=staged.bundle(a.dataset,a.budget,a.rate,staged.SCREEN);family=a.family or 'LSTM';cfg=staged.grid(family,a.budget)[0]
        start=time.perf_counter();train(data,family,cfg,staged.SCREEN,cell/f'pilot_{family}_{a.device.replace(":","_")}_{a.graph}',a.pilot_updates,[a.pilot_updates],eval_forecast=False)
        print(json.dumps(dict(seconds=time.perf_counter()-start,updates=a.pilot_updates,models=data['m'],device=a.device,graph=a.graph)),flush=True);return
    if a.stage in ['all','compact']:
        compact.run(a.dataset,a.budget,a.rate);recurrent.run(a.dataset,a.budget,a.rate)
        compact.OUT=cell/'continuous';compact.MAX_STEP=.04;compact.rollout=continuous.rollout
        compact.run(a.dataset,a.budget,a.rate)
        fix_compact_metadata(cell,a.budget,a.rate,a.origin)
    if a.stage in ['all','expanded']:
        for family in ([a.family] if a.family else ['DNN','LSTM']):staged.run(a.dataset,a.budget,a.rate,family)
    if a.stage in ['all','report']:report_cell(a.dataset,a.budget,a.rate,a.origin,cell)

if __name__=='__main__':main()
