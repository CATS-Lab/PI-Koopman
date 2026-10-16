#!/usr/bin/env python3
"""Source-only staged resource amendment to the frozen encounter audit."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import strengthen_encounter_neural as core

OUT=core.OUT/'staged';SCREEN=[73,211];FINAL=[73,211,347,509,863]

def grid(family,budget=None):
    configurations=[dict(c,burnin=1.) for c in core.configs(family)]
    if budget is not None and budget>=8:
        configurations.append(dict(id='large_5s',width=32,horizon=5.,lr=.001,burnin=1.))
    return configurations

def initial(data,family,width,seeds):
    net=core.Network(data['n']*len(seeds),family,width,601)
    for si,seed in enumerate(seeds):
        for ti,tr in enumerate(data['trials']):
            stable=int(hashlib.sha256(f'{tr.uid}|{family}|{width}|{seed}'.encode()).hexdigest()[:12],16)
            block=core.Network(1,family,width,stable)
            with torch.no_grad():
                for dst,src in zip(net.parameters(),block.parameters()):dst[si*data['n']+ti:si*data['n']+ti+1].copy_(src)
    return net

def bundle(ds,b,rate,seeds,uids=None):
    core.SEEDS=seeds;data=core.bundle(ds,b,rate)
    if uids is None:return data
    keep=[i for i,t in enumerate(data['trials']) if t.uid in uids];ix=np.array([si*data['n']+i for si in range(len(seeds)) for i in keep])
    for k in ['z','u','mu','sc']:data[k]=data[k][ix]
    for k in ['x','leader','mt','st']:data[k]=data[k][ix]
    for k in ['trials','train','test']:data[k]=[data[k][i] for i in keep]
    data['n']=len(keep);data['m']=len(ix);return data

def full_loss(net,data,H,burn,starts):
    result=np.zeros(data['m'])
    with torch.no_grad():
        for j in range(0,len(starts),8):result+=core.losses(net,data,H,burn,starts[j:j+8]).numpy()*len(starts[j:j+8])/len(starts)
    return result

def train(data,family,cfg,seeds,dest,maxupdates,saveupdates,resume=None,eval_forecast=True):
    dest.mkdir(parents=True,exist_ok=True);core.SEEDS=seeds;net=initial(data,family,cfg['width'],seeds);opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=1e-5)
    H,burn,starts=core.sequences(data,cfg,family);failed=torch.zeros(data['m'],dtype=torch.bool);begin=0
    if resume:
        arch=torch.load(resume,weights_only=False);net.load_state_dict(arch['state_dict']);opt.load_state_dict(arch['optimizer']);failed=torch.tensor(arch['training_failed']);begin=arch['updates']
    elapsedstart=time.perf_counter();curves=[];rows=[]
    for update in range(begin+1,maxupdates+1):
        for pg in opt.param_groups:pg['lr']=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*min(update,3000)/3000)))
        chosen=starts[(update-1)%len(starts):(update-1)%len(starts)+1]
        opt.zero_grad();lossper=core.losses(net,data,H,burn,chosen);failed|=~torch.isfinite(lossper);loss=torch.where(failed,torch.zeros_like(lossper),lossper).sum();loss.backward();norms=torch.zeros(data['m'])
        for p in net.parameters():
            p.grad.nan_to_num_(nan=0.,posinf=0.,neginf=0.);p.grad[failed]=0.;norms+=p.grad.reshape(data['m'],-1).square().sum(1)
        factors=torch.clamp(10/(norms.sqrt()+1e-9),max=1.)
        for p in net.parameters():p.grad*=factors.reshape(data['m'],*([1]*(p.ndim-1)))
        opt.step()
        if update%100==0:
            for i,value in enumerate(lossper.detach().numpy()):curves.append(dict(updates=update,trial_uid=data['trials'][i%data['n']].uid,neural_seed=seeds[i//data['n']],minibatch_loss=float(value),training_failed=int(failed[i]),window_start_s=float(chosen[0]/data['rate']),minimum_exposures_per_start=update//len(starts),maximum_exposures_per_start=int(math.ceil(update/len(starts)))))
        if update%400==0:print(dest.relative_to(OUT),update,'seconds',round(time.perf_counter()-elapsedstart,1),flush=True)
        if update not in saveupdates:continue
        arch=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),training_failed=failed.numpy(),updates=update,config=cfg,seeds=seeds,uids=[t.uid for t in data['trials']],mu=data['mu'],sc=data['sc'],family=family,seconds_this_stage=time.perf_counter()-elapsedstart)
        path=dest/f'update{update}.pt';torch.save(arch,path)
        ls=full_loss(net,data,H,burn,starts)
        pd.DataFrame([dict(updates=update,trial_uid=data['trials'][i%data['n']].uid,neural_seed=seeds[i//data['n']],loss=float(value)) for i,value in enumerate(ls)]).to_csv(dest/f'calibration_loss{update}.csv',index=False)
        if eval_forecast:
            pred=core.predictions(net,data,burn);pred[failed.numpy(),1:]=np.nan
            replay=initial(data,family,cfg['width'],seeds);replay.load_state_dict(torch.load(path,weights_only=False)['state_dict']);pr=core.predictions(replay,data,burn);pr[failed.numpy(),1:]=np.nan
            assert np.array_equal(pred,pr,equal_nan=True)
            met=pd.DataFrame(core.metrics(pred,data,family,cfg,update)).rename(columns={'epoch':'updates'});met.to_csv(dest/f'metrics{update}.csv',index=False)
            np.savez_compressed(dest/f'predictions{update}.npz',predictions=pred,trial_uid=np.array([t.uid for seed in seeds for t in data['trials']]),seed=np.repeat(seeds,data['n']))
            rows.append(met)
        pd.DataFrame(curves).to_csv(dest/f'training_curve_from{begin}.csv',index=False)
    (dest/f'fit_stage_from{begin}_to{maxupdates}.json').write_text(json.dumps(dict(updates=maxupdates,begin=begin,seeds=seeds,n_trials=data['n'],n_windows=len(starts),training_horizon_s=H/data['rate'],burnin_s=burn/data['rate'],minimum_exposures_per_start=maxupdates//len(starts),maximum_exposures_per_start=int(math.ceil(maxupdates/len(starts))),seconds=time.perf_counter()-elapsedstart,parameter_count=sum(p.numel() for p in net.parameters())//data['m']),indent=2))

def read_candidates(base,family,updates):
    rows=[]
    budget=int(base.name.split('_B')[1].split('_')[0])
    for cfg in grid(family,budget):
        for u in updates:
            p=base/'screen'/family/cfg['id']/f'metrics{u}.csv'
            if p.exists():rows.append(pd.read_csv(p,dtype={'group':str}))
    return pd.concat(rows,ignore_index=True)

def scores(df,group):
    source=df[df.group!=str(group)];return source.groupby(['variant','updates','trial_uid']).score.mean().groupby(['variant','updates']).mean().sort_values()

def old_candidates(ds,b,rate,family):
    section='neural' if family=='DNN' else 'lstm'
    base=core.ROOT/'results/budget_expansion/matched30' if ds=='local' else core.ROOT/'results/revision'
    return pd.read_csv(base/section/f'{ds}_B{b}_Hz{rate}_clean_0/candidate_metrics.csv',dtype={'group':str}).query('model == @family')

def run(ds,b,rate,family):
    base=OUT/f'{ds}_B{b}_Hz{rate}';familyout=base/family;familyout.mkdir(parents=True,exist_ok=True)
    if (familyout/'complete.json').exists():return
    data=bundle(ds,b,rate,SCREEN);groups=sorted({str(t.run) if t.run else t.case for t in data['trials']})
    for cfg in grid(family,b):
        dest=base/'screen'/family/cfg['id']
        if not (dest/'metrics1200.csv').exists():train(data,family,cfg,SCREEN,dest,1200,[400,1200])
    candidates=read_candidates(base,family,[400,1200]);traces=[]
    for g in groups:
        ss=scores(candidates,g);v,u=ss.index[0]
        traces.append(dict(group=g,variant=v,screen_best_updates=int(u),screen_best_source_score=float(ss.iloc[0]),screen_all_scores=[dict(variant=v,updates=int(u),score=float(x)) for (v,u),x in ss.items()],source_groups=[x for x in groups if x!=g]))
    (familyout/'screen_selection_locked.json').write_text(json.dumps(traces,indent=2))
    chosen={t['variant'] for t in traces}
    for cfg in grid(family,b):
        if cfg['id'] not in chosen:continue
        dest=base/'screen'/family/cfg['id']
        if not (dest/'metrics3000.csv').exists():train(data,family,cfg,SCREEN,dest,3000,[3000],dest/'update1200.pt')
    candidates=read_candidates(base,family,[400,1200,3000])
    for trace in traces:
        ss=scores(candidates[candidates.variant==trace['variant']],trace['group']);s3000=float(ss.loc[(trace['variant'],3000)]);oldbest=min(float(ss.loc[(trace['variant'],u)]) for u in [400,1200])
        trace.update(source_score3000=s3000,source_score_best_before3000=oldbest,extension_to6000=bool(s3000<.99*oldbest),extension_relative_gain=float(1-s3000/oldbest))
    (familyout/'extension_decisions_locked.json').write_text(json.dumps(traces,indent=2))
    extensions={t['variant'] for t in traces if t['extension_to6000']}
    for cfg in grid(family,b):
        if cfg['id'] not in extensions:continue
        dest=base/'screen'/family/cfg['id']
        if not (dest/'metrics6000.csv').exists():train(data,family,cfg,SCREEN,dest,6000,[6000],dest/'update3000.pt')
    candidates=read_candidates(base,family,[400,1200,3000,6000]);old=old_candidates(ds,b,rate,family)
    for trace in traces:
        admissible=[400,1200,3000]+([6000] if trace['extension_to6000'] else []);df=candidates[(candidates.variant==trace['variant'])&candidates.updates.isin(admissible)];ss=scores(df,trace['group']);v,u=ss.index[0]
        os=old[old.group!=trace['group']].groupby(['config','trial_uid']).score.mean().groupby('config').mean().sort_values()
        trace.update(admissible_updates=admissible,locked_updates=int(u),locked_source_score=float(ss.iloc[0]),locked_configuration=v,legacy_config=os.index[0],legacy_source_score=float(os.iloc[0]),fallback_family='legacy_2seeds' if os.iloc[0]<=ss.iloc[0] else 'strong_5seeds')
    (familyout/'final_selection_locked_before_five_seed_test.json').write_text(json.dumps(traces,indent=2))
    selected=[];fallback=[]
    for cfg in grid(family,b):
        ts=[t for t in traces if t['variant']==cfg['id']]
        if not ts:continue
        targetgroups={t['group'] for t in ts};uids={t.uid for t in data['trials'] if (str(t.run) if t.run else t.case) in targetgroups};finaldata=bundle(ds,b,rate,FINAL,uids);wanted=sorted({t['locked_updates'] for t in ts});dest=base/'final5'/family/cfg['id']
        if not all((dest/f'metrics{u}.csv').exists() for u in wanted):train(finaldata,family,cfg,FINAL,dest,max(wanted),wanted)
        for trace in ts:
            ff=pd.read_csv(dest/f'metrics{trace["locked_updates"]}.csv',dtype={'group':str});ff=ff[ff.group==trace['group']].copy();ff['variant_source']='strong_5seeds';selected.append(ff)
            if trace['fallback_family']=='strong_5seeds':fallback.append(ff)
            else:
                oo=old[(old.group==trace['group'])&(old.config==trace['legacy_config'])].copy();oo['variant_source']='legacy_2seeds';fallback.append(oo)
    pd.concat(selected).to_csv(familyout/'selected_strong5_metrics.csv',index=False);pd.concat(fallback).to_csv(familyout/'selected_fallback_metrics.csv',index=False)
    candidates.to_csv(familyout/'screen_all_candidate_metrics.csv',index=False)
    # Poisoning target errors cannot change anyof its own decisions.
    poisonchecks=[]
    screen=candidates[candidates.updates.isin([400,1200])]
    for trace in traces:
        g=trace['group'];before=scores(screen,g);bad=screen.copy();bad.loc[bad.group==g,'score']=np.arange(sum(bad.group==g))*1e7-1e10;after=scores(bad,g);assert before.equals(after)
        poisonchecks.append(dict(group=g,screen_target_poison_invariant=True,source_groups_disjoint=g not in trace['source_groups'],admissible_updates=trace['admissible_updates'],selected_updates=trace['locked_updates']))
    (familyout/'selection_audit.json').write_text(json.dumps(poisonchecks,indent=2));(familyout/'complete.json').write_text(json.dumps(dict(dataset=ds,budget=b,family=family,n_groups=len(groups),n_trials=data['n'],training_seeds_screen=SCREEN,training_seeds_final=FINAL),indent=2))


