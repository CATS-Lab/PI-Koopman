#!/usr/bin/env python3
"""30-s parent/end20 adapter; fitting and prediction rules are unchanged.
The run function below differs from revision_neural.run only in parent
release duration and calibration endpoint. The architectures, training,
selection, saved model schemas, and scoring functions are imported unchanged.
"""
from revision_neural import *
OUT = ROOT/'results/budget_expansion/matched30/neural'

def run(dataset,budget,rate,scenario='clean',corruption_seed=0,checkpoints=DEFAULT_CHECKPOINTS,pilot=False,force=False,frozen_selection_path=None):
    tag=f'{dataset}_B{budget}_Hz{rate}_{scenario}_{corruption_seed}';out=OUT/(tag+'_pilot' if pilot else tag);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.csv').exists() and not force:print('Already complete',tag,flush=True);return
    if dataset=='local':trials,path=load_trials(30,125);meta=None;end=20
    else:trials,meta=openacc();path=ROOT/'data/external/openacc/episode_manifest.csv';end=20
    if pilot:trials=trials[:14];meta=None if meta is None else meta.iloc[:14].copy()
    # Corrupt the native parent once before decimation, exactly as the physical benchmark.
    observed=[corrupt(tr,scenario,corruption_seed) for tr in trials]
    train=[crop(tr,end-budget,end,rate) for tr in observed];test=[crop(tr,end,end+5) for tr in trials]
    n=len(train);raw=np.array([tr.z for seed in SEEDS for tr in train],np.float32);observed_u=np.array([tr.u for seed in SEEDS for tr in train],np.float32)
    mu=raw.mean(1,keepdims=True);scale=np.maximum(raw.std(1,keepdims=True),np.array([1.,.5,.5],np.float32));X=(raw-mu)/scale
    # Fixed .4 s physical training horizon where observable. At1Hz, the minimum is1s.
    H=max(1,int(round(.4*rate)));stride=max(1,int(round(.4*rate)));starts=np.arange(0,raw.shape[1]-H,stride)
    target=np.stack([X[:,j:j+H+1] for j in starts],axis=1);inputs=np.stack([observed_u[:,j:j+H+1] for j in starts],axis=1)
    tar=torch.tensor(target);inp=torch.tensor(inputs);mt=torch.tensor(mu);st=torch.tensor(scale)
    dt=float(np.median(np.diff(train[0].t)));substeps=max(1,int(math.ceil(dt/MAX_STEP-1e-8)));h=dt/substeps
    rows=[];predictions=[];lossrows=[];consistency=[];all_started=time.perf_counter()
    training_audit=dict(n_trials=n,train_samples=raw.shape[1],train_start_s=end-budget,train_end_s=end,observed_rate_hz=rate,observed_step_s=dt,internal_substeps_per_observed_interval=substeps,internal_step_s=h,training_horizon_samples=H,training_horizon_s=H*dt,n_sequences_per_trial=len(starts),clean_reference_forecast_step_s=float(np.median(np.diff(test[0].t))),clean_reference_forecast_samples=len(test[0].t),no_hidden_high_rate_calibration_targets=True)
    config=dict(dataset=dataset,budget_s=budget,rate_hz=rate,scenario=scenario,corruption_seed=corruption_seed,neural_initializations=SEEDS,checkpoints=checkpoints,families=FAMILIES,architecture='individual-object width16x16 tanh DNN and6latent projected controlled DeepKoopman; independent parameters for every trial/init',koopman_scope='new Tian-style adaptation with raw s,v,d coordinates and learned latent, linear generator plus observed leader increments; enforce d=u-v and kinematic gap; not exact paper reproduction or exact finite-dimensional invariant closure',training='Adam lr0.003 weight_decay1e-5, loss sum over independent objects; per-object gradient norm cap10; physical normalized multi-step error plus0.1 latent consistency forDK',prediction='same clean dense observed future leader and clean observed initial state as primary physical benchmark; Euler generator substeps<=.04, trapezoid spacing; no follower teacher forcing',normalization='own calibration window mean/std only; floors1/.5/.5',selection='other run index(local) or other source-file stem(OpenACC); average neural seeds per source trial; pick checkpoint only; retain both seeds',failure='nonfinite or |gap|>1e4 or |speed|>1e3 stops trial; same fixed1e4m/1e3m/s penalties',training_audit=training_audit,frozen_selection_path=str(frozen_selection_path) if frozen_selection_path else None,data_file=str(path),data_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for family in FAMILIES:
        net=initialize(n,family);opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=1e-5);started=time.perf_counter();m=n*len(SEEDS);npar=sum(p.numel() for p in net.parameters())//m
        for epoch in range(1,max(checkpoints)+1):
            opt.zero_grad();state=net.initial(tar[:,:,0]);loss_per=torch.zeros(m)
            for k in range(H):
                for sub in range(substeps):
                    ul=inp[:,:,k]+(inp[:,:,k+1]-inp[:,:,k])*sub/substeps
                    ur=inp[:,:,k]+(inp[:,:,k+1]-inp[:,:,k])*(sub+1)/substeps
                    state=net.step(state,ul,ur,h,mt,st)
                err=(net.physical(state)-tar[:,:,k+1]).square().mean((1,2))
                if family=='DeepKoopman-projected':
                    reference=net.encode(tar[:,:,k+1]);err=err+.1*(state[:,:,3:]-reference[:,:,3:]).square().mean((1,2))
                loss_per+=.9**k*err
            loss=loss_per.sum()/H
            if not torch.isfinite(loss):raise RuntimeError(f'{tag} {family} nonfinite calibration loss at epoch{epoch}')
            loss.backward();norms=torch.zeros(m)
            for p in net.parameters():norms+=p.grad.reshape(m,-1).square().sum(1)
            fac=torch.clamp(10/(norms.sqrt()+1e-9),max=1.)
            for p in net.parameters():p.grad*=fac.reshape(m,*([1]*(p.ndim-1)))
            opt.step()
            if epoch%50==0:print(tag,family,'epoch',epoch,'train_loss',float(loss.detach()/m),'seconds',round(time.perf_counter()-started,1),flush=True)
            if epoch not in checkpoints:continue
            fit_seconds=time.perf_counter()-started
            torch.save(dict(state_dict=net.state_dict(),mu=mu,scale=scale,uids=[tr.uid for tr in trials],neural_seeds=SEEDS,family=family,epoch=epoch,config=config),out/f'{family}_epoch{epoch}.pt')
            pred,audit=rollout(net,test,mu,scale);consistency.append(dict(family=family,epoch=epoch,**audit))
            cfg=json.dumps(dict(epoch=epoch,width=16,latent=6 if family=='DeepKoopman-projected' else 0,max_internal_step_s=MAX_STEP,projection='d=u-v,gap_trapezoid',training_horizon_s=H*dt),sort_keys=True)
            for init_index,seed in enumerate(SEEDS):
                for i,tr in enumerate(trials):
                    base=dict(dataset=dataset,trial_uid=tr.uid,case_id=tr.case,mode=tr.mode,group=str(tr.run) if dataset=='local' else tr.case,budget_s=budget,rate_hz=rate,scenario=scenario,seed=corruption_seed,neural_seed=seed,model=family,config=cfg,parameters=npar,fit_seconds=fit_seconds/(n*len(SEEDS)),batch_fit_seconds=fit_seconds,optimizer_success=True)
                    if meta is not None:base.update(campaign=str(meta.iloc[i].campaign),vehicle=str(meta.iloc[i].vehicle))
                    rows.append(dict(**base,**metric(pred[init_index*n+i],test[i])));predictions.append(pred[init_index*n+i])
                    lossrows.append(dict(trial_uid=tr.uid,model=family,neural_seed=seed,epoch=epoch,train_loss=float(loss_per[init_index*n+i].detach()/H),batch_fit_seconds=fit_seconds))
            print(tag,family,'saved epoch',epoch,'prediction failures',audit['failures'],flush=True)
        write_csv(out/'candidate_metrics.csv',rows);np.savez_compressed(out/'candidate_predictions.npz',predictions=np.array(predictions));write_csv(out/'training_loss.csv',lossrows)
    df=pd.DataFrame(rows);selected=[];choices=[]
    for group in df.group.unique():
        source=df[df.group!=group];target=df[df.group==group];assert len(source)>0
        for family in FAMILIES:
            scores=source[source.model==family].groupby(['config','trial_uid']).score.mean().groupby('config').mean().sort_values()
            if frozen_selection_path:
                frozen=json.loads(Path(frozen_selection_path).read_text());record=next(r for r in frozen if str(r['group'])==str(group) and r['model']==family);cfg=record['config'];source_score=record['source_mean_score']
            else:cfg=scores.index[0];source_score=float(scores.iloc[0])
            ix=target[(target.model==family)&(target.config==cfg)].index.to_list();selected.extend(ix)
            source_groups=sorted(source.group.unique().tolist());assert group not in source_groups
            choices.append(dict(group=group,model=family,config=cfg,source_mean_score=source_score,source_groups=source_groups,selection_from=str(frozen_selection_path) if frozen_selection_path else 'other source groups in this condition',n_source_trials=int(source.trial_uid.nunique())))
    ss=df.loc[selected].copy();ss.to_csv(out/'selected_metrics.csv',index=False)
    agg=ss.groupby(['dataset','trial_uid','case_id','mode','group','budget_s','rate_hz','scenario','seed','model'],as_index=False).agg(speed_rmse_mps=('speed_rmse_mps','mean'),gap_rmse_m=('gap_rmse_m','mean'),score=('score','mean'),failed_any_seed=('failed','max'),n_initializations=('neural_seed','count'),fit_seconds_amortized=('fit_seconds','mean'))
    agg.to_csv(out/'trial_metrics.csv',index=False)
    agg.groupby('model',sort=False).agg(n_trials=('trial_uid','size'),speed_rmse_mps=('speed_rmse_mps','mean'),gap_rmse_m=('gap_rmse_m','mean'),score=('score','mean'),failures_any_seed=('failed_any_seed','sum'),n_initializations=('n_initializations','min')).reset_index().to_csv(out/'summary.csv',index=False)
    (out/'selection.json').write_text(json.dumps(choices,indent=2));(out/'kinematic_projection_audit.json').write_text(json.dumps(consistency,indent=2))
    np.savez_compressed(out/'selected_predictions.npz',predictions=np.array(predictions)[selected],trial_uid=ss.trial_uid.to_numpy(),model=ss.model.to_numpy(),neural_seed=ss.neural_seed.to_numpy())
    config.update(total_seconds=time.perf_counter()-all_started,torch=torch.__version__,n_selected_models=len(ss),all_selection_groups_disjoint=True)
    (out/'config.json').write_text(json.dumps(config,indent=2));print('DONE',tag,config['total_seconds'],flush=True)
