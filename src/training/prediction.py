"""Grouped, frozen protocol sensitivity and paired PI noise extension.

Every candidate is retained. Prediction origins are averaged, never chosen for
their target performance. All configuration decisions exclude the target run or
source file. Historical, already reported outcomes are not called fresh data.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from src.data.trials import ROOT, Trial, load_trials
from src.data.openacc import openacc, crop, corrupt
import src.models.physics as physics
import src.models.koopman as koop
from src.evaluation.metrics import metric, grid

OUT = ROOT / 'outputs/physics'
ORIGINS = [15, 20, 24]
BUDGETS = [3, 8, 15]
RATES = [1, 5, 10, 25, 125]
METHODS = ['CTH', 'IDM', 'PI-Koopman-CTH', 'PI-Koopman-IDM']
PI = {'PI-Koopman-CTH': ('quadratic', 'cth'), 'PI-Koopman-IDM': ('cth', 'idm')}
METRICS = ['speed_rmse_mps', 'gap_rmse_m']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample(tr, start, end, rate):
    native=round(1/np.median(np.diff(tr.t)))
    if native % rate == 0:
        result=crop(tr,start,end,rate)
    else:
        # Restrict the interpolation support before evaluating any target point.
        keep=(tr.t>=start-1e-6)&(tr.t<=end+1e-6)
        tt=np.arange(int(round((end-start)*rate))+1)/rate+start
        source=tr.t[keep]
        assert tt[0]>=source[0]-1e-9 and tt[-1]<=source[-1]+1e-9
        s=np.interp(tt,source,tr.z[keep,0]);v=np.interp(tt,source,tr.z[keep,1]);u=np.interp(tt,source,tr.u[keep])
        result=Trial(tr.uid,tr.case,tr.mode,tr.run,tt-start,np.c_[s,v,u-v],u,np.interp(tt,source,tr.corr[keep]),tr.top2)
    assert len(result.t)==int(round((end-start)*rate))+1
    assert np.isclose(result.t[-1],end-start)
    return result


def run_cell(dataset, budget, rate, origin, trials):
    out=OUT/'grid'/f'{dataset}_B{budget}_Hz{rate}_end{origin}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    rows=[];started=time.perf_counter()
    for tr in trials:
        train=resample(tr,origin-budget,origin,rate);truth=crop(tr,origin,origin+5)
        base=dict(dataset=dataset,budget_s=budget,rate_hz=rate,origin_s=origin,trial_uid=tr.uid,case_id=tr.case,
            mode=tr.mode,group=str(tr.run) if dataset=='local' else tr.case)
        def append(name,cfg,model):
            pred=physics.predict_field(model,truth.t,truth.u,truth.z[0,:2])
            rows.append(dict(**base,model=name,config=json.dumps(cfg,sort_keys=True),fit_seconds=model.get('fit_seconds',0),**metric(pred,truth)))
        priors={}
        for family in ['CTH','IDM']:
            p,audit=physics.fit(train.t,train.z,train.u,family)
            priors[family.lower()]=dict(kind='physical',family=family.lower(),coef=p,audit=audit,fit_seconds=audit['fit_seconds'])
            append(family,{},priors[family.lower()])
        for name in PI:
            dictionary,prior=PI.get(name,('hybrid',None))
            for cfg in grid('expweak',rate,budget):
                full=dict(**cfg,dictionary=dictionary,prior=prior)
                if prior:full['prior_model']=priors[prior]
                model=koop.fit(train.t,train.z,train.u,'expweak',full)
                model['fit_seconds']+=priors[prior]['fit_seconds'] if prior else 0
                append(name,cfg,model)
    df=pd.DataFrame(rows);df.to_csv(out/'candidate_metrics.csv',index=False)
    selected,choices=select(df)
    selected.to_csv(out/'selected_metrics.csv',index=False)
    (out/'selection.json').write_text(json.dumps(choices,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-started,rows=len(df),selected=len(selected),n_records=len(trials),protocol_sha256=sha(OUT/'protocol_frozen.json')),indent=2))
    print(out.name,'seconds',round(time.perf_counter()-started,1),flush=True)


def select(df):
    selected=[];choices=[]
    for group in sorted(df.group.unique()):
        source=df[df.group.ne(group)];target=df[df.group.eq(group)]
        for model in METHODS:
            d=source[source.model.eq(model)]
            rank=d.groupby('config').score.mean().sort_values(kind='stable')
            cfg=rank.index[0]
            selected.append(target[target.model.eq(model)&target.config.eq(cfg)])
            choices.append(dict(group=group,model=model,config=cfg,source_score=float(rank.iloc[0]),source_groups=sorted(source.group.unique()),target_excluded=group not in source.group.unique()))
    return pd.concat(selected,ignore_index=True),choices


def summarize(df,keys):
    rows=[]
    for values,d in df.groupby(keys,sort=False):
        if not isinstance(values,tuple):values=(values,)
        good=d[d.failed.eq(0)]
        rows.append(dict(zip(keys,values),attempts=len(d),unique_records=d.trial_uid.nunique(),failures=int(d.failed.sum()),
            **{m:float(d[m].mean()) for m in METRICS},**{'completed_'+m:float(good[m].mean()) for m in METRICS},
            score=float(d.score.mean()),mean_fit_seconds=float(d.fit_seconds.mean()) if 'fit_seconds' in d else np.nan))
    return pd.DataFrame(rows)


