"""Independent latency measurements for one previously specified encounter fit.

Run inside the portable bundle made by prepare_timing_single_encounter.py.
No model batching, hyperparameter search, or test-outcome selection is performed.
"""
import os
for _var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[_var] = '1'
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
from types import SimpleNamespace
import numpy as np
import scipy
import pandas as pd
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import revision_koopman as koop
import revision_physics as physics


def fit_one(case, arrays, name):
    t,z,u = (arrays[case['id']+'_'+key] for key in ['t','z','u'])
    cfg = case['models'][name]['config'].copy()
    started=time.perf_counter_ns();prior_ns=0
    if name.startswith('PI-'):
        family='CTH' if name=='PI-CTH' else 'IDM'
        p,audit=physics.fit(t,z,u,family)
        prior_ns=time.perf_counter_ns()-started
        prior=dict(kind='physical',family=family.lower(),coef=p,audit=audit)
        cfg['prior_model']=prior
        model=koop.fit(t,z,u,case['models'][name]['method'],cfg)
    else:
        model=koop.fit(t,z,u,'expweak',cfg)
    return model, (time.perf_counter_ns()-started)/1e6, prior_ns/1e6


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cases=json.loads((root/'data/timing/cases.json').read_text());arrays=np.load(root/'data/timing/state_arrays.npz')
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        limit=threadpool_limits(limits=1);pools=threadpool_info()
    except ImportError:
        limit=None;pools=[]
    host=dict(label=args.label,python=sys.version,platform=platform.platform(),
              numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
              requested_numerical_threads=1,threadpool_info=pools,
              cpu_count=os.cpu_count(),input_sha256=hashlib.sha256((root/'data/timing/state_arrays.npz').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              native_sha256=hashlib.sha256((root/'experiments/revision_native.c').read_bytes()).hexdigest(),
              cases_sha256=hashlib.sha256((root/'data/timing/cases.json').read_bytes()).hexdigest())
    (out/'runtime.json').write_text(json.dumps(host,indent=2)+'\n')
    rows=[];steps=[];replay=[];start=time.perf_counter()
    for case in cases:
        ts=arrays[case['id']+'_future_t'];us=arrays[case['id']+'_future_u'];x0=arrays[case['id']+'_x0']
        for name in ['PI-CTH','PI-IDM']:
            # One full untimed warmup per encounter/configuration, including fit.
            model,_,_=fit_one(case,arrays,name)
            pred=physics.predict_field(model,ts,us,x0)
            expected=arrays[case['id']+'_'+name+'_expected']
            finite=np.isfinite(pred)&np.isfinite(expected)
            error=float(np.max(np.abs(pred[finite]-expected[finite]))) if finite.any() else 0.
            replay.append(dict(case=case['id'],model=name,max_abs_prediction_difference=error,
                equal_finite_mask=bool(np.array_equal(np.isfinite(pred),np.isfinite(expected)))))
            # Individual two-point deployment calls; each timer wraps one call.
            step_t=np.array([0.,1./case['rate_hz']]);step_u=np.array([us[0],np.interp(step_t[1],ts,us)])
            for repeat in range(200):
                t0=time.perf_counter_ns();physics.predict_field(model,step_t,step_u,x0)
                steps.append(dict(case=case['id'],dataset=case['dataset'],budget_s=case['budget_s'],
                                  model=name,repeat=repeat,single_step_us=(time.perf_counter_ns()-t0)/1e3))
            for repeat in range(7):
                total=time.perf_counter_ns();model,fit_ms,prior_ms=fit_one(case,arrays,name)
                t0=time.perf_counter_ns();pred=physics.predict_field(model,ts,us,x0)
                roll_ms=(time.perf_counter_ns()-t0)/1e6;total_ms=(time.perf_counter_ns()-total)/1e6
                rows.append(dict(case=case['id'],dataset=case['dataset'],budget_s=case['budget_s'],
                    rate_hz=case['rate_hz'],model=name,repeat=repeat,fit_ms=fit_ms,
                    physical_prior_ms=prior_ms,correction_or_generator_fit_ms=fit_ms-prior_ms,
                    rollout_5s_ms=roll_ms,total_fit_rollout_ms=total_ms,
                    rollout_completed=bool(np.isfinite(pred).all())))
        print(case['id'],case['dataset'],case['budget_s'],round(time.perf_counter()-start,2),flush=True)
    df=pd.DataFrame(rows);st=pd.DataFrame(steps)
    df.to_csv(out/'latency_samples.csv',index=False);st.to_csv(out/'single_step_samples.csv',index=False)
    pd.DataFrame(replay).to_csv(out/'archive_replay.csv',index=False)
    summary=[]
    for (dataset,budget,model),g in df.groupby(['dataset','budget_s','model']):
        row=dict(hardware=args.label,dataset=dataset,budget_s=budget,model=model,
                 target_count=g['case'].nunique(),timed_fit_count=len(g),rollout_failed_calls=int((~g.rollout_completed).sum()))
        sg=st[(st.dataset==dataset)&(st.budget_s==budget)&(st.model==model)]
        for col in ['fit_ms','physical_prior_ms','correction_or_generator_fit_ms','rollout_5s_ms','total_fit_rollout_ms']:
            row[col+'_median']=float(g[col].median());row[col+'_p95']=float(g[col].quantile(.95));row[col+'_max']=float(g[col].max())
        for q in ['median','p95']:
            row['single_step_us_'+q]=float(sg.single_step_us.median() if q=='median' else sg.single_step_us.quantile(.95))
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'latency_summary.csv',index=False)
    (out/'completion.json').write_text(json.dumps(dict(completed=True,wall_seconds=time.perf_counter()-start,
        target_budget_cases=len(cases),fit_calls=len(rows),single_step_calls=len(steps),
        warmup_fits_per_case_model=1,timed_repetitions=7,source_selection_time_excluded=True,
        fit_includes='two-start physical output-error calibration for each PI model, normalization, weak rows, regression and diagnostics',
        rollout='same native RK4, max step 0.04s, complete5s future at local125Hz/OpenACC10Hz, no follower feedback',
        scope='isolated single-target CPU calls after warmup, no batched neural amortization, no device-level hard-real-time guarantee'),indent=2)+'\n')


if __name__=='__main__':main()
