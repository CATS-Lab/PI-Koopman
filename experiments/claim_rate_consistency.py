"""Frozen sampling-rate response-field audit; no rate or result based selection.

Run with the project Python packages on PYTHONPATH. The local 125 Hz parent is
corrupted once before thinning. A numerical agreement reference is not truth.
This experiment concerns acquisition sampling, not physical frequency response.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from empirical_benchmark import Trial, load_trials
from revision_data import crop, corrupt
import revision_koopman as koop
import revision_physics as physics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/claim_reset/rate_consistency'
METHODS = ['fd', 'integral', 'weak', 'expweak']
RATES = [125, 25, 5, 1]
SCENARIOS = [('clean', 0)] + [(s, k) for s in ['white', 'correlated'] for k in [1101, 1102, 1103]]
CONFIG = dict(dictionary='quadratic', ridge=.1, T=2., kappa=1.)
LABELS = dict(fd='Finite difference', integral='Ordinary integral', weak='Weak (ours)', expweak='Exp. weak (ours)')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        title='Paired acquisition-rate response-field consistency, October 2026',
        protocol_version=1,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        local='All 42 released 15 s, 125 Hz trials; 14 cases, 3 runs each.',
        calibration_absolute_s=[2., 10.], future_absolute_s=[10., 15.],
        rates_hz=RATES, methods=METHODS, estimator_config=CONFIG,
        physical_support='2 s, starts at 0,1,...,6 s; 50% overlap at every rate. FD uses adjacent intervals.',
        scenarios=SCENARIOS, noise_sd=dict(gap_m=.5, follower_speed_mps=.1, leader_speed_mps=.1),
        corruption='revision_data.corrupt; applied on full 125 Hz parent before rate thinning; correlated time constant 0.3 s.',
        preprocessing='Independent features standardized using only each rate-specific calibration data; no smoothing; d=noisy u-noisy v.',
        probes='Same clean observed 5 Hz calibration states at absolute 2,2.2,...,10 s; no extrapolation grid.',
        primary_repeatability='RMS acceleration-field difference from same-method/same-noise-parent 125 Hz fit on common probes, m/s^2.',
        secondary_stability='RMS difference from same-method clean 125 Hz fit; this is stability, never acceleration ground truth.',
        response_noise='RMS change in regression response y relative to clean same-method/same-rate rows, m/s^2.',
        normalization='Common per-trajectory RMS clean 1 s velocity increments divided by 1 s, floor 0.1 m/s^2. Observed response scale, not acceleration truth.',
        accuracy='Common dense clean future input and clean initial state; next approximately 5 s gap/speed RMSE excludes initial state; failed rollouts retained. Conditional model accuracy, not noisy-input deployment.',
        failure_penalty=dict(gap_rmse_m=10000., speed_rmse_mps=1000.),
        synthetic='12 independent objects (seeds 39000..39011), each with 3 input/initial-state realizations (seeds 49000+100*object+run); known stable local quadratic law; exact same fit and acquisition protocol.',
        synthetic_law='a*(s-s0-tau*v)+b*d+c*(s-s0-tau*v)^2-h*d^2; a~U(.08,.18), b~U(.3,.8), tau~U(1,1.8), s0~U(2,5), c~U(.001,.003), h~U(.003,.01).',
        synthetic_leader='base~U(14,24) + 1.5*sin(.45*t+phase1) + .5*sin(1.2*t+phase2); RK4 dt=.008 s; v0=u0+U(-3,3), s0_initial=law_s0+tau*v0+U(-7,7).',
        aggregation='Average seeds and runs inside local case/synthetic object first. Report equal-weight case/object means; paired bootstrap 2000 resamples of 14 cases or 12 objects, seed 73001. No timepoint/seed independence claim.',
        tuning='None: ridge, dictionary, support, kernel and all scenarios fixed before fitting. Retain every grid cell and numerical failure.',
        scope='Sampling-rate sensitivity; neither physical excitation-frequency invariance nor string stability.'
    )
    path = OUT / 'protocol_frozen.json'
    if path.exists():
        prior = json.loads(path.read_text())
        # The source hash records the original execution, not a tunable rule.
        # Plot-only revisions retain that immutable hash and source snapshot.
        old_rules = {k: v for k, v in prior.items() if k!='script_sha256'}
        new_rules = {k: v for k, v in protocol.items() if k!='script_sha256'}
        if old_rules != new_rules:
            raise RuntimeError('Frozen protocol differs; do not silently overwrite or retune.')
    else:
        path.write_text(json.dumps(protocol, indent=2))


def law(z, p):
    s, v, d = np.asarray(z).T
    a, b, tau, s0, c, h = p
    e = s-s0-tau*v
    return a*e+b*d+c*e*e-h*d*d


def synthetic_trials():
    trials = []
    parameters = {}
    for obj in range(12):
        rng = np.random.default_rng(39000+obj)
        p = np.array([rng.uniform(.08, .18), rng.uniform(.3, .8), rng.uniform(1, 1.8),
                      rng.uniform(2, 5), rng.uniform(.001, .003), rng.uniform(.003, .01)])
        for run in range(3):
            rng = np.random.default_rng(49000+100*obj+run)
            base = rng.uniform(14, 24)
            phase = rng.uniform(0, 2*np.pi, 2)
            def leader(t):
                return base+1.5*np.sin(.45*t+phase[0])+.5*np.sin(1.2*t+phase[1])
            t = np.arange(1875)/125
            x = np.empty((len(t), 2))
            v0 = leader(0)+rng.uniform(-3, 3)
            x[0] = [p[3]+p[2]*v0+rng.uniform(-7, 7), v0]
            def rhs(tt, xx):
                d = leader(tt)-xx[1]
                return np.array([d, law(np.array([*xx, d]), p)])
            dt = .008
            for j, tt in enumerate(t[:-1]):
                a = rhs(tt, x[j]); b = rhs(tt+dt/2, x[j]+dt*a/2)
                c = rhs(tt+dt/2, x[j]+dt*b/2); d = rhs(tt+dt, x[j]+dt*c)
                x[j+1] = x[j]+dt*(a+2*b+2*c+d)/6
            u = leader(t)
            uid = f'quadratic_object{obj:02d}_run{run}'
            tr = Trial(uid, f'object{obj:02d}', 'synthetic', run, t,
                       np.c_[x, u-x[:, 1]], u, np.zeros(len(t)))
            assert np.isfinite(tr.z).all() and (tr.z[:, 0]>0).all()
            trials.append(tr)
            parameters[uid] = p.tolist()
    return trials, parameters


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def evaluate_rollout(m, truth):
    if m is None:
        return dict(rollout_failed=1, speed_rmse_mps=1000., gap_rmse_m=10000., negative_speed=0, negative_gap=0)
    pred = physics.predict_field(m, truth.t, truth.u, truth.z[0, :2])
    failed = not np.isfinite(pred).all()
    return dict(rollout_failed=int(failed),
                speed_rmse_mps=1000. if failed else rms(pred[1:, 1]-truth.z[1:, 1]),
                gap_rmse_m=10000. if failed else rms(pred[1:, 0]-truth.z[1:, 0]),
                negative_speed=int(np.any(pred[:, 1]<0)), negative_gap=int(np.any(pred[:, 0]<0)))


def run_dataset(dataset, trials, parameters=None):
    rows = []
    started = time.time()
    for j, tr in enumerate(trials):
        clean = crop(tr, 2, 10)
        probes = crop(tr, 2, 10, 5).z
        truth = crop(tr, 10, 15)
        scale = max(.1, rms(clean.z[125:, 1]-clean.z[:-125, 1]))
        clean_models = {}
        clean_y = {}
        for rate in RATES:
            train = crop(tr, 2, 10, rate)
            for method in METHODS:
                clean_models[rate, method] = koop.fit(train.t, train.z, train.u, method, CONFIG)
                F = koop.features(train.z, clean_models[rate, method])
                clean_y[rate, method] = koop.weak_rows(train.t, F, train.z[:, 1], method, 2., 1.)[1]
        for scenario, seed in SCENARIOS:
            obs = corrupt(tr, scenario, seed)
            models = {}
            for rate in RATES:
                train = crop(obs, 2, 10, rate)
                for method in METHODS:
                    record = dict(dataset=dataset, trial_uid=tr.uid, group=tr.case, run=tr.run,
                                  method=method, scenario=scenario, seed=seed, rate_hz=rate,
                                  response_scale_mps2=scale, fit_failed=0, fit_error='')
                    try:
                        m = clean_models[rate, method] if scenario=='clean' else koop.fit(train.t, train.z, train.u, method, CONFIG)
                        models[rate, method] = m
                        field = koop.acceleration(m, probes)
                        ref = koop.acceleration(models[125, method], probes)
                        clean_ref = koop.acceleration(clean_models[125, method], probes)
                        F = koop.features(train.z, m)
                        A, y = koop.weak_rows(train.t, F, train.z[:, 1], method, 2., 1.)
                        record.update(field_rate_rmse_mps2=rms(field-ref),
                                      field_clean125_rmse_mps2=rms(field-clean_ref),
                                      field_corruption_rmse_mps2=rms(field-koop.acceleration(clean_models[rate, method], probes)),
                                      response_noise_rmse_mps2=rms(y-clean_y[rate, method]),
                                      rows=m['rows'], rank=m['rank'], effective_rank=m['effective_rank_1e3'],
                                      fit_seconds=m['fit_seconds'])
                        if parameters is not None:
                            record['field_truth_rmse_mps2'] = rms(field-law(probes, parameters[tr.uid]))
                    except Exception as e:
                        m = None
                        record.update(fit_failed=1, fit_error=repr(e))
                        for key in ['field_rate_rmse_mps2', 'field_clean125_rmse_mps2', 'field_corruption_rmse_mps2', 'response_noise_rmse_mps2']:
                            record[key] = np.nan
                        if parameters is not None:
                            record['field_truth_rmse_mps2'] = np.nan
                    record.update(evaluate_rollout(m, truth))
                    for key in list(record):
                        if key.startswith('field_') and key.endswith('_mps2') or key=='response_noise_rmse_mps2':
                            record[key.replace('_mps2', '_normalized')] = record[key]/scale
                    rows.append(record)
        if (j+1)%6==0:
            print(dataset, j+1, '/', len(trials), 'seconds', round(time.time()-started, 1), flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'{dataset}_metrics.csv', index=False)


METRICS = ['field_rate_rmse_mps2', 'field_clean125_rmse_mps2', 'field_corruption_rmse_mps2',
           'response_noise_rmse_mps2', 'field_truth_rmse_mps2', 'field_rate_rmse_normalized',
           'field_clean125_rmse_normalized', 'field_corruption_rmse_normalized', 'field_truth_rmse_normalized',
           'speed_failure_inclusive_score', 'gap_failure_inclusive_score',
           'successful_rollout_speed_rmse_mps', 'successful_rollout_gap_rmse_m',
           'rollout_failed', 'fit_failed', 'negative_gap', 'negative_speed']


def summarize():
    d = pd.concat([pd.read_csv(OUT/f'{name}_metrics.csv') for name in ['local', 'synthetic']], ignore_index=True)
    # Preserve the original raw outcomes, while clearly distinguishing penalty
    # scores from physical RMSE in every derived summary and comparison.
    d['speed_failure_inclusive_score'] = d.speed_rmse_mps
    d['gap_failure_inclusive_score'] = d.gap_rmse_m
    d['successful_rollout_speed_rmse_mps'] = d.speed_rmse_mps.where(d.rollout_failed==0)
    d['successful_rollout_gap_rmse_m'] = d.gap_rmse_m.where(d.rollout_failed==0)
    metrics = [key for key in METRICS if key in d]
    groups = d.groupby(['dataset', 'scenario', 'rate_hz', 'method', 'group'])[metrics].mean().reset_index()
    groups.to_csv(OUT/'case_object_metrics.csv', index=False)
    rows = []
    for keys, dd in groups.groupby(['dataset', 'scenario', 'rate_hz', 'method']):
        row = dict(zip(['dataset', 'scenario', 'rate_hz', 'method'], keys))
        row['resampling_groups'] = len(dd)
        ix = np.random.default_rng(73001).integers(0, len(dd), (2000, len(dd)))
        for metric in metrics:
            arr = dd[metric].to_numpy()
            if np.isfinite(arr).all():
                row[metric] = float(arr.mean())
                row[metric+'_ci_lo'], row[metric+'_ci_hi'] = np.quantile(arr[ix].mean(axis=1), [.025, .975])
            else:
                row[metric] = np.nan
        raw = d[(d.dataset==keys[0]) & (d.scenario==keys[1]) & (d.rate_hz==keys[2]) & (d.method==keys[3])]
        row.update(n_fits=len(raw), rollout_failures=int(raw.rollout_failed.sum()), fit_failures=int(raw.fit_failed.sum()))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv', index=False)
    comparisons = []
    for keys, dd in groups.groupby(['dataset', 'scenario', 'rate_hz']):
        for metric in ['field_rate_rmse_mps2', 'field_clean125_rmse_mps2', 'response_noise_rmse_mps2', 'field_truth_rmse_mps2', 'speed_failure_inclusive_score']:
            for comparator in ['fd', 'integral']:
                wide = dd.pivot(index='group', columns='method', values=metric)
                for method in ['weak', 'expweak']:
                    diff = (wide[method]-wide[comparator]).to_numpy()
                    ix = np.random.default_rng(73001).integers(0, len(diff), (2000, len(diff)))
                    lo, hi = np.quantile(diff[ix].mean(axis=1), [.025, .975])
                    comparisons.append(dict(dataset=keys[0], scenario=keys[1], rate_hz=keys[2], metric=metric,
                                            method=method, comparator=comparator, difference=float(diff.mean()),
                                            ci_lo=lo, ci_hi=hi, resampling_groups=len(diff)))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_group_comparisons.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    started = time.time()
    if not args.report_only:
        freeze()  # persisted before reading/processing outcomes
        checks = koop.unit_checks()
        assert max(q['acceleration_error'] for q in checks)<1e-12
        (OUT/'implementation_checks.json').write_text(json.dumps(checks, indent=2))
        trials, _ = load_trials(15, 125)
        run_dataset('local', trials)
        trials, params = synthetic_trials()
        (OUT/'synthetic_parameters.json').write_text(json.dumps(params, indent=2))
        run_dataset('synthetic', trials, params)
    summary = summarize()
    if not args.report_only:
        (OUT/'runtime.json').write_text(json.dumps(dict(seconds=time.time()-started, completed=True,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print(summary[['dataset', 'scenario', 'rate_hz', 'method', 'field_rate_rmse_mps2', 'field_truth_rmse_mps2', 'rollout_failures']].to_string(index=False), flush=True)


if __name__=='__main__':
    main()
