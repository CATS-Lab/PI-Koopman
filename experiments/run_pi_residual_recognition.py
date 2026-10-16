"""Frozen in-sample residual recognition demonstration. Run explicitly only."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathlib import Path
import csv
import hashlib
import json
import traceback
import numpy as np

try:
    from . import pi_recognition_followup as helpers
    from . import pi_recognition_statistics as stats
    from .revision_koopman import acceleration
except ImportError:
    import pi_recognition_followup as helpers
    import pi_recognition_statistics as stats
    from revision_koopman import acceleration

OUT = helpers.OUT
N_BOOT = 5000
SEED = 20261015


def save_json(path, obj):
    path.write_text(json.dumps(helpers.to_jsonable(obj), indent=2, allow_nan=False)+'\n')


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordinary_residuals(t, z, model):
    field = acceleration(model, z)
    if not np.isfinite(field).all():
        raise ValueError('Nonfinite observed-state acceleration')
    rows = []
    for j in range(0, len(t)-25, 12):
        end = j+25
        duration = t[end]-t[j]
        if not np.isclose(duration, 1., atol=1e-8):
            raise ValueError('Expected exactly 1 s support')
        integral = np.trapezoid(field[j:end+1], t[j:end+1])
        residual = (z[end, 1]-z[j, 1]-integral)/duration
        if not np.isfinite(residual):
            raise ValueError('Nonfinite residual; no encounter may be dropped')
        rows.append(dict(window_index=len(rows), start_index=j, end_index=end,
                         start_s=float(t[j]), end_s=float(t[end]), residual=float(residual)))
    return rows


def choose_threshold(y, values):
    y, values = np.asarray(y), np.asarray(values, float)
    if not np.isfinite(values).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Threshold requires finite scores and both classes')
    unique = np.unique(values)
    candidates = np.r_[np.nextafter(unique[0], -np.inf),
                       unique[:-1]+np.diff(unique)/2,
                       np.nextafter(unique[-1], np.inf)]
    ba = np.array([.5*(np.mean((values[y == 0] > c))+
                           np.mean((values[y == 1] <= c))) for c in candidates])
    tied = candidates[np.isclose(ba, ba.max(), rtol=0, atol=1e-14)]
    return float(tied[(len(tied)-1)//2])


def synthetic_checks():
    t = np.arange(376)*.04
    z = np.column_stack((np.full(len(t), 20.), 10+1.7*t, np.zeros(len(t))))
    model = dict(kind='physical', family='cth', coef=[1., 0., 18.3, 0.], prior=None)
    residual = ordinary_residuals(t, z, model)
    assert len(residual) == 30
    assert np.max(np.abs([r['residual'] for r in residual])) < 1e-12
    threshold = choose_threshold([1, 1, 0, 0], [1., 2., 3., 4.])
    assert threshold == 2.5
    # Disconnected tied optima: their arithmetic median need not be optimal.
    y = np.array([1, 0, 0, 1, 1, 0])
    values = np.arange(1., 7.)
    threshold = choose_threshold(y, values)
    ba = lambda c: .5*(np.mean(values[y == 0] > c)+np.mean(values[y == 1] <= c))
    candidates = np.r_[0., np.arange(1.5, 6., 1.), 7.]
    assert np.isclose(ba(threshold), max(ba(c) for c in candidates))
    assert threshold == 1.5


def analyze(rows, score_key='residual_variance'):
    methods = list(dict.fromkeys(row['model'] for row in rows))
    reference = [r for r in rows if r['model'] == methods[0]]
    y = np.array([r['is_acc'] for r in reference])
    cases = np.array([r['case_id'] for r in reference])
    folds = np.array([r['run_index'] for r in reference])
    draws = stats.cluster_draws(y, cases, n=N_BOOT, seed=SEED)
    summary, oof, boot_values = {}, [], {}
    for method in methods:
        data = [r for r in rows if r['model'] == method]
        assert [r['trial_uid'] for r in data] == [r['trial_uid'] for r in reference]
        values = np.array([r[score_key] for r in data])
        pred = np.full(len(y), -1)
        for fold in np.unique(folds):
            train, test = folds != fold, folds == fold
            threshold = choose_threshold(y[train], values[train])
            pred[test] = values[test] <= threshold
            for i in np.flatnonzero(test):
                oof.append(dict(trial_uid=data[i]['trial_uid'], case_id=cases[i],
                                model=method, held_out_run_index=int(fold),
                                is_acc=int(y[i]), score_name=score_key, score_value=float(values[i]),
                                threshold=threshold, predicted_acc=int(pred[i])))
        result = stats.scores(y, pred, -values)
        boot_values[method], ci = stats.bootstrap_ba(y, pred, draws)
        acc, hdv = float(np.median(values[y == 1])), float(np.median(values[y == 0]))
        result.update(score_name=score_key, acc_median_score=acc, hdv_median_score=hdv,
                      hdv_acc_median_ratio=hdv/acc if acc > 0 else None,
                      balanced_accuracy_bootstrap_ci95=ci)
        summary[method] = result
    differences = {}
    for first in methods:
        for second in methods:
            if first.startswith('PI-') and not second.startswith('PI-'):
                delta = boot_values[first]-boot_values[second]
                differences[first+' minus '+second] = dict(
                    balanced_accuracy_difference=summary[first]['balanced_accuracy']-summary[second]['balanced_accuracy'],
                    paired_bootstrap_ci95=np.quantile(delta, [.025, .975]).tolist())
    return dict(score_name=score_key, models=summary, paired_comparisons=differences,
                bootstrap=dict(replicates=N_BOOT, seed=SEED,
                               unit='case_id conditions within class, retaining all runs',
                               scope='Conditional uncertainty for fixed out-of-fold predictions; thresholds not refitted in bootstrap')) , oof


def main():
    synthetic_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(helpers.__file__), Path(stats.__file__),
               Path(helpers.__file__).with_name('revision_koopman.py'),
               Path(helpers.__file__).with_name('revision_physics.py'), helpers.DATA]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    save_json(OUT/'protocol.json', dict(source_sha256=hashes, window_seconds=1.,
              stride_samples=12, sampling_hz=25, encounter_seconds=15,
              orientation='score <= training threshold => ACC',
              primary_score='residual_variance', secondary_score='residual_rmse',
              endpoint_policy='variance primary; RMSE prespecified secondary; no winner selection',
              threshold='maximize training balanced accuracy; lower central tied candidate threshold',
              features='label-blind per-encounter fitting, full observed encounter',
              config=helpers.PI_CONFIGS))
    models, rows, windows = {}, [], []
    current_uid = None
    try:
        trials = helpers.load_trials()
        for index, trial in enumerate(trials):
            metadata = trial['metadata']; current_uid = metadata['trial_uid']
            t, z, u = trial['t'], trial['z'], trial['u']
            print(f'[{index+1}/{len(trials)}] fitting {current_uid}', flush=True)
            fitted = helpers.fit_pi_models(t, z, u)
            models[current_uid] = dict(metadata=metadata, fitted=fitted)
            for name, item in fitted.items():
                residual_rows = ordinary_residuals(t, z, item['model'])
                if len(residual_rows) != 30:
                    raise ValueError(f'{current_uid}: expected 30 windows, found {len(residual_rows)}')
                residual = np.array([r['residual'] for r in residual_rows])
                row = dict(metadata, model=name, n_windows=len(residual),
                           residual_mean=float(residual.mean()),
                           residual_variance=float(residual.var(ddof=1)),
                           residual_rmse=float(np.sqrt(np.mean(residual**2))))
                if not np.isfinite([row[k] for k in ('residual_mean','residual_variance','residual_rmse')]).all():
                    raise ValueError('Nonfinite encounter score')
                rows.append(row)
                windows.extend(dict(metadata, model=name, **r) for r in residual_rows)
            save_json(OUT/'models.json', models)
            save_csv(OUT/'rows.csv', rows)
            save_csv(OUT/'residual_windows.csv', windows)
            print(f'[{index+1}/{len(trials)}] saved {current_uid}', flush=True)
        summary, oof = analyze(rows)
        common_rows = [r for r in rows if r['leader_mode'].startswith('ACC')]
        common_trials = {r['trial_uid']:r for r in common_rows}
        if len(common_trials) != 24 or sum(r['is_acc'] for r in common_trials.values()) != 12:
            raise ValueError('Expected common ACC-leader cohort of 24 encounters, 12 per class')
        common_summary, common_oof = analyze(common_rows)
        common_summary.update(n_encounters=24, cohort='leader_mode starts with ACC',
                              threshold_training='independent within this cohort; reused encounter fits')
        summary['common_acc_leader_validation'] = common_summary
        rmse_summary, rmse_oof = analyze(rows, 'residual_rmse')
        common_rmse_summary, common_rmse_oof = analyze(common_rows, 'residual_rmse')
        common_rmse_summary.update(n_encounters=24, cohort='leader_mode starts with ACC',
                                   threshold_training='independent within this cohort; reused encounter fits')
        rmse_summary.update(role='prespecified secondary fitting-residual endpoint; no winner selection',
                            common_acc_leader_validation=common_rmse_summary)
        summary['secondary_rmse'] = rmse_summary
        oof.extend(rmse_oof)
        common_oof.extend(common_rmse_oof)
        for row in oof:
            row['cohort'] = 'all42'
        for row in common_oof:
            row['cohort'] = 'common_acc_leader24'
        oof.extend(common_oof)
        summary.update(source_sha256=hashes, n_encounters=len(trials),
                       claim='In-sample observed-state residual variability recognition demonstration; not independent system-identification validation')
        save_csv(OUT/'oof.csv', oof)
        save_json(OUT/'summary.json', summary)
    except Exception as exc:
        save_json(OUT/'failure.json', dict(trial_uid=current_uid, error=str(exc), traceback=traceback.format_exc(),
                                         policy='Stopped; no failed encounter dropped', source_sha256=hashes))
        raise


if __name__ == '__main__':
    main()
