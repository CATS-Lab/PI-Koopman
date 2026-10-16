"""Importable, label-blind helpers for the proposed PI recognition follow-up.

No import-time fitting, output creation, classification, or experiment runner.
Observation-window selection and the frozen protocol belong to the caller.
"""
from pathlib import Path
import csv
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/release/15s/koopman_ready/koopman_25hz.csv'
OUT = ROOT / 'results/pi_recognition_2026-10-15'
PI_CONFIGS = {
    'PI-Koopman-CTH': dict(dictionary='quadratic', prior='cth', T=1., kappa=1., ridge=.1),
    'PI-Koopman-IDM': dict(dictionary='cth', prior='idm', T=1., kappa=1., ridge=.1),
}


def to_jsonable(value):
    """Recursively convert arrays/scalars; nonfinite numbers become JSON null."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_trials(path=DATA):
    """Load exactly the original 42 CSV groups, preserving first-seen row order.

    Each result separates metadata (including labels) from t/z/u. Fit helpers
    accept only numeric observation channels and cannot inspect this metadata.
    """
    groups = {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row['trial_uid'], []).append(row)
    if len(groups) != 42:
        raise ValueError(f'Expected 42 trials, found {len(groups)}')
    trials = []
    for uid, rows in groups.items():
        def arr(key):
            return np.array([float(row[key]) for row in rows])
        t = arr('t_rel_s')
        s = arr('gap_m')
        v = arr('follower_longitudinal_velocity_mps')
        u = arr('leader_velocity_in_follower_longitudinal_mps')
        z = np.column_stack((s, v, u-v))
        _validate(t, z, u)
        if not np.allclose(np.diff(t), .04, atol=1e-8):
            raise ValueError(f'{uid}: expected original uniform 25 Hz observations')
        first = rows[0]
        metadata = dict(trial_uid=uid, case_id=first['case_id'],
                        follower_mode=first['follower_mode'], leader_mode=first['leader_mode'],
                        run_index=int(first['run_index']),
                        is_acc=int(first['follower_mode'].startswith('ACC')))
        trials.append(dict(metadata=metadata, t=t, z=z, u=u))
    return trials


def _validate(t, z, u=None):
    t, z = np.asarray(t, float), np.asarray(z, float)
    if t.ndim != 1 or z.shape != (len(t), 3) or len(t) < 2:
        raise ValueError('Expected t[n] and z[n,3] with at least two samples')
    if not np.isfinite(t).all() or not np.isfinite(z).all() or not np.all(np.diff(t) > 0):
        raise ValueError('Observation channels must be finite with increasing times')
    if u is not None:
        u = np.asarray(u, float)
        if u.shape != t.shape or not np.isfinite(u).all() or not np.allclose(z[:, 2], u-z[:, 1], atol=1e-5):
            raise ValueError('Expected finite leader input with d = u-v')
    return t, z


def fit_pi_models(t, z, u):
    """Fit both current main PI variants, reusing native physical calibrations.

    Returns {name: {model, audit}} including physical baselines. No labels,
    prediction scores, tuning, classifier, or output writes enter this API.
    """
    t, z = _validate(t, z, u)
    try:
        from . import revision_physics as physics, revision_koopman as koop
    except ImportError:
        import revision_physics as physics
        import revision_koopman as koop
    results, priors = {}, {}
    for family in ('CTH', 'IDM'):
        coef, audit = physics.fit(t, z, u, family, max_nfev=400)
        model = dict(kind='physical', family=family.lower(), coef=coef, prior=None,
                     fit_seconds=audit['fit_seconds'], optimizer_audit=audit,
                     optimizer_success=audit['success'])
        priors[family.lower()] = model
        results[family+'-OE'] = dict(model=model, audit=audit)
    for name, config in PI_CONFIGS.items():
        prior = priors[config['prior']]
        model = koop.fit(t, z, u, 'expweak', dict(config, prior_model=prior))
        residual_seconds = model['fit_seconds']
        model['fit_seconds'] += prior['fit_seconds']
        audit = dict(config=config.copy(), physical_audit=prior['optimizer_audit'],
                     physical_fit_seconds=prior['fit_seconds'],
                     residual_fit_seconds=residual_seconds, total_fit_seconds=model['fit_seconds'],
                     rows=model['rows'], rank=model['rank'],
                     singular_values=model['singular_values'], condition=model['condition'])
        results[name] = dict(model=model, audit=audit)
    return results


