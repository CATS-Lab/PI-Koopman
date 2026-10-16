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
COEFFICIENT_NAMES = ('c0', 'cs', 'cv', 'cd')
KINEMATIC_NAMES = tuple(f'{x}_{q}' for x in ('gap', 'v', 'd') for q in ('mean', 'sd'))
LEADER_NAMES = ('u_mean', 'u_sd', 'u_range', 'leader_acc_sd')
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


def fit_affine4(t, z, u=None):
    """Exact exploratory_findings.affine arithmetic, without importing its runner.

    Fixed 26-sample ordinary integral windows, stride12, ridge0.1 and local
    scale floors [1,.5,.5]. Returns a physical-coordinate full field and audit.
    """
    t, z = _validate(t, z, u)
    if len(t) < 26 or not np.allclose(np.diff(t), .04, atol=1e-8):
        raise ValueError('Frozen affine4 requires at least 26 samples at 25 Hz')
    start = time.perf_counter()
    mu = z.mean(0); sc = np.maximum(z.std(0), [1., .5, .5])
    X = np.c_[np.ones(len(z)), (z-mu)/sc]
    A, y = [], []
    for j in range(0, len(t)-26+1, 12):
        sl = slice(j, j+26); T = t[j+25]-t[j]
        A.append(np.trapezoid(X[sl], t[sl], axis=0)/T)
        y.append((z[j+25, 1]-z[j, 1])/T)
    A = np.array(A); y = np.array(y)
    sv = np.linalg.svd(A/np.sqrt(len(A)), compute_uv=False)
    c = np.linalg.solve(A.T@A/len(A)+.1*np.eye(4), A.T@y/len(A))
    beta = np.r_[c[0]-np.sum(c[1:]*mu/sc), c[1:]/sc]
    rss = np.sum((A@c-y)**2); tss = np.sum((y-y.mean())**2)
    audit = dict(zip(COEFFICIENT_NAMES, beta))
    audit.update(condition=sv[0]/sv[-1], effective_rank=int(np.sum(sv>sv[0]*1e-3)),
                 integral_r2=1-rss/tss if tss>1e-12 else np.nan,
                 rows=len(y), singular_values=sv, local_mean=mu, local_scale=sc,
                 standardized_coef=c, fit_seconds=time.perf_counter()-start)
    model = dict(kind='field', dictionary='affine', prior=None, coef=beta,
                 feature_mean=np.zeros(4), feature_scale=np.ones(4),
                 zmean=mu, zscale=sc, method='ordinary_integral_frozen_exploratory',
                 config=dict(support_samples=26, stride_samples=12, ridge=.1),
                 training_min=z.min(0), training_max=z.max(0))
    return model, audit


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


def fixed_features(t, z, u):
    """Original kinematic6 and leader4 in their original ordered coordinates."""
    t, z = _validate(t, z, u); u = np.asarray(u, float)
    kinematic = np.array([stat for j in range(3) for stat in (z[:, j].mean(), z[:, j].std(ddof=1))])
    leader = np.array([u.mean(), u.std(ddof=1), np.ptp(u), np.gradient(u, t).std(ddof=1)])
    return dict(kinematic6=kinematic, leader4=leader,
                kinematic_names=KINEMATIC_NAMES, leader_names=LEADER_NAMES)


def project_response4(model, probes, weights=None):
    """Project the full response onto common physical [1,s,v,d] coordinates.

    The caller supplies the same external probes/weights for every method.
    No probe generation or target/label-dependent selection occurs here.
    Centering/scaling stabilizes least squares and is inverted exactly.
    """
    try:
        from .revision_koopman import acceleration
    except ImportError:
        from revision_koopman import acceleration
    probes = np.asarray(probes, float)
    if probes.ndim != 2 or probes.shape[1] != 3 or len(probes) < 4 or not np.isfinite(probes).all():
        raise ValueError('Expected at least four finite physical-coordinate probes')
    w = np.ones(len(probes)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(probes),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, and have positive sum')
    w = w/w.sum()
    mu = np.sum(w[:, None]*probes, axis=0)
    sc = np.sqrt(np.sum(w[:, None]*(probes-mu)**2, axis=0))
    sc = np.where(sc > 0, sc, 1.)
    X = np.c_[np.ones(len(probes)), (probes-mu)/sc]
    response = acceleration(model, probes)
    if not np.isfinite(response).all():
        raise ValueError('Model response is nonfinite on supplied probes')
    c, _, rank, singular = np.linalg.lstsq(X*np.sqrt(w[:, None]), response*np.sqrt(w), rcond=None)
    if rank != 4:
        raise ValueError('Supplied common probes do not identify four affine coordinates')
    beta = np.r_[c[0]-np.sum(c[1:]*mu/sc), c[1:]/sc]
    audit = dict(rank=int(rank), singular_values=singular, n_probes=len(probes),
                 projection_rmse=float(np.sqrt(np.sum(w*(X@c-response)**2))),
                 probe_mean=mu, probe_scale=sc)
    return dict(zip(COEFFICIENT_NAMES, beta)), audit
