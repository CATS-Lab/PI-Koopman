"""PI model fitting and data helpers for residual-based recognition."""

from src.paths import OUTPUTS

from pathlib import Path
import numpy as np

from src.data.trials import load_trials as load_recordings
from src.paths import RELEASE_DIR

DATA = RELEASE_DIR
OUT = OUTPUTS / "recognition"
RATES_HZ = (10, 25)
OBSERVATION_S = 15.0
RESIDUAL_SUPPORT_S = 1.0
RESIDUAL_STRIDE_S = 0.4
PI_CONFIGS = {
    "PI-Koopman-CTH": dict(
        dictionary="quadratic", prior="cth", T=0.5, kappa=1.0, ridge=10.0
    ),
    "PI-Koopman-IDM": dict(dictionary="cth", prior="idm", T=2.0, kappa=1.0, ridge=10.0),
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


def load_trials(path=DATA, rate=25):
    """Observe the first 15 seconds of each 30-second recording."""
    if rate not in RATES_HZ:
        raise ValueError(f"The recognition example uses rates {RATES_HZ}")
    recordings, _ = load_recordings(30, 125, release_dir=Path(path))
    trials = []
    for record in recordings:
        t = np.arange(int(OBSERVATION_S * rate)) / rate
        keep = record.t < OBSERVATION_S - 1e-9
        if 125 % rate == 0:
            index = np.flatnonzero(keep)[:: 125 // rate]
            z, u = record.z[index].copy(), record.u[index].copy()
        else:
            # As in the prediction experiment, 10 Hz uses exact 0.1 s times
            # and interpolation supported only by the observation interval.
            source = record.t[keep]
            assert t[0] >= source[0] and t[-1] <= source[-1]
            s = np.interp(t, source, record.z[keep, 0])
            v = np.interp(t, source, record.z[keep, 1])
            u = np.interp(t, source, record.u[keep])
            z = np.column_stack((s, v, u - v))
        _validate(t, z, u)
        if len(t) != 15 * rate or not np.allclose(np.diff(t), 1 / rate, atol=1e-8):
            raise ValueError(f"{record.uid}: invalid recognition observation grid")
        follower, leader = record.case.split("_follows_")
        metadata = dict(
            trial_uid=record.uid,
            case_id=record.case,
            follower_mode=follower,
            leader_mode=leader,
            run_index=record.run,
            is_acc=int(follower.startswith("ACC")),
        )
        trials.append(dict(metadata=metadata, t=t, z=z, u=u))
    return trials


def _validate(t, z, u=None):
    t, z = np.asarray(t, float), np.asarray(z, float)
    if t.ndim != 1 or z.shape != (len(t), 3) or len(t) < 2:
        raise ValueError("Expected t[n] and z[n,3] with at least two samples")
    if (
        not np.isfinite(t).all()
        or not np.isfinite(z).all()
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError("Observation channels must be finite with increasing times")
    if u is not None:
        u = np.asarray(u, float)
        if (
            u.shape != t.shape
            or not np.isfinite(u).all()
            or not np.allclose(z[:, 2], u - z[:, 1], atol=1e-5)
        ):
            raise ValueError("Expected finite leader input with d = u-v")
    return t, z


def fit_pi_models(t, z, u):
    """Fit both physical priors and their fixed weak-form corrections."""
    t, z = _validate(t, z, u)
    import src.models.physics as physics
    import src.models.koopman as koop

    results, priors = {}, {}
    for family in ("CTH", "IDM"):
        coef, audit = physics.fit(t, z, u, family, max_nfev=400)
        model = dict(
            kind="physical",
            family=family.lower(),
            coef=coef,
            prior=None,
            fit_seconds=audit["fit_seconds"],
            optimizer_audit=audit,
            optimizer_success=audit["success"],
        )
        priors[family.lower()] = model
        results[family + "-OE"] = dict(model=model, audit=audit)
    rate = round(1 / np.median(np.diff(t)))
    for name, nominal in PI_CONFIGS.items():
        # Match the prediction grid: a half interval rounds up (12.5 -> 13 at 25 Hz).
        intervals = int(np.floor(nominal["T"] * rate + 0.5))
        config = dict(nominal, T=intervals / rate)
        prior = priors[config["prior"]]
        model = koop.fit(t, z, u, "expweak", dict(config, prior_model=prior))
        residual_seconds = model["fit_seconds"]
        model["fit_seconds"] += prior["fit_seconds"]
        audit = dict(
            config=config.copy(),
            physical_audit=prior["optimizer_audit"],
            physical_fit_seconds=prior["fit_seconds"],
            residual_fit_seconds=residual_seconds,
            total_fit_seconds=model["fit_seconds"],
            rows=model["rows"],
            rank=model["rank"],
            singular_values=model["singular_values"],
            condition=model["condition"],
        )
        results[name] = dict(model=model, audit=audit)
    return results
