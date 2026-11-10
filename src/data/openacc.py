"""Load OpenACC trajectories and select the paper's 30-second observations."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.trials import Trial
from src.paths import OPENACC_RAW


TRAJECTORIES = {
    "AstaZero": [f"ASta_040719_platoon{i}.csv" for i in range(3, 11)]
    + [f"ASta_050719_platoon{i}.csv" for i in (1, 2)],
    "Casale": [f"part{i}.csv" for i in range(3, 12)],
}


def openacc(*, raw_dir=None):
    """Select eligible 30-second blocks from the two recorded campaigns."""
    root = Path(raw_dir) if raw_dir is not None else OPENACC_RAW
    trials, records = [], []
    for campaign, files in TRAJECTORIES.items():
        for name in sorted(files):
            path = root / campaign / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Run python -m scripts.download_openacc."
                )
            lines = path.read_text().splitlines()
            header = next(i for i, line in enumerate(lines) if line.startswith("Time,"))
            meta = {
                parts[0]: [v for v in parts[1:] if v]
                for parts in (line.split(",") for line in lines[:header])
            }
            data = pd.read_csv(path, skiprows=header)
            for follower in range(2, int(meta["Number_of_vehicles"][0]) + 1):
                candidates = []
                for start in range(0, len(data) - 299, 300):
                    block = data.iloc[start : start + 300]
                    t = block.Time.to_numpy()
                    t = t - t[0]
                    v = block[f"Speed{follower}"].to_numpy()
                    u = block[f"Speed{follower - 1}"].to_numpy()
                    gap = block[f"IVS{follower - 1}"].to_numpy()
                    state = np.column_stack((gap, v, u - v))
                    modes = block[f"Driver{follower}"].astype(str).str.strip().unique()
                    if (
                        len(modes) != 1
                        or modes[0] not in ("ACC", "Human")
                        or not np.isfinite(state).all()
                        or not np.all(gap > 0)
                        or not np.all(v >= 0)
                        or not np.all(u >= 0)
                        or not np.all(np.abs(np.diff(t) - 0.1) < 1e-4)
                    ):
                        continue
                    uid = f"{campaign}_{path.stem}_F{follower}_B{start // 300:04d}"
                    trial = Trial(
                        uid,
                        path.stem,
                        modes[0],
                        0,
                        np.arange(300) / 10,
                        state,
                        u,
                    )
                    record = dict(
                        uid=uid,
                        campaign=campaign,
                        source_file=name,
                        source_row_start=start,
                        source_time_start=float(block.Time.iloc[0]),
                        follower_index=follower,
                        vehicle=meta["Vehicle_order"][follower - 1],
                        mode=modes[0],
                        distance_setting=meta.get("Distance_setting", ["unknown"])[0],
                        gap_min=float(gap.min()),
                        leader_speed_range_mps=float(np.ptp(u)),
                        follower_speed_range_mps=float(np.ptp(v)),
                    )
                    candidates.append((trial, record))
                count = len(candidates)
                for index in np.rint(np.linspace(0, count - 1, min(6, count))).astype(
                    int
                ):
                    trial, record = candidates[index]
                    trials.append(trial)
                    records.append(dict(record, qualifying_blocks_for_file_pair=count))
    return trials, pd.DataFrame(records)


def crop(tr, a, b, rate=None):
    if a < tr.t[0] - 1e-6 or b > tr.t[-1] + 1e-6 or b < a:
        raise ValueError(
            f"{tr.uid}: requested [{a:g}, {b:g}] s outside available [{tr.t[0]:g}, {tr.t[-1]:g}] s; choose an interval fully inside the recording."
        )
    mask = (tr.t >= a - 1e-6) & (tr.t <= b + 1e-6)
    idx = np.flatnonzero(mask)
    if rate is not None:
        native = round(1 / np.median(np.diff(tr.t)))
        stride = int(round(native / rate))
        assert abs(native / stride - rate) < 1e-6
        idx = idx[::stride]
    t = tr.t[idx].copy()
    t -= t[0]
    return Trial(
        tr.uid,
        tr.case,
        tr.mode,
        tr.run,
        t,
        tr.z[idx].copy(),
        tr.u[idx].copy(),
    )


def observation_errors(uid, count, dt, scenario, seed):
    """Draw paired gap, follower-speed and leader-speed errors."""
    stable = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable]))
    errors = rng.normal(size=(count, 3))
    if scenario == "correlated":
        rho = np.exp(-dt / 0.3)
        for i in range(1, count):
            errors[i] = rho * errors[i - 1] + np.sqrt(1 - rho * rho) * errors[i]
    elif scenario != "white":
        raise ValueError(scenario)
    return errors * np.array([0.2, 0.1, 0.1])


def noisy_state(z, uid, scenario, seed, dt=0.1):
    """Perturb measured gap and absolute speeds on the observation clock."""
    if scenario == "clean":
        return z
    errors = observation_errors(uid, len(z), dt, scenario, seed)
    gap = z[:, 0] + errors[:, 0]
    if not np.all(gap > 0.1):
        raise ValueError(f"{uid}: observed gap at or below 0.1 m")
    follower = np.maximum(0.0, z[:, 1] + errors[:, 1])
    leader = np.maximum(0.0, z[:, 1] + z[:, 2] + errors[:, 2])
    return np.column_stack((gap, follower, leader - follower))


def corrupt(tr, scenario, seed):
    if scenario == "clean":
        return tr
    z = noisy_state(tr.z, tr.uid, scenario, seed, np.median(np.diff(tr.t)))
    return Trial(tr.uid, tr.case, tr.mode, tr.run, tr.t.copy(), z, z[:, 1] + z[:, 2])
