"""CATS-CF encounter records and release-table loading."""

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.geometry import BASE_HZ


from src.paths import RELEASE_DIR


@dataclass
class Trial:
    uid: str
    case: str
    mode: str
    run: int
    t: np.ndarray
    z: np.ndarray
    u: np.ndarray


def load_trials(duration=30, rate=125, *, release_dir=RELEASE_DIR):
    """Read a uniform prefix from each released 125 Hz recording."""
    if not (0 < duration <= 30 and rate > 0 and BASE_HZ % rate == 0):
        raise ValueError(
            "Expected a prefix of at most 30 seconds and a native divisor rate"
        )
    folder = Path(release_dir)
    trials = []
    for file in sorted(folder.glob("*_run??.csv")):
        df = pd.read_csv(file, float_precision="round_trip")
        if len(df) != 30 * BASE_HZ or not np.allclose(
            df.t_s, np.arange(30 * BASE_HZ) / BASE_HZ, atol=1e-12, rtol=0
        ):
            raise ValueError(f"Invalid recording clock in {file}")
        df = df.iloc[: int(duration * BASE_HZ) : BASE_HZ // rate]
        # Both speeds use the follower's longitudinal axis.
        heading = np.deg2rad(df.follower_heading_deg.to_numpy())
        u = df.leader_vx_mps.to_numpy() * np.sin(
            heading
        ) + df.leader_vy_mps.to_numpy() * np.cos(heading)
        v = df.follower_v_lon_mps.to_numpy()
        z = np.column_stack((df.gap_m, v, u - v))
        if not np.isfinite(z).all():
            raise ValueError(f"Nonfinite state in {file}")
        case, run = file.stem.rsplit("_run", 1)
        trials.append(
            Trial(
                file.stem,
                case,
                case.split("_follows_")[0],
                int(run),
                df.t_s.to_numpy(),
                z,
                u,
            )
        )
    if len(trials) != 42 or len({tr.case for tr in trials}) != 14:
        raise ValueError(f"Expected 42 trajectories across 14 conditions in {folder}")
    return trials, folder


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
