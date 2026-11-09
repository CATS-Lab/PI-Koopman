"""CATS-CF encounter records and release-table loading."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.geometry import BASE_HZ, GAP_OFFSET_M
from src.data.windows import paper_window_start, paper_slice_start, rank_and_select

from src.paths import RELEASE_DIR, ROOT

@dataclass
class Trial:
    uid: str
    case: str
    mode: str
    run: int
    t: np.ndarray
    z: np.ndarray
    u: np.ndarray
    corr: np.ndarray
    top2: bool = False

    def prefix(self, seconds):
        keep = self.t <= seconds + 1e-9
        return Trial(self.uid,self.case,self.mode,self.run,self.t[keep],self.z[keep],self.u[keep],self.corr[keep],self.top2)


def release_path(duration, rate=125, folder=RELEASE_DIR):
    folder = Path(folder)
    if any(folder.glob("*_run??.csv")):
        return folder
    if any(folder.glob("*/run_*/navigation.csv")):
        return folder
    measurements = folder / f"{duration}s_125hz.csv"
    if measurements.exists():
        return measurements
    return folder / f"{duration}s/koopman_ready/koopman_{rate}hz.csv"


def _following_run(path, duration):
    table = pd.read_csv(path, float_precision="round_trip")
    if len(table) != 30 * BASE_HZ or not np.allclose(table.t_s, np.arange(30 * BASE_HZ) / BASE_HZ, atol=1e-12, rtol=0):
        raise ValueError(f"Expected one uniform 30 s, 125 Hz table: {path}")
    uid = path.stem
    case, run = uid.rsplit("_run", 1)
    follower, leader = case.split("_follows_")
    start = paper_slice_start(table, uid, duration)
    table = table.iloc[start:start + duration * BASE_HZ].copy()
    if len(table) != duration * BASE_HZ:
        raise ValueError(f"The original {duration} s window is not contained in {path}")
    table["t_rel_s"] = np.arange(len(table)) / BASE_HZ
    table["trial_uid"], table["case_id"], table["run_index"] = uid, case, int(run)
    table["follower_mode"], table["leader_mode"] = follower, leader
    angle = np.deg2rad(table.follower_heading_deg)
    sin, cos = np.sin(angle), np.cos(angle)
    table["leader_velocity_in_follower_longitudinal_mps"] = table.leader_vx_mps * sin + table.leader_vy_mps * cos
    east, north = table.leader_x_m-table.follower_x_m, table.leader_y_m-table.follower_y_m
    table["lateral_offset_left_m"] = -east * cos + north * sin
    for role in ("leader", "follower"):
        table[f"{role}_horizontal_position_stdev_m"] = table[f"{role}_position_std_m"]
    return table.rename(columns={"follower_v_lon_mps": "follower_longitudinal_velocity_mps",
        "relative_v_mps": "relative_longitudinal_velocity_mps",
        "follower_yaw_rate_radps": "follower_yaw_rate_corrimu_radps"})


def _paper_run(run_dir, duration):
    nav = pd.read_csv(run_dir / "navigation.csv", float_precision="round_trip")
    corr = pd.read_csv(run_dir / "corrected_imu.csv", float_precision="round_trip")
    start = paper_window_start(nav, duration)
    relative = np.arange(int(duration * BASE_HZ)) / BASE_HZ
    grid = start + relative
    case = run_dir.parent.name
    run = int(run_dir.name.split("_")[-1])
    follower, leader = case.split("_follows_")
    data = {"t_rel_s": relative, "trial_uid": f"{case}_run{run:02d}", "case_id": case,
            "run_index": run, "follower_mode": follower, "leader_mode": leader}
    for role in ("follower", "leader"):
        car = nav.dropna(subset=[f"{role}_ins_status"])
        t = car.t_rel_s.to_numpy()
        for field in ("east_m", "north_m", "up_m", "azimuth_deg", "east_velocity_mps", "north_velocity_mps",
                      "north_position_stdev_m", "east_position_stdev_m"):
            name = f"{role}_{field}"
            data[name] = np.interp(grid, t, car[name])
        data[f"{role}_horizontal_position_stdev_m"] = np.hypot(
            data[f"{role}_north_position_stdev_m"], data[f"{role}_east_position_stdev_m"])
        # Discrete receiver status stays discrete across resampling.
        ix = np.searchsorted(t, grid).clip(0, len(t) - 1)
        prev = np.maximum(ix - 1, 0)
        ix = np.where(np.abs(grid - t[prev]) <= np.abs(grid - t[ix]), prev, ix)
        for field in ("ins_status", "position_type"):
            name = f"{role}_{field}"
            data[name] = car[name].to_numpy()[ix]
        if role == "follower":
            samples = corr[corr.vehicle == role]
            imu_t = samples.t_rel_s.to_numpy()
            dt = np.diff(imu_t)
            if not len(dt) or not np.all(dt > 0):
                raise ValueError(f"Invalid corrected IMU clock in {run_dir}")
            yaw_rate = samples.yaw_delta_rad.to_numpy() / np.r_[np.median(dt), dt]
            matched = pd.merge_asof(car[["t_rel_s"]], pd.DataFrame({"t_rel_s": imu_t, "yaw": yaw_rate}),
                                    on="t_rel_s", direction="nearest", tolerance=0.05).dropna()
            if len(matched) < 2:
                raise ValueError(f"Missing corrected IMU observations in {run_dir}")
            data["follower_yaw_rate_radps"] = np.interp(grid, matched.t_rel_s, matched.yaw)
    return pd.DataFrame(data)


def read_model_inputs(path, rate, duration=None):
    path = Path(path)
    single_tables = sorted(path.glob("*_run??.csv")) if path.is_dir() else []
    if path.is_dir():
        if duration is None:
            raise ValueError("Choose an experiment window duration when reading full recordings")
        if single_tables:
            data = pd.concat([_following_run(file, duration) for file in single_tables], ignore_index=True)
        else:
            runs = sorted(f.parent for f in path.glob("*/run_*/navigation.csv"))
            if not runs:
                raise FileNotFoundError(f"No CATS-CF recordings in {path}")
            data = pd.concat([_paper_run(run, duration) for run in runs], ignore_index=True)
    else:
        data = pd.read_csv(path)
    if single_tables:
        native = BASE_HZ
        # Current paper experiments use all 42 trials. Receiver-status-based
        # legacy subsets are not reconstructed from position uncertainty.
        data["top2"] = False
    elif "gap_m" in data:
        native = int(data["sample_rate_hz"].iloc[0])
        legacy = path.parent.parent / "metadata/cv_fold_manifest.csv"
        if legacy.is_file():
            folds = pd.read_csv(legacy).set_index("trial_uid")
            data["top2"] = data["trial_uid"].map(folds["include_in_top2_quality_subset"])
        else:
            data["top2"] = False
    else:
        native = BASE_HZ
        scores = []
        for uid, rows in data.groupby("trial_uid", sort=False):
            status = rows[["follower_ins_status", "leader_ins_status"]].to_numpy()
            position = rows[["follower_position_type", "leader_position_type"]].to_numpy()
            error = rows[["follower_horizontal_position_stdev_m", "leader_horizontal_position_stdev_m"]].to_numpy()
            scores.append(dict(trial_uid=uid, case_id=rows.case_id.iloc[0], run_index=rows.run_index.iloc[0],
                               ins_not3_samples=np.sum(status != 3), pos0_or53_samples=np.isin(position, [0, 53]).sum(),
                               pos55_samples=np.sum(position == 55), horizontal_position_stdev_p95_m=np.quantile(error, .95),
                               horizontal_position_stdev_max_m=error.max()))
        ranks = rank_and_select(pd.DataFrame(scores)).set_index("trial_uid")
        data["top2"] = data["trial_uid"].map(ranks["selected_top2"])
    if rate <= 0 or native % rate:
        raise ValueError(f"Cannot decimate {native} Hz to {rate} Hz")
    offsets = data.groupby("trial_uid", sort=False).cumcount()
    data = data.loc[offsets.mod(native // rate).eq(0)].copy()
    if "gap_m" not in data:
        angle = np.deg2rad(data["follower_azimuth_deg"])
        sin, cos = np.sin(angle), np.cos(angle)
        east = data["leader_east_m"] - data["follower_east_m"]
        north = data["leader_north_m"] - data["follower_north_m"]
        data["gap_m"] = east * sin + north * cos - GAP_OFFSET_M
        data["lateral_offset_left_m"] = -east * cos + north * sin
        for role, name in (("follower", "follower_longitudinal_velocity_mps"),
                           ("leader", "leader_velocity_in_follower_longitudinal_mps")):
            data[name] = data[f"{role}_east_velocity_mps"] * sin + data[f"{role}_north_velocity_mps"] * cos
        data["relative_longitudinal_velocity_mps"] = (
            (data.leader_east_velocity_mps - data.follower_east_velocity_mps) * sin
            + (data.leader_north_velocity_mps - data.follower_north_velocity_mps) * cos
        )
        data["follower_yaw_rate_corrimu_radps"] = data["follower_yaw_rate_radps"]
    return data


def load_trials(duration=15, rate=25, *, release_dir=RELEASE_DIR):
    path = release_path(duration, rate, release_dir)
    data = read_model_inputs(path, rate, duration)
    trials = []
    for uid, rows in data.groupby("trial_uid", sort=False):
        t = rows.t_rel_s.to_numpy()
        v = rows.follower_longitudinal_velocity_mps.to_numpy()
        u = rows.leader_velocity_in_follower_longitudinal_mps.to_numpy()
        z = np.column_stack((rows.gap_m, v, u - v))
        corr = (rows.follower_yaw_rate_corrimu_radps * rows.lateral_offset_left_m).to_numpy()
        first = rows.iloc[0]
        trials.append(Trial(uid, first.case_id, first.follower_mode, int(first.run_index), t, z, u, corr, bool(first.top2)))
    if len(trials) != 42 or len({t.case for t in trials}) != 14:
        raise ValueError("Expected 42 CATS-CF trials across 14 conditions")
    if any(len(t.t) != duration * rate or not np.isfinite(t.z).all() or
           not np.allclose(np.diff(t.t), 1 / rate) for t in trials):
        raise ValueError("Invalid CATS-CF trial length, clock, or state")
    return trials, path


def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


