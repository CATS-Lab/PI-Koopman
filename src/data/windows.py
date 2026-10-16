#!/usr/bin/env python3
"""Window-selection and release metadata helpers for CATS-CF."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.geometry import BASE_HZ, FLOAT_FORMAT, GAP_OFFSET_M, RATES_HZ, decimate, make_data_dictionary, make_koopman_table, make_quality_long, make_quality_manifest, role_mode, role_values


def add_case_columns(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    follower_mode = role_mode(pair, "follower")
    leader_mode = role_mode(pair, "leader")
    pair["_case_id"] = pd.Series(follower_mode).astype(str) + "_follows_" + pd.Series(leader_mode).astype(str)
    pair["_trial_uid"] = (
        pair["_case_id"]
        + "_run"
        + pair["run_index"].astype(int).astype(str).str.zfill(2)
    )
    return pair


def sliding_sum(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.r_[0, np.cumsum(values)]
    return cumulative[width:] - cumulative[:-width]


def status_arrays(group: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "ins_bad": (
            group["red_ins_status_status"].ne(3).astype(int)
            + group["black_ins_status_status"].ne(3).astype(int)
        ).to_numpy(),
        "ins0": (
            group["red_ins_status_status"].eq(0).astype(int)
            + group["black_ins_status_status"].eq(0).astype(int)
        ).to_numpy(),
        "pos_severe": (
            group["red_pos_type_type"].isin([0, 53]).astype(int)
            + group["black_pos_type_type"].isin([0, 53]).astype(int)
        ).to_numpy(),
        "pos0": (
            group["red_pos_type_type"].eq(0).astype(int)
            + group["black_pos_type_type"].eq(0).astype(int)
        ).to_numpy(),
        "pos53": (
            group["red_pos_type_type"].eq(53).astype(int)
            + group["black_pos_type_type"].eq(53).astype(int)
        ).to_numpy(),
        "pos55": (
            group["red_pos_type_type"].eq(55).astype(int)
            + group["black_pos_type_type"].eq(55).astype(int)
        ).to_numpy(),
        "pos56": (
            group["red_pos_type_type"].eq(56).astype(int)
            + group["black_pos_type_type"].eq(56).astype(int)
        ).to_numpy(),
        "hpos_sum": (
            np.hypot(group["red_latitude_stdev"], group["red_longitude_stdev"])
            + np.hypot(group["black_latitude_stdev"], group["black_longitude_stdev"])
        ).to_numpy(),
    }


def summarize_window(group: pd.DataFrame, start_index: int, width: int) -> dict:
    window = group.iloc[start_index : start_index + width]
    status = status_arrays(window)
    hpos = np.r_[
        np.hypot(window["red_latitude_stdev"], window["red_longitude_stdev"]),
        np.hypot(window["black_latitude_stdev"], window["black_longitude_stdev"]),
    ]
    return {
        "source_trial_id": group["trial_id"].iloc[0],
        "source_config": group["config"].iloc[0],
        "case_id": group["_case_id"].iloc[0],
        "trial_uid": group["_trial_uid"].iloc[0],
        "run_index": int(group["run_index"].iloc[0]),
        "window_start_offset_in_source_s": float(group["t_rel_s"].iloc[start_index]),
        "window_start_gps_s": float(window["t_abs_s"].iloc[0]),
        "ins_not3_samples": int(status["ins_bad"].sum()),
        "ins0_samples": int(status["ins0"].sum()),
        "pos0_or53_samples": int(status["pos_severe"].sum()),
        "pos0_samples": int(status["pos0"].sum()),
        "pos53_samples": int(status["pos53"].sum()),
        "pos55_samples": int(status["pos55"].sum()),
        "pos56_samples": int(status["pos56"].sum()),
        "vehicle_samples": int(width * 2),
        "horizontal_position_stdev_p95_m": float(np.quantile(hpos, 0.95)),
        "horizontal_position_stdev_max_m": float(np.max(hpos)),
    }


def best_window(group: pd.DataFrame, duration_s: int) -> tuple[pd.DataFrame, dict]:
    group = group.sort_values("t_rel_s").reset_index(drop=True)
    width = int(duration_s * BASE_HZ)
    if len(group) < width:
        raise ValueError(f"{group['trial_id'].iloc[0]} is shorter than {duration_s} s")
    status = status_arrays(group)
    ins = sliding_sum(status["ins_bad"], width)
    severe = sliding_sum(status["pos_severe"], width)
    pos55 = sliding_sum(status["pos55"], width)
    mean_hpos = sliding_sum(status["hpos_sum"], width) / width / 2.0
    starts = np.arange(len(ins))
    start_index = int(np.lexsort((starts, mean_hpos, pos55, ins + severe))[0])
    selected = group.iloc[start_index : start_index + width].copy()
    metrics = summarize_window(group, start_index, width)
    return selected, metrics


def rank_and_select(run_manifest: pd.DataFrame) -> pd.DataFrame:
    ranked = run_manifest.sort_values(
        [
            "case_id",
            "ins_not3_samples",
            "pos0_or53_samples",
            "pos55_samples",
            "horizontal_position_stdev_p95_m",
            "horizontal_position_stdev_max_m",
            "run_index",
        ]
    ).copy()
    ranked["quality_rank_within_case"] = ranked.groupby("case_id").cumcount() + 1
    ranked["selected_top2"] = ranked["quality_rank_within_case"].le(2)
    return ranked.sort_values(["case_id", "quality_rank_within_case"]).reset_index(drop=True)


def select_variant(pair: pd.DataFrame, duration_s: int, optimize_subwindow: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = add_case_columns(pair)
    windows: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    width = int(duration_s * BASE_HZ)
    for trial_id, group in pair.groupby("trial_id", sort=False):
        group = group.sort_values("t_rel_s").reset_index(drop=True)
        if optimize_subwindow:
            window, metrics = best_window(group, duration_s)
        else:
            if len(group) != width:
                raise ValueError(f"Expected {width} rows in {trial_id}; found {len(group)}")
            window = group.copy()
            metrics = summarize_window(group, 0, width)
        windows[trial_id] = window
        rows.append(metrics)

    manifest = rank_and_select(pd.DataFrame(rows))
    selected_ids = set(manifest.loc[manifest["selected_top2"], "source_trial_id"])
    selected = pd.concat([windows[key] for key in windows if key in selected_ids], ignore_index=True)
    selected = selected.sort_values(["_case_id", "run_index", "t_abs_s"]).reset_index(drop=True)
    selected["t_rel_s"] = selected.groupby("_trial_uid", sort=False).cumcount() / BASE_HZ
    return selected, manifest


def selected_summary(pair: pd.DataFrame, manifest: pd.DataFrame, duration_s: int) -> dict:
    hpos = np.r_[
        np.hypot(pair["red_latitude_stdev"], pair["red_longitude_stdev"]),
        np.hypot(pair["black_latitude_stdev"], pair["black_longitude_stdev"]),
    ]
    chosen = manifest.loc[manifest["selected_top2"]]
    vehicle_samples = int(len(pair) * 2)
    pos56 = int(chosen["pos56_samples"].sum())
    return {
        "duration_s": duration_s,
        "trials": int(pair["_trial_uid"].nunique()),
        "cases": int(pair["_case_id"].nunique()),
        "runs_per_case": int(pair.drop_duplicates("_trial_uid").groupby("_case_id").size().min()),
        "vehicle_samples": vehicle_samples,
        "ins_not3_samples": int(chosen["ins_not3_samples"].sum()),
        "ins0_samples": int(chosen["ins0_samples"].sum()),
        "pos0_or53_samples": int(chosen["pos0_or53_samples"].sum()),
        "pos0_samples": int(chosen["pos0_samples"].sum()),
        "pos53_samples": int(chosen["pos53_samples"].sum()),
        "pos55_samples": int(chosen["pos55_samples"].sum()),
        "pos56_samples": pos56,
        "pos56_ratio": float(pos56 / vehicle_samples),
        "horizontal_position_stdev_p95_m": float(np.quantile(hpos, 0.95)),
        "horizontal_position_stdev_max_m": float(np.max(hpos)),
        "selected_runs_with_pos0_or53": int(chosen["pos0_or53_samples"].gt(0).sum()),
        "selected_runs_with_pos55": int(chosen["pos55_samples"].gt(0).sum()),
    }


def make_public_measurements(pair: pd.DataFrame, koopman: pd.DataFrame) -> pd.DataFrame:
    out = koopman.loc[:, [
        "case_id", "trial_uid", "run_index", "sample_index", "sample_rate_hz",
        "t_rel_s", "t_gps_s", "follower_mode", "leader_mode",
    ]].copy()
    source_map = {
        "latitude_deg": "latitude",
        "longitude_deg": "longitude",
        "height_m": "height",
        "north_velocity_mps": "north_velocity",
        "east_velocity_mps": "east_velocity",
        "up_velocity_mps": "up_velocity",
        "roll_deg": "roll",
        "pitch_deg": "pitch",
        "azimuth_deg": "azimuth",
        "corrimu_sample_dt_s": "corrimu_sample_dt_s",
        "pitch_delta_corrimu_rad": "pitch_rate",
        "roll_delta_corrimu_rad": "roll_rate",
        "yaw_delta_corrimu_rad": "yaw_rate",
        "lateral_delta_v_corrimu_mps": "lateral_acc",
        "longitudinal_delta_v_corrimu_mps": "longitudinal_acc",
        "vertical_delta_v_corrimu_mps": "vertical_acc",
    }
    for role in ("follower", "leader"):
        for target, source in source_map.items():
            out[f"{role}_{target}"] = role_values(pair, role, source)
    return out


def quality_percentiles_present(quality_long: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "latitude_stdev_m", "longitude_stdev_m", "horizontal_position_stdev_m",
        "height_stdev_m", "north_velocity_stdev_mps", "east_velocity_stdev_mps",
        "horizontal_velocity_stdev_mps", "up_velocity_stdev_mps",
    ]
    rows = []
    for position_type in sorted(quality_long["position_type"].dropna().astype(int).unique()):
        subset = quality_long.loc[
            quality_long["ins_status"].eq(3) & quality_long["position_type"].eq(position_type)
        ]
        for metric in metrics:
            values = subset[metric].dropna()
            q = values.quantile([0.25, 0.50, 0.75, 0.95, 1.0])
            rows.append({
                "ins_status": 3,
                "position_type": position_type,
                "metric": metric,
                "n": int(len(values)),
                "p25": float(q.loc[0.25]),
                "p50": float(q.loc[0.50]),
                "p75": float(q.loc[0.75]),
                "p95": float(q.loc[0.95]),
                "max": float(q.loc[1.0]),
            })
    return pd.DataFrame(rows)


def trial_manifest(pair: pd.DataFrame, koopman: pd.DataFrame, duration_s: int) -> pd.DataFrame:
    data = pair.copy()
    data["case_id"] = koopman["case_id"].to_numpy()
    data["trial_uid"] = koopman["trial_uid"].to_numpy()
    rows = []
    for trial_uid, group in data.groupby("trial_uid", sort=True):
        first = group.iloc[0]
        rows.append({
            "case_id": first["case_id"],
            "trial_uid": trial_uid,
            "run_index": int(first["run_index"]),
            "follower_mode": first["case_id"].split("_follows_")[0],
            "leader_mode": first["case_id"].split("_follows_")[1],
            "window_start_gps_s": float(group["t_abs_s"].min()),
            "window_end_gps_s_exclusive": float(group["t_abs_s"].min() + duration_s),
            "window_duration_s": duration_s,
            "rows_125hz": int(len(group)),
            "enu_origin_latitude_deg": float((first["red_latitude"] + first["black_latitude"]) / 2.0),
            "enu_origin_longitude_deg": float((first["red_longitude"] + first["black_longitude"]) / 2.0),
            "enu_origin_height_m": float((first["red_height"] + first["black_height"]) / 2.0),
            "local_frame": "WGS84 geodetic -> ECEF -> per-trial ENU",
            "orientation_confidence": first["orientation_confidence"],
            "orientation_positive_ratio": float(first["selected_positive_ratio"]),
            "orientation_median_abs_lateral_offset_m": float(first["selected_median_abs_lateral_offset_m"]),
            "gap_offset_m": GAP_OFFSET_M,
        })
    return pd.DataFrame(rows)




def is_materialized(path: Path) -> bool:
    if not path.exists() or getattr(path.stat(), "st_blocks", 0) <= 0:
        return False
    try:
        with path.open("rb") as stream:
            return bool(stream.read(1))
    except OSError:
        return False


def refresh_source_lineage(workspace: Path, source_csv: Path) -> pd.DataFrame:
    lineage = pd.read_csv(source_csv)
    for source_name in ("inspvax", "corrimu"):
        path_col = f"source_{source_name}_csv"
        logical_col = f"{source_name}_logical_bytes"
        materialized_col = f"{source_name}_materialized_at_build"
        paths = [workspace / value for value in lineage[path_col]]
        lineage[logical_col] = [path.stat().st_size if path.exists() else np.nan for path in paths]
        lineage[materialized_col] = [is_materialized(path) for path in paths]
    return lineage


def source_materialization_summary(raw_root: Path) -> dict:
    summary = {}
    for filename in ("bynav_inspvax.csv", "bynav_corrimudata.csv", "bynav_imu_data_raw.csv"):
        paths = sorted(raw_root.glob(f"*/*/csv_out/{filename}"))
        summary[filename] = {
            "files": len(paths),
            "materialized_and_readable": sum(is_materialized(path) for path in paths),
            "dataless_or_empty": sum(not is_materialized(path) for path in paths),
        }
    return summary




