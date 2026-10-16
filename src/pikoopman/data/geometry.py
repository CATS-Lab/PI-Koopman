#!/usr/bin/env python3
"""Core coordinate, quality, decimation, and metadata helpers for CATS-CF."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE_HZ = 125
WINDOW_S = 15.0
GAP_OFFSET_M = 3.0
RATES_HZ = (125, 25, 5, 1)
FLOAT_FORMAT = "%.12g"
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def role_values(pair: pd.DataFrame, role: str, suffix: str) -> np.ndarray:
    follower_is_red = pair["inferred_follower_color"].eq("Red").to_numpy()
    use_red = follower_is_red if role == "follower" else ~follower_is_red
    return np.where(
        use_red,
        pair[f"red_{suffix}"].to_numpy(),
        pair[f"black_{suffix}"].to_numpy(),
    )


def role_mode(pair: pd.DataFrame, role: str) -> np.ndarray:
    follower_is_red = pair["inferred_follower_color"].eq("Red").to_numpy()
    use_red = follower_is_red if role == "follower" else ~follower_is_red
    return np.where(
        use_red,
        pair["red_mode"].to_numpy(),
        pair["black_mode"].to_numpy(),
    )


def wrapped_degrees(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, height_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    height = np.asarray(height_m, dtype=float)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    prime_vertical_radius = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (prime_vertical_radius + height) * cos_lat * np.cos(lon)
    y = (prime_vertical_radius + height) * cos_lat * np.sin(lon)
    z = (prime_vertical_radius * (1.0 - WGS84_E2) + height) * sin_lat
    return x, y, z


def ecef_to_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    x0: float,
    y0: float,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    dx, dy, dz = x - x0, y - y0, z - z0
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def recompute_pair_enu(pair: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recompute both vehicles' WGS84 ECEF/ENU coordinates for this release."""
    output = {f"{color}_{axis}": np.empty(len(pair), dtype=float) for color in ("red", "black") for axis in ("east", "north", "up")}
    for _, group in pair.groupby("trial_id", sort=False):
        index = group.index.to_numpy()
        first = group.iloc[0]
        lat0 = float((first["red_latitude"] + first["black_latitude"]) / 2.0)
        lon0 = float((first["red_longitude"] + first["black_longitude"]) / 2.0)
        height0 = float((first["red_height"] + first["black_height"]) / 2.0)
        x0, y0, z0 = geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([height0]))
        for color in ("red", "black"):
            x, y, z = geodetic_to_ecef(
                group[f"{color}_latitude"].to_numpy(),
                group[f"{color}_longitude"].to_numpy(),
                group[f"{color}_height"].to_numpy(),
            )
            east, north, up = ecef_to_enu(x, y, z, lat0, lon0, float(x0[0]), float(y0[0]), float(z0[0]))
            output[f"{color}_east"][index] = east
            output[f"{color}_north"][index] = north
            output[f"{color}_up"][index] = up
    return output


def role_array(pair: pd.DataFrame, role: str, red_values: np.ndarray, black_values: np.ndarray) -> np.ndarray:
    follower_is_red = pair["inferred_follower_color"].eq("Red").to_numpy()
    use_red = follower_is_red if role == "follower" else ~follower_is_red
    return np.where(use_red, red_values, black_values)


def make_koopman_table(pair: pd.DataFrame) -> pd.DataFrame:
    follower_mode = role_mode(pair, "follower")
    leader_mode = role_mode(pair, "leader")
    case_id = pd.Series(follower_mode).astype(str) + "_follows_" + pd.Series(leader_mode).astype(str)
    run_index = pair["run_index"].astype(int).to_numpy()
    trial_uid = case_id + "_run" + pd.Series(run_index).astype(str).str.zfill(2)

    enu = recompute_pair_enu(pair)
    f_e = role_array(pair, "follower", enu["red_east"], enu["black_east"])
    f_n = role_array(pair, "follower", enu["red_north"], enu["black_north"])
    f_u = role_array(pair, "follower", enu["red_up"], enu["black_up"])
    l_e = role_array(pair, "leader", enu["red_east"], enu["black_east"])
    l_n = role_array(pair, "leader", enu["red_north"], enu["black_north"])
    l_u = role_array(pair, "leader", enu["red_up"], enu["black_up"])

    f_az = role_values(pair, "follower", "azimuth").astype(float)
    l_az = role_values(pair, "leader", "azimuth").astype(float)
    f_az_rad = np.deg2rad(f_az)
    l_az_rad = np.deg2rad(l_az)
    fh_e, fh_n = np.sin(f_az_rad), np.cos(f_az_rad)
    fl_e, fl_n = -fh_n, fh_e
    lh_e, lh_n = np.sin(l_az_rad), np.cos(l_az_rad)
    ll_e, ll_n = -lh_n, lh_e

    f_ve = role_values(pair, "follower", "east_velocity").astype(float)
    f_vn = role_values(pair, "follower", "north_velocity").astype(float)
    f_vu = role_values(pair, "follower", "up_velocity").astype(float)
    l_ve = role_values(pair, "leader", "east_velocity").astype(float)
    l_vn = role_values(pair, "leader", "north_velocity").astype(float)
    l_vu = role_values(pair, "leader", "up_velocity").astype(float)

    dp_e, dp_n, dp_u = l_e - f_e, l_n - f_n, l_u - f_u
    dv_e, dv_n, dv_u = l_ve - f_ve, l_vn - f_vn, l_vu - f_vu
    sensor_spacing = dp_e * fh_e + dp_n * fh_n
    lateral_offset = dp_e * fl_e + dp_n * fl_n

    out = pd.DataFrame(
        {
            "case_id": case_id.to_numpy(),
            "trial_uid": trial_uid.to_numpy(),
            "run_index": run_index,
            "sample_index": pair.groupby([case_id.to_numpy(), run_index], sort=False).cumcount().to_numpy(),
            "sample_rate_hz": float(BASE_HZ),
            "t_rel_s": pair["t_rel_s"].to_numpy(dtype=float),
            "t_gps_s": pair["t_abs_s"].to_numpy(dtype=float),
            "follower_mode": follower_mode,
            "leader_mode": leader_mode,
            "follower_east_m": f_e,
            "follower_north_m": f_n,
            "follower_up_m": f_u,
            "leader_east_m": l_e,
            "leader_north_m": l_n,
            "leader_up_m": l_u,
            "sensor_spacing_longitudinal_m": sensor_spacing,
            "gap_m": sensor_spacing - GAP_OFFSET_M,
            "gap_offset_m": GAP_OFFSET_M,
            "lateral_offset_left_m": lateral_offset,
            "sensor_range_horizontal_m": np.hypot(dp_e, dp_n),
            "vertical_separation_m": dp_u,
            "follower_azimuth_deg": f_az,
            "leader_azimuth_deg": l_az,
            "leader_minus_follower_heading_deg": wrapped_degrees(l_az - f_az),
            "follower_longitudinal_velocity_mps": f_ve * fh_e + f_vn * fh_n,
            "follower_lateral_velocity_left_mps": f_ve * fl_e + f_vn * fl_n,
            "follower_vertical_velocity_mps": f_vu,
            "leader_longitudinal_velocity_mps": l_ve * lh_e + l_vn * lh_n,
            "leader_lateral_velocity_left_mps": l_ve * ll_e + l_vn * ll_n,
            "leader_vertical_velocity_mps": l_vu,
            "leader_velocity_in_follower_longitudinal_mps": l_ve * fh_e + l_vn * fh_n,
            "leader_velocity_in_follower_lateral_left_mps": l_ve * fl_e + l_vn * fl_n,
            "relative_longitudinal_velocity_mps": dv_e * fh_e + dv_n * fh_n,
            "relative_lateral_velocity_left_mps": dv_e * fl_e + dv_n * fl_n,
            "relative_vertical_velocity_mps": dv_u,
            "follower_longitudinal_acc_corrimu_mps2": role_values(pair, "follower", "longitudinal_acc_mps2"),
            "follower_lateral_acc_corrimu_mps2": role_values(pair, "follower", "lateral_acc_mps2"),
            "follower_vertical_acc_corrimu_mps2": role_values(pair, "follower", "vertical_acc_mps2"),
            "leader_longitudinal_acc_corrimu_mps2": role_values(pair, "leader", "longitudinal_acc_mps2"),
            "leader_lateral_acc_corrimu_mps2": role_values(pair, "leader", "lateral_acc_mps2"),
            "leader_vertical_acc_corrimu_mps2": role_values(pair, "leader", "vertical_acc_mps2"),
            "relative_longitudinal_acc_corrimu_mps2": (
                role_values(pair, "leader", "longitudinal_acc_mps2")
                - role_values(pair, "follower", "longitudinal_acc_mps2")
            ),
            "follower_yaw_rate_corrimu_radps": role_values(pair, "follower", "yaw_rate_radps"),
            "leader_yaw_rate_corrimu_radps": role_values(pair, "leader", "yaw_rate_radps"),
        }
    )
    return out


def make_public_measurements(cleaned_raw: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "case_id",
        "trial_uid",
        "run_index",
        "sample_rate_hz",
        "t_rel_s",
        "t_gps_s",
        "follower_mode",
        "leader_mode",
    ]
    measured_suffixes = [
        "latitude_deg",
        "longitude_deg",
        "height_m",
        "north_velocity_mps",
        "east_velocity_mps",
        "up_velocity_mps",
        "roll_deg",
        "pitch_deg",
        "azimuth_deg",
        "corrimu_sample_dt_s",
        "pitch_delta_corrimu_rad",
        "roll_delta_corrimu_rad",
        "yaw_delta_corrimu_rad",
        "lateral_delta_v_corrimu_mps",
        "longitudinal_delta_v_corrimu_mps",
        "vertical_delta_v_corrimu_mps",
    ]
    columns = metadata + [f"{role}_{suffix}" for role in ("follower", "leader") for suffix in measured_suffixes]
    out = cleaned_raw.loc[:, columns].copy()
    out.insert(3, "sample_index", out.groupby("trial_uid", sort=False).cumcount())
    return out


def decimate(table: pd.DataFrame, hz: int) -> pd.DataFrame:
    factor = BASE_HZ // hz
    if BASE_HZ % hz:
        raise ValueError(f"{hz} Hz is not an integer divisor of {BASE_HZ} Hz")
    counts = table.groupby("trial_uid", sort=False).cumcount()
    out = table.loc[counts.mod(factor).eq(0)].copy()
    out["sample_rate_hz"] = float(hz)
    out["sample_index"] = out.groupby("trial_uid", sort=False).cumcount()
    return out.reset_index(drop=True)


def longest_true_run_s(values: pd.Series, dt_s: float) -> float:
    a = values.to_numpy(dtype=bool)
    if not a.any():
        return 0.0
    padded = np.r_[False, a, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return float(np.max(edges[1::2] - edges[::2]) * dt_s)


def make_quality_long(pair: pd.DataFrame, trial_ids: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for role in ("follower", "leader"):
        frame = pd.DataFrame(
            {
                "trial_uid": trial_ids["trial_uid"],
                "case_id": trial_ids["case_id"],
                "run_index": pair["run_index"].astype(int),
                "role": role,
                "mode": role_mode(pair, role),
                "t_rel_s": pair["t_rel_s"],
                "ins_status": role_values(pair, role, "ins_status_status"),
                "position_type": role_values(pair, role, "pos_type_type"),
                "latitude_stdev_m": role_values(pair, role, "latitude_stdev"),
                "longitude_stdev_m": role_values(pair, role, "longitude_stdev"),
                "height_stdev_m": role_values(pair, role, "height_stdev"),
                "north_velocity_stdev_mps": role_values(pair, role, "north_velocity_stdev"),
                "east_velocity_stdev_mps": role_values(pair, role, "east_velocity_stdev"),
                "up_velocity_stdev_mps": role_values(pair, role, "up_velocity_stdev"),
            }
        )
        frame["horizontal_position_stdev_m"] = np.hypot(
            frame["latitude_stdev_m"], frame["longitude_stdev_m"]
        )
        frame["horizontal_velocity_stdev_mps"] = np.hypot(
            frame["north_velocity_stdev_mps"], frame["east_velocity_stdev_mps"]
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_quality_manifest(quality_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["case_id", "trial_uid", "run_index", "role", "mode"]
    for keys, group in quality_long.groupby(group_cols, sort=True):
        ins = group["ins_status"]
        pos = group["position_type"]
        only_55_56 = bool(pos.isin([55, 56]).all())
        all_ins3 = bool(ins.eq(3).all())
        if all_ins3 and pos.eq(56).all():
            quality_class = "strict_ins_good_rtk_fixed"
        elif all_ins3 and only_55_56:
            quality_class = "accepted_ins_good_rtk_float_or_fixed"
        else:
            quality_class = "review_required"
        rows.append(
            {
                "case_id": keys[0],
                "trial_uid": keys[1],
                "run_index": int(keys[2]),
                "role": keys[3],
                "mode": keys[4],
                "rows": int(len(group)),
                "quality_class": quality_class,
                "ins3_ratio": float(ins.eq(3).mean()),
                "pos56_ratio": float(pos.eq(56).mean()),
                "pos55_ratio": float(pos.eq(55).mean()),
                "pos53_ratio": float(pos.eq(53).mean()),
                "pos0_ratio": float(pos.eq(0).mean()),
                "non56_longest_contiguous_s": longest_true_run_s(pos.ne(56), 1.0 / BASE_HZ),
                "horizontal_position_stdev_p95_m": float(group["horizontal_position_stdev_m"].quantile(0.95)),
                "horizontal_position_stdev_max_m": float(group["horizontal_position_stdev_m"].max()),
                "height_stdev_p95_m": float(group["height_stdev_m"].quantile(0.95)),
                "height_stdev_max_m": float(group["height_stdev_m"].max()),
                "horizontal_velocity_stdev_p95_mps": float(group["horizontal_velocity_stdev_mps"].quantile(0.95)),
                "horizontal_velocity_stdev_max_mps": float(group["horizontal_velocity_stdev_mps"].max()),
                "up_velocity_stdev_p95_mps": float(group["up_velocity_stdev_mps"].quantile(0.95)),
                "up_velocity_stdev_max_mps": float(group["up_velocity_stdev_mps"].max()),
            }
        )
    return pd.DataFrame(rows)


def make_quality_percentiles(quality_long: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "latitude_stdev_m",
        "longitude_stdev_m",
        "horizontal_position_stdev_m",
        "height_stdev_m",
        "north_velocity_stdev_mps",
        "east_velocity_stdev_mps",
        "horizontal_velocity_stdev_mps",
        "up_velocity_stdev_mps",
    ]
    rows = []
    for position_type in (55, 56):
        subset = quality_long.loc[
            quality_long["ins_status"].eq(3) & quality_long["position_type"].eq(position_type)
        ]
        for metric in metrics:
            values = subset[metric].dropna()
            q = values.quantile([0.25, 0.50, 0.75, 0.95, 1.0])
            rows.append(
                {
                    "ins_status": 3,
                    "position_type": position_type,
                    "metric": metric,
                    "n": int(len(values)),
                    "p25": float(q.loc[0.25]),
                    "p50": float(q.loc[0.50]),
                    "p75": float(q.loc[0.75]),
                    "p95": float(q.loc[0.95]),
                    "max": float(q.loc[1.0]),
                }
            )
    return pd.DataFrame(rows)


def window_summary(csv_path: Path, duration_s: int) -> dict:
    columns = []
    for color in ("red", "black"):
        columns.extend(
            [
                f"{color}_ins_status_status",
                f"{color}_pos_type_type",
                f"{color}_latitude_stdev",
                f"{color}_longitude_stdev",
            ]
        )
    data = pd.read_csv(csv_path, usecols=["trial_id", *columns])
    ins = pd.concat([data["red_ins_status_status"], data["black_ins_status_status"]], ignore_index=True)
    pos = pd.concat([data["red_pos_type_type"], data["black_pos_type_type"]], ignore_index=True)
    hpos = pd.concat(
        [
            np.hypot(data["red_latitude_stdev"], data["red_longitude_stdev"]),
            np.hypot(data["black_latitude_stdev"], data["black_longitude_stdev"]),
        ],
        ignore_index=True,
    )
    return {
        "window_duration_s": duration_s,
        "trials": int(data["trial_id"].nunique()),
        "vehicle_samples": int(len(pos)),
        "ins3_ratio": float(ins.eq(3).mean()),
        "pos56_ratio": float(pos.eq(56).mean()),
        "pos55_ratio": float(pos.eq(55).mean()),
        "pos53_ratio": float(pos.eq(53).mean()),
        "pos0_ratio": float(pos.eq(0).mean()),
        "pos55_or_56_ratio": float(pos.isin([55, 56]).mean()),
        "horizontal_position_stdev_p25_m": float(hpos.quantile(0.25)),
        "horizontal_position_stdev_p50_m": float(hpos.quantile(0.50)),
        "horizontal_position_stdev_p75_m": float(hpos.quantile(0.75)),
        "horizontal_position_stdev_p95_m": float(hpos.quantile(0.95)),
        "horizontal_position_stdev_max_m": float(hpos.max()),
    }


def trial_table(pair: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    temp = pair.loc[:, [
        "trial_id", "config", "run_index", "t_abs_s", "t_rel_s",
        "red_latitude", "black_latitude", "red_longitude", "black_longitude",
        "red_height", "black_height", "orientation_confidence",
        "selected_positive_ratio", "selected_median_abs_lateral_offset_m",
        "inferred_follower_color",
    ]].copy()
    temp["case_id"] = ids["case_id"].to_numpy()
    temp["trial_uid"] = ids["trial_uid"].to_numpy()
    rows = []
    for trial_uid, group in temp.groupby("trial_uid", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "case_id": first["case_id"],
                "trial_uid": trial_uid,
                "run_index": int(first["run_index"]),
                "follower_mode": first["case_id"].split("_follows_")[0],
                "leader_mode": first["case_id"].split("_follows_")[1],
                "window_start_gps_s": float(group["t_abs_s"].min()),
                "window_end_gps_s_exclusive": float(group["t_abs_s"].min() + WINDOW_S),
                "window_duration_s": WINDOW_S,
                "rows_125hz": int(len(group)),
                "enu_origin_latitude_deg": float((first["red_latitude"] + first["black_latitude"]) / 2.0),
                "enu_origin_longitude_deg": float((first["red_longitude"] + first["black_longitude"]) / 2.0),
                "enu_origin_height_m": float((first["red_height"] + first["black_height"]) / 2.0),
                "local_frame": "WGS84 geodetic -> ECEF -> per-trial ENU",
                "orientation_confidence": first["orientation_confidence"],
                "orientation_positive_ratio": float(first["selected_positive_ratio"]),
                "orientation_median_abs_lateral_offset_m": float(first["selected_median_abs_lateral_offset_m"]),
                "gap_offset_m": GAP_OFFSET_M,
                "audit_source_trial_id": first["trial_id"],
                "audit_source_config": first["config"],
                "audit_follower_color": first["inferred_follower_color"],
            }
        )
    return pd.DataFrame(rows)


def source_lineage(trials: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    rows = []
    for row in trials.itertuples(index=False):
        match = re.search(r"_(\d{6})_(\d{6})$", row.audit_source_trial_id)
        if not match:
            raise ValueError(f"Cannot parse source times from {row.audit_source_trial_id}")
        red_hms, black_hms = match.groups()
        for role in ("follower", "leader"):
            color = row.audit_follower_color if role == "follower" else ("Black" if row.audit_follower_color == "Red" else "Red")
            hms = red_hms if color == "Red" else black_hms
            log = raw_root / row.audit_source_config / f"SI_{color}_20250824_{hms}" / "csv_out"
            insp = log / "bynav_inspvax.csv"
            corr = log / "bynav_corrimudata.csv"
            rows.append(
                {
                    "case_id": row.case_id,
                    "trial_uid": row.trial_uid,
                    "run_index": row.run_index,
                    "role": role,
                    "source_inspvax_csv": insp.relative_to(raw_root.parents[1]).as_posix(),
                    "source_corrimu_csv": corr.relative_to(raw_root.parents[1]).as_posix(),
                    "inspvax_logical_bytes": insp.stat().st_size if insp.exists() else np.nan,
                    "inspvax_materialized_at_build": bool(insp.exists() and getattr(insp.stat(), "st_blocks", 0) > 0),
                    "corrimu_logical_bytes": corr.stat().st_size if corr.exists() else np.nan,
                    "corrimu_materialized_at_build": bool(corr.exists() and getattr(corr.stat(), "st_blocks", 0) > 0),
                }
            )
    return pd.DataFrame(rows)


def split_manifest(top2_path: Path, trials: pd.DataFrame) -> pd.DataFrame:
    top2 = pd.read_csv(top2_path)
    mapping = trials.loc[:, ["audit_source_config", "run_index", "case_id", "trial_uid"]]
    mapping = mapping.rename(columns={"audit_source_config": "config"})
    out = top2.merge(mapping, on=["config", "run_index"], how="left", validate="one_to_one")
    return out.loc[:, ["case_id", "trial_uid", "run_index", "top2_quality_rank", "selected_top2_quality"]].rename(
        columns={
            "top2_quality_rank": "strict_subset_rank_within_case",
            "selected_top2_quality": "strict_top2_selected",
        }
    ).assign(primary_all_3_runs=True)


def add_dictionary_row(rows: list[dict], group: str, column: str, dtype: str, unit: str, source: str, description: str, transformation: str) -> None:
    rows.append(
        {
            "file_group": group,
            "column": column,
            "data_type": dtype,
            "unit": unit,
            "source": source,
            "description": description,
            "transformation": transformation,
        }
    )


def make_data_dictionary(public_columns: list[str], koopman_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    common = {
        "case_id": ("string", "-", "metadata", "Follower/leader mode pairing", "role relabeling only"),
        "trial_uid": ("string", "-", "metadata", "Unique trial identifier", "case plus run index"),
        "run_index": ("integer", "-", "metadata", "Run number within case", "1 to 3"),
        "sample_index": ("integer", "-", "time grid", "Zero-based sample index within trial", "enumerated after decimation"),
        "sample_rate_hz": ("float", "Hz", "time grid", "Nominal output rate", "fixed by file"),
        "t_rel_s": ("float", "s", "GPS time", "Time from selected window start", "t_gps minus window start"),
        "t_gps_s": ("float", "s", "NovAtel GPS week time", "Absolute GPS time", "week*604800 + milliseconds/1000"),
        "follower_mode": ("string", "-", "experiment metadata", "Follower control mode", "role relabeling only"),
        "leader_mode": ("string", "-", "experiment metadata", "Leader control mode", "role relabeling only"),
    }
    for group, columns in (("public_measurements", public_columns), ("koopman", koopman_columns)):
        for column in columns:
            if column in common:
                dtype, unit, source, desc, transform = common[column]
                add_dictionary_row(rows, group, column, dtype, unit, source, desc, transform)

    public_specs = {
        "latitude_deg": ("degree", "INSPVAX", "Geodetic latitude"),
        "longitude_deg": ("degree", "INSPVAX", "Geodetic longitude"),
        "height_m": ("m", "INSPVAX", "Ellipsoidal height"),
        "north_velocity_mps": ("m/s", "INSPVAX", "North velocity"),
        "east_velocity_mps": ("m/s", "INSPVAX", "East velocity"),
        "up_velocity_mps": ("m/s", "INSPVAX", "Up velocity"),
        "roll_deg": ("degree", "INSPVAX", "Roll angle"),
        "pitch_deg": ("degree", "INSPVAX", "Pitch angle"),
        "azimuth_deg": ("degree", "INSPVAX", "Clockwise heading from north"),
        "corrimu_sample_dt_s": ("s", "CORRIMU GPS time", "Source sample interval"),
        "pitch_delta_corrimu_rad": ("rad/sample", "CORRIMU", "Pitch angle increment"),
        "roll_delta_corrimu_rad": ("rad/sample", "CORRIMU", "Roll angle increment"),
        "yaw_delta_corrimu_rad": ("rad/sample", "CORRIMU", "Yaw angle increment"),
        "lateral_delta_v_corrimu_mps": ("m/s per sample", "CORRIMU", "Lateral delta velocity"),
        "longitudinal_delta_v_corrimu_mps": ("m/s per sample", "CORRIMU", "Longitudinal delta velocity"),
        "vertical_delta_v_corrimu_mps": ("m/s per sample", "CORRIMU", "Vertical delta velocity"),
    }
    for column in public_columns:
        for suffix, (unit, source, desc) in public_specs.items():
            if column.endswith(suffix):
                add_dictionary_row(rows, "public_measurements", column, "float", unit, source, desc, "linear GPS-time synchronization to 125 Hz; no physical-variable derivation")
                break

    koopman_specs = {
        "follower_east_m": ("m", "ENU", "Follower east coordinate"),
        "follower_north_m": ("m", "ENU", "Follower north coordinate"),
        "follower_up_m": ("m", "ENU", "Follower up coordinate"),
        "leader_east_m": ("m", "ENU", "Leader east coordinate"),
        "leader_north_m": ("m", "ENU", "Leader north coordinate"),
        "leader_up_m": ("m", "ENU", "Leader up coordinate"),
        "sensor_spacing_longitudinal_m": ("m", "ENU and follower heading", "Leader-minus-follower sensor displacement projected longitudinally"),
        "gap_m": ("m", "derived", "Approximate bumper gap"),
        "gap_offset_m": ("m", "assumption", "Shared sensor-to-bumper correction"),
        "lateral_offset_left_m": ("m", "ENU and follower heading", "Leader lateral offset; positive left"),
        "sensor_range_horizontal_m": ("m", "ENU", "Horizontal sensor-to-sensor range"),
        "vertical_separation_m": ("m", "ENU", "Leader up minus follower up"),
        "follower_azimuth_deg": ("degree", "INSPVAX", "Follower azimuth clockwise from north"),
        "leader_azimuth_deg": ("degree", "INSPVAX", "Leader azimuth clockwise from north"),
        "leader_minus_follower_heading_deg": ("degree", "derived", "Wrapped heading difference in [-180,180)"),
        "follower_longitudinal_velocity_mps": ("m/s", "INSPVAX projected", "Follower velocity along own heading"),
        "follower_lateral_velocity_left_mps": ("m/s", "INSPVAX projected", "Follower velocity left of own heading"),
        "follower_vertical_velocity_mps": ("m/s", "INSPVAX", "Follower up velocity"),
        "leader_longitudinal_velocity_mps": ("m/s", "INSPVAX projected", "Leader velocity along own heading"),
        "leader_lateral_velocity_left_mps": ("m/s", "INSPVAX projected", "Leader velocity left of own heading"),
        "leader_vertical_velocity_mps": ("m/s", "INSPVAX", "Leader up velocity"),
        "leader_velocity_in_follower_longitudinal_mps": ("m/s", "INSPVAX projected", "Leader velocity in follower longitudinal axis"),
        "leader_velocity_in_follower_lateral_left_mps": ("m/s", "INSPVAX projected", "Leader velocity in follower left axis"),
        "relative_longitudinal_velocity_mps": ("m/s", "INSPVAX projected", "Leader-minus-follower velocity in follower longitudinal axis"),
        "relative_lateral_velocity_left_mps": ("m/s", "INSPVAX projected", "Leader-minus-follower velocity in follower left axis"),
        "relative_vertical_velocity_mps": ("m/s", "INSPVAX", "Leader-minus-follower up velocity"),
        "follower_longitudinal_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Follower longitudinal acceleration"),
        "follower_lateral_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Follower lateral acceleration"),
        "follower_vertical_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Follower vertical acceleration"),
        "leader_longitudinal_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Leader longitudinal acceleration"),
        "leader_lateral_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Leader lateral acceleration"),
        "leader_vertical_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Leader vertical acceleration"),
        "relative_longitudinal_acc_corrimu_mps2": ("m/s^2", "CORRIMU", "Leader-minus-follower longitudinal acceleration"),
        "follower_yaw_rate_corrimu_radps": ("rad/s", "CORRIMU", "Follower yaw rate"),
        "leader_yaw_rate_corrimu_radps": ("rad/s", "CORRIMU", "Leader yaw rate"),
    }
    for column in koopman_columns:
        if column not in koopman_specs:
            continue
        unit, source, desc = koopman_specs[column]
        if "corrimu" in column:
            transform = "source increment divided by per-row source CORRIMU sample interval"
        elif column == "gap_m":
            transform = "sensor_spacing_longitudinal_m - gap_offset_m"
        elif column == "gap_offset_m":
            transform = "constant 3.0 m"
        else:
            transform = "WGS84 ECEF/ENU conversion and/or follower-frame projection as named"
        add_dictionary_row(rows, "koopman", column, "float", unit, source, desc, transform)
    return pd.DataFrame(rows).sort_values(["file_group", "column"]).reset_index(drop=True)




def assert_release(public: pd.DataFrame, rates: dict[int, pd.DataFrame], quality: pd.DataFrame, splits: pd.DataFrame) -> dict:
    expected_rows = {125: 1875, 25: 375, 5: 75, 1: 15}
    checks = {}
    checks["public_trials"] = int(public["trial_uid"].nunique())
    checks["public_rows"] = int(len(public))
    checks["public_has_no_status_or_stdev"] = not any(
        "status" in c.lower() or "stdev" in c.lower() for c in public.columns
    )
    checks["public_has_no_red_black_columns"] = not any(
        "red" in c.lower() or "black" in c.lower() for c in public.columns
    )
    checks["quality_all_ins3"] = bool(quality["ins3_ratio"].eq(1.0).all())
    checks["quality_only_pos55_or_56"] = bool(quality[["pos53_ratio", "pos0_ratio"]].eq(0.0).all().all())
    checks["strict_subset_trials"] = int(splits["strict_top2_selected"].sum())
    checks["primary_trials"] = int(splits["primary_all_3_runs"].sum())
    for hz, table in rates.items():
        counts = table.groupby("trial_uid").size()
        checks[f"koopman_{hz}hz_trials"] = int(table["trial_uid"].nunique())
        checks[f"koopman_{hz}hz_rows"] = int(len(table))
        checks[f"koopman_{hz}hz_rows_per_trial_ok"] = bool(counts.eq(expected_rows[hz]).all())
        checks[f"koopman_{hz}hz_unique_key"] = not table.duplicated(["trial_uid", "sample_index"]).any()
        checks[f"koopman_{hz}hz_critical_nan_count"] = int(
            table[[
                "gap_m", "relative_longitudinal_velocity_mps",
                "follower_longitudinal_velocity_mps", "leader_longitudinal_velocity_mps",
            ]].isna().sum().sum()
        )
    required_true = [
        "public_has_no_status_or_stdev", "public_has_no_red_black_columns",
        "quality_all_ins3", "quality_only_pos55_or_56",
        *[f"koopman_{hz}hz_rows_per_trial_ok" for hz in RATES_HZ],
        *[f"koopman_{hz}hz_unique_key" for hz in RATES_HZ],
    ]
    if checks["public_trials"] != 42 or checks["primary_trials"] != 42 or checks["strict_subset_trials"] != 28:
        raise AssertionError(checks)
    if not all(checks[name] for name in required_true):
        raise AssertionError(checks)
    if any(checks[f"koopman_{hz}hz_critical_nan_count"] for hz in RATES_HZ):
        raise AssertionError(checks)
    return checks


