"""Build one synchronized car-following table per CATS-CF field trial."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data.geometry import BASE_HZ, GAP_OFFSET_M, geodetic_to_ecef, ecef_to_enu
from src.data.windows import paper_window_start

LOG = re.compile(r"SI_(Red|Black)_(\d{8})_(\d{6})")
MODES = {"ACC1", "ACC2", "HV1", "HV2"}
GPS_WEEK = "nov_header.gps_week_number"
GPS_MS = "nov_header.gps_week_milliseconds"


def gps_seconds(frame):
    return frame[GPS_WEEK].to_numpy() * 604800.0 + frame[GPS_MS].to_numpy() / 1000.0


def paired_recordings(raw):
    for condition in sorted(Path(raw).iterdir()):
        if not condition.is_dir():
            continue
        first, second = condition.name.split("_")
        if {first[0], second[0]} != {"r", "b"} or not {first[1:], second[1:]} <= MODES:
            raise ValueError(f"Unknown recording condition {condition.name}")
        groups = {"Red": [], "Black": []}
        for log in sorted(condition.iterdir()):
            if match := LOG.fullmatch(log.name):
                _, date, clock = match.groups()
                stamp = pd.Timestamp(
                    f"{date} {clock[:2]}:{clock[2:4]}:{clock[4:]}"
                ).value
                groups[match[1]].append((stamp, log / "csv_out"))
        if len(groups["Red"]) != len(groups["Black"]) or not groups["Red"]:
            raise ValueError(f"Unpaired vehicle recordings under {condition}")
        black = groups["Black"].copy()
        pairs = []
        for red_time, red_dir in groups["Red"]:
            mate = min(black, key=lambda entry: abs(entry[0] - red_time))
            black.remove(mate)
            paths = {"r": red_dir, "b": mate[1]}
            pairs.append((red_time + mate[0], paths))
        case = f"{second[1:]}_follows_{first[1:]}"
        for run, (_, paths) in enumerate(
            sorted(pairs, key=lambda item: item[0]), start=1
        ):
            yield case, run, {"leader": paths[first[0]], "follower": paths[second[0]]}


def following_table(paths, uid):
    nav = {
        role: pd.read_csv(folder / "bynav_inspvax.csv")
        for role, folder in paths.items()
    }
    clocks = {role: gps_seconds(frame) for role, frame in nav.items()}
    if any(np.any(np.diff(t) <= 0) for t in clocks.values()):
        raise ValueError(f"Nonmonotonic navigation clock in {uid}")
    anchor = min(t[0] for t in clocks.values())
    common_start = max(t[0] for t in clocks.values())
    origin = [
        np.mean(
            [np.interp(common_start, clocks[k], frame[col]) for k, frame in nav.items()]
        )
        for col in ("latitude", "longitude", "height")
    ]
    xyz0 = geodetic_to_ecef(*origin)
    parts = []
    for role, frame in nav.items():
        values = frame[["ins_status.status", "pos_type.type"]].rename(
            columns={
                "ins_status.status": f"{role}_ins_status",
                "pos_type.type": f"{role}_position_type",
            }
        )
        values.index = pd.Index(clocks[role] - anchor, name="t_rel_s")
        parts.append(values)
    status = pd.concat(parts, axis=1).sort_index().reset_index()
    start = paper_window_start(status, 30)
    relative = np.arange(30 * BASE_HZ) / BASE_HZ
    target = start + relative
    out = pd.DataFrame({"t_s": relative})

    for role, frame in nav.items():
        t = clocks[role] - anchor
        xyz = geodetic_to_ecef(frame.latitude, frame.longitude, frame.height)
        east, north, _ = ecef_to_enu(*xyz, *origin[:2], *xyz0)
        fields = {
            "x_m": east,
            "y_m": north,
            "vx_mps": frame.east_velocity,
            "vy_mps": frame.north_velocity,
            "heading_deg": frame.azimuth,
        }
        for name, values in fields.items():
            out[f"{role}_{name}"] = np.interp(target, t, values)
        out[f"{role}_position_std_m"] = np.hypot(
            np.interp(target, t, frame.latitude_stdev),
            np.interp(target, t, frame.longitude_stdev),
        )
        angle = np.deg2rad(out[f"{role}_heading_deg"])
        out[f"{role}_v_lon_mps"] = out[f"{role}_vx_mps"] * np.sin(angle) + out[
            f"{role}_vy_mps"
        ] * np.cos(angle)
        # Vehicle axes follow the receiver convention: longitudinal forward,
        # lateral right. Global x/y columns remain east/north.
        out[f"{role}_v_lat_mps"] = out[f"{role}_vx_mps"] * np.cos(angle) - out[
            f"{role}_vy_mps"
        ] * np.sin(angle)

        corr = pd.read_csv(paths[role] / "bynav_corrimudata.csv")
        imu_t = gps_seconds(corr) - anchor
        dt = np.diff(imu_t)
        if not len(dt) or not np.all(dt > 0):
            raise ValueError(f"Invalid corrected IMU clock in {uid}")
        # CORRIMUDATA contains angular/velocity increments per IMU sample.
        increments = pd.DataFrame({"t_rel_s": imu_t})
        for name, source in [
            ("yaw_rate_radps", "yaw_rate"),
            ("acc_lon_mps2", "longitudinal_acc"),
            ("acc_lat_mps2", "lateral_acc"),
        ]:
            increments[name] = corr[source].to_numpy() / np.r_[np.median(dt), dt]
        matched = pd.merge_asof(
            pd.DataFrame({"t_rel_s": t}),
            increments,
            on="t_rel_s",
            direction="nearest",
            tolerance=0.05,
        ).dropna()
        if (
            len(matched) < 2
            or target[0] < matched.t_rel_s.iloc[0] - 1e-8
            or target[-1] > matched.t_rel_s.iloc[-1] + 1e-8
        ):
            raise ValueError(f"Missing corrected IMU coverage in {uid}")
        for name in increments.columns[1:]:
            out[f"{role}_{name}"] = np.interp(target, matched.t_rel_s, matched[name])

    angle = np.deg2rad(out.follower_heading_deg)
    sin, cos = np.sin(angle), np.cos(angle)
    east, north = out.leader_x_m - out.follower_x_m, out.leader_y_m - out.follower_y_m
    out["distance_m"] = east * sin + north * cos
    out["gap_m"] = out.distance_m - GAP_OFFSET_M
    leader_along_follower = out.leader_vx_mps * sin + out.leader_vy_mps * cos
    out["relative_v_mps"] = leader_along_follower - out.follower_v_lon_mps
    fields = [
        "x_m",
        "y_m",
        "vx_mps",
        "vy_mps",
        "heading_deg",
        "v_lon_mps",
        "v_lat_mps",
        "acc_lon_mps2",
        "acc_lat_mps2",
        "yaw_rate_radps",
        "position_std_m",
    ]
    columns = ["t_s"] + [
        f"{role}_{name}" for role in ("leader", "follower") for name in fields
    ]
    columns += ["distance_m", "gap_m", "relative_v_mps"]
    out = out[columns]
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError(f"Nonfinite car-following measurements in {uid}")
    return out


def export_recordings(raw, output):
    runs = list(paired_recordings(raw))
    keys = {(case, run) for case, run, _ in runs}
    if len(runs) != 42 or len(keys) != len(runs):
        raise ValueError("Expected 42 distinct CATS-CF trials")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for case, run, paths in runs:
        uid = f"{case}_run{run:02d}"
        following_table(paths, uid).to_csv(
            output / f"{uid}.csv", index=False, float_format="%.17g"
        )
    return len(runs)
