#!/usr/bin/env python3
"""Build the four-duration 0824 release from the retained working tables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _release_core import (
    BASE_HZ,
    FLOAT_FORMAT,
    RATES_HZ,
    decimate,
    make_data_dictionary,
    make_koopman_table,
    make_quality_long,
    make_quality_manifest,
)
from _window_selection import (
    add_case_columns,
    best_window,
    make_public_measurements,
    quality_percentiles_present,
    rank_and_select,
    refresh_source_lineage,
    source_materialization_summary,
    status_arrays,
    summarize_window,
    trial_manifest,
)


def select_all_runs(
    pair: pd.DataFrame,
    duration_s: int,
    optimize_subwindow: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the best fixed-length window in every run without dropping a run."""
    pair = add_case_columns(pair)
    windows: list[pd.DataFrame] = []
    rows: list[dict] = []
    width = int(duration_s * BASE_HZ)

    for _, group in pair.groupby("trial_id", sort=False):
        group = group.sort_values("t_rel_s").reset_index(drop=True)
        if optimize_subwindow:
            window, metrics = best_window(group, duration_s)
        else:
            if len(group) != width:
                raise ValueError(
                    f"Expected {width} rows in {group['trial_id'].iloc[0]}; found {len(group)}"
                )
            window = group.copy()
            metrics = summarize_window(group, 0, width)
        windows.append(window)
        rows.append(metrics)

    manifest = rank_and_select(pd.DataFrame(rows))
    all_runs = pd.concat(windows, ignore_index=True)
    all_runs = all_runs.sort_values(["_case_id", "run_index", "t_abs_s"]).reset_index(drop=True)
    all_runs["t_rel_s"] = all_runs.groupby("_trial_uid", sort=False).cumcount() / BASE_HZ
    return all_runs, manifest


def status_summary(pair: pd.DataFrame) -> dict:
    status = status_arrays(pair)
    hpos = np.r_[
        np.hypot(pair["red_latitude_stdev"], pair["red_longitude_stdev"]),
        np.hypot(pair["black_latitude_stdev"], pair["black_longitude_stdev"]),
    ]
    vehicle_samples = int(len(pair) * 2)
    pos56 = int(status["pos56"].sum())
    return {
        "trials": int(pair["_trial_uid"].nunique()),
        "cases": int(pair["_case_id"].nunique()),
        "runs_per_case": int(pair.drop_duplicates("_trial_uid").groupby("_case_id").size().min()),
        "vehicle_samples": vehicle_samples,
        "ins_not3_samples": int(status["ins_bad"].sum()),
        "ins0_samples": int(status["ins0"].sum()),
        "pos0_or53_samples": int(status["pos_severe"].sum()),
        "pos0_samples": int(status["pos0"].sum()),
        "pos53_samples": int(status["pos53"].sum()),
        "pos55_samples": int(status["pos55"].sum()),
        "pos56_samples": pos56,
        "pos56_ratio": float(pos56 / vehicle_samples),
        "horizontal_position_stdev_p95_m": float(np.quantile(hpos, 0.95)),
        "horizontal_position_stdev_max_m": float(np.max(hpos)),
    }


def top2_status_summary(selection: pd.DataFrame) -> dict:
    chosen = selection.loc[selection["selected_top2"]]
    vehicle_samples = int(chosen["vehicle_samples"].sum())
    pos56 = int(chosen["pos56_samples"].sum())
    return {
        "trials": int(len(chosen)),
        "vehicle_samples": vehicle_samples,
        "ins_not3_samples": int(chosen["ins_not3_samples"].sum()),
        "ins0_samples": int(chosen["ins0_samples"].sum()),
        "pos0_or53_samples": int(chosen["pos0_or53_samples"].sum()),
        "pos0_samples": int(chosen["pos0_samples"].sum()),
        "pos53_samples": int(chosen["pos53_samples"].sum()),
        "pos55_samples": int(chosen["pos55_samples"].sum()),
        "pos56_samples": pos56,
        "pos56_ratio": float(pos56 / vehicle_samples),
    }


def make_cv_manifest(selection: pd.DataFrame) -> pd.DataFrame:
    cv = selection.loc[:, [
        "case_id",
        "trial_uid",
        "source_trial_id",
        "run_index",
        "quality_rank_within_case",
        "selected_top2",
        "ins_not3_samples",
        "pos0_samples",
        "pos53_samples",
        "pos55_samples",
        "pos56_samples",
        "horizontal_position_stdev_p95_m",
        "horizontal_position_stdev_max_m",
    ]].copy()
    cv = cv.rename(columns={"selected_top2": "include_in_top2_quality_subset"})
    cv["include_in_all3_dataset"] = True
    cv["leave_one_run_out_test_fold"] = cv["run_index"].astype(int)
    cv["quality_subset_role"] = np.where(
        cv["include_in_top2_quality_subset"],
        "quality_top2",
        "third_ranked_run",
    )
    return cv.sort_values(["case_id", "run_index"]).reset_index(drop=True)


def write_variant(
    root: Path,
    name: str,
    pair: pd.DataFrame,
    selection: pd.DataFrame,
    duration_s: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    variant = root / name
    public_dir = variant / "lightly_cleaned"
    koopman_dir = variant / "koopman_ready"
    metadata_dir = variant / "metadata"
    for directory in (public_dir, koopman_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all3 = status_summary(pair)
    top2 = top2_status_summary(selection)
    summary = {
        "duration_s": duration_s,
        "all3_dataset": all3,
        "top2_quality_subset": top2,
    }
    if all3["trials"] != 42 or all3["cases"] != 14 or all3["runs_per_case"] != 3:
        raise AssertionError(summary)

    clean_pair = pair.drop(
        columns=[column for column in pair.columns if column.startswith("_")],
        errors="ignore",
    ).reset_index(drop=True)
    koopman125 = make_koopman_table(clean_pair)
    public = make_public_measurements(clean_pair, koopman125)
    rates = {hz: decimate(koopman125, hz) for hz in RATES_HZ}
    ids = koopman125.loc[:, ["case_id", "trial_uid"]]
    quality_long = make_quality_long(clean_pair, ids)
    quality = make_quality_manifest(quality_long)
    percentiles = quality_percentiles_present(quality_long)
    trials = trial_manifest(clean_pair, koopman125, duration_s)
    cv = make_cv_manifest(selection)

    if any("red" in column.lower() or "black" in column.lower() for column in public.columns):
        raise AssertionError("Public measurement table exposes Red/Black columns")
    if any("status" in column.lower() or "stdev" in column.lower() for column in public.columns):
        raise AssertionError("Public measurement table exposes QA columns")

    public.to_csv(public_dir / "measurements_125hz.csv", index=False, float_format=FLOAT_FORMAT)
    for hz, table in rates.items():
        expected = int(duration_s * hz)
        if not table.groupby("trial_uid").size().eq(expected).all():
            raise AssertionError(f"Unexpected {hz} Hz row count in {name}")
        if table.duplicated(["trial_uid", "sample_index"]).any():
            raise AssertionError(f"Duplicate model keys in {name} at {hz} Hz")
        table.to_csv(koopman_dir / f"koopman_{hz}hz.csv", index=False, float_format=FLOAT_FORMAT)

    selection.to_csv(
        metadata_dir / "selection_manifest_all_runs.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )
    cv.to_csv(metadata_dir / "cv_fold_manifest.csv", index=False, float_format=FLOAT_FORMAT)
    trials.to_csv(metadata_dir / "trial_manifest.csv", index=False, float_format=FLOAT_FORMAT)
    quality.to_csv(metadata_dir / "quality_manifest.csv", index=False, float_format=FLOAT_FORMAT)
    percentiles.to_csv(
        metadata_dir / "quality_percentiles_by_position_type.csv",
        index=False,
        float_format=FLOAT_FORMAT,
    )
    (metadata_dir / "qa_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return public, koopman125, summary






