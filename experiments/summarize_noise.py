"""Aggregate paired errors over seeds and origins before averaging encounters."""

from src.paths import OUTPUTS
from src.data.replays import NOISE, ORIGINS

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(folder):
    blocks = [pd.read_csv(folder / "response_physics.csv")]
    blocks.extend(
        pd.read_csv(f) for f in sorted((folder / "neural").glob("field_*.csv"))
    )
    data = pd.concat(blocks, ignore_index=True)
    return summarize_responses(data, folder)


def summarize_responses(data, folder):
    metrics = ["change", "signed_change", "truth_rmse"]
    keys = ["dataset", "object", "uid", "model", "noise", "origin", "noise_seed"]
    paired_keys = ["dataset", "object", "uid", "origin", "noise", "noise_seed"]
    per_record = data.groupby(keys[:3], sort=False).size()
    scenarios = set(zip(data.noise, data.noise_seed))
    if (
        len(per_record) != 3380
        or data.duplicated(keys).any()
        or scenarios != set(NOISE)
        or set(data.origin) != set(ORIGINS)
        or data.model.nunique() != 13
        or not per_record.eq(13 * len(NOISE) * len(ORIGINS)).all()
        or not data.groupby(paired_keys).size().eq(13).all()
    ):
        raise ValueError("Incomplete paired response comparison")
    if (
        not np.isfinite(data[metrics[:2]]).all().all()
        or not np.isfinite(data.loc[data.object != "recorded", "truth_rmse"]).all()
    ):
        raise ValueError("Nonfinite response")
    encounters = data.groupby(keys[:5], as_index=False)[metrics].mean()
    summary = encounters.groupby(
        ["dataset", "object", "noise", "model"], as_index=False
    )[metrics].mean()
    summary.to_csv(folder / "by_follower.csv", index=False)
    summary["cohort"] = np.where(summary.object == "recorded", "recorded", "simulated")
    means = summary.groupby(["dataset", "cohort", "noise", "model"], as_index=False)[
        metrics
    ].mean()
    means.to_csv(folder / "summary.csv", index=False)
    return means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=(OUTPUTS / "noise"))
    summarize(parser.parse_args().input)
