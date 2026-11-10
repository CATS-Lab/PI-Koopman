"""Aggregate paired errors over seeds and origins before averaging encounters."""

from src.paths import OUTPUTS

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(folder):
    metrics = ["change", "signed_change", "truth_rmse"]
    blocks = [pd.read_csv(folder / "response_physics.csv")]
    blocks.extend(
        pd.read_csv(f) for f in sorted((folder / "neural").glob("field_*.csv"))
    )
    data = pd.concat(blocks, ignore_index=True)
    keys = ["dataset", "object", "uid", "model", "noise", "origin", "noise_seed"]
    if data.duplicated(keys).any() or len(data) != 3380 * 13 * 3 * 7:
        raise ValueError(
            "Expected 3,380 records, 13 methods, three origins and seven noise realizations"
        )
    by = data.groupby(keys[:3], sort=False)
    if not by.size().eq(13 * 3 * 7).all() or not by.model.nunique().eq(13).all():
        raise ValueError("Incomplete method comparison")
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
