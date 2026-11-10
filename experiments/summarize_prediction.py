"""Aggregate complete seven-model main-text prediction axes without plots."""

from src.paths import OUTPUTS

import argparse, json, pathlib
import pandas as pd

root = pathlib.Path(__file__).resolve().parents[1]
from src.config import cells, DEFAULT_CONFIG


def select_rate_physics(candidates, source):
    """Transfer the source-selected reference configuration to each observation clock."""
    selected = []
    for (dataset, model, fold), target in candidates.groupby(
        ["dataset", "model", "group"]
    ):
        development = source[
            (source.dataset == dataset)
            & (source.model == model)
            & (source.group != fold)
        ]
        chosen = json.loads(development.groupby("config").score.mean().idxmin())
        for rate, group in target.groupby("rate_hz"):
            config = chosen.copy()
            if "T" in config:
                config["T"] = max(2 / rate, config["T"])
            rows = group[group.config.map(json.loads).map(lambda c: c == config)]
            if len(rows) != len(group[["trial_uid", "origin_s"]].drop_duplicates()):
                raise ValueError(
                    f"Missing transferred configuration for {model} at {rate} Hz"
                )
            selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["duration", "rate"], required=True)
    p.add_argument("--dataset", choices=["local", "openacc"], default="local")
    p.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    a = p.parse_args()
    if a.axis == "rate" and a.dataset != "local":
        p.error("Rate axis is local only.")
    grid = cells(a.axis, a.dataset, a.config)
    settings = []
    pi_choices = []
    frames = []
    rate_candidates = []
    for _, b, r, o in grid:
        name = f"{a.dataset}_B{b}_Hz{r}_end{o}"
        paths = [
            OUTPUTS / "physics/grid" / name / "selected_metrics.csv",
            OUTPUTS / "neural/grid" / name / "selected_encounter_metrics.csv",
        ]
        if a.axis == "duration":
            nn = pd.read_csv(
                paths[1].with_name("selected_raw_initializations.csv"),
                dtype={"group": str},
            )
            settings.append(
                nn.drop_duplicates(["group", "model"])[
                    [
                        "dataset",
                        "budget_s",
                        "group",
                        "model",
                        "variant",
                        "updates",
                        "config",
                        "numerical_variant",
                    ]
                ]
            )
            for record in json.loads(paths[0].with_name("selection.json").read_text()):
                if record["model"].startswith("PI-"):
                    pi_choices.append(
                        dict(
                            dataset=a.dataset,
                            budget_s=b,
                            heldout_group=record["group"],
                            model=record["model"],
                            **json.loads(record["config"]),
                        )
                    )
        if a.axis == "rate":
            candidate_path = paths[0].with_name("candidate_metrics.csv")
            rate_candidates.append(pd.read_csv(candidate_path, dtype={"group": str}))
            paths = paths[1:]
        for path in paths:
            if not path.exists():
                raise SystemExit(f"Missing completed result: {path}")
            frames.append(pd.read_csv(path, dtype={"group": str}))
    if rate_candidates:
        reference_origin = json.loads(a.config.read_text())["duration"]["origin_s"]
        name = f"{a.dataset}_B{grid[0][1]}_Hz10_end{reference_origin}"
        source = pd.read_csv(
            OUTPUTS / "physics/grid" / name / "candidate_metrics.csv",
            dtype={"group": str},
        )
        frames.append(
            select_rate_physics(pd.concat(rate_candidates, ignore_index=True), source)
        )
    d = pd.concat(frames, ignore_index=True)
    d["model"] = d.model.replace({"DeepKoopman-projected": "Deep Koopman"})
    d = d[d["mode"].str.startswith("ACC")].copy()
    expected = {
        "CTH",
        "IDM",
        "PI-Koopman-CTH",
        "PI-Koopman-IDM",
        "DNN",
        "LSTM",
        "Deep Koopman",
    }
    keys = ["budget_s", "rate_hz", "origin_s", "trial_uid", "model"]
    assert not d.duplicated(keys).any()
    for _, cell in d.groupby(["budget_s", "rate_hz", "origin_s"]):
        assert set(cell.model) == expected
        assert (
            cell.groupby("model")
            .trial_uid.nunique()
            .eq(24 if a.dataset == "local" else 236)
            .all()
        )
    metrics = ["speed_rmse_mps", "gap_rmse_m"]
    # Equal origin weight within encounter, then equal encounter weight.
    enc = d.groupby(["budget_s", "rate_hz", "trial_uid", "model"], as_index=False)[
        metrics
    ].mean()
    summary = enc.groupby(["budget_s", "rate_hz", "model"], as_index=False)[
        metrics
    ].mean()
    out = OUTPUTS / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"{a.dataset}_{a.axis}.csv", index=False)
    if settings:
        pd.concat(settings).to_csv(
            out / f"{a.dataset}_neural_settings.csv", index=False
        )
        pd.DataFrame(pi_choices).to_csv(
            out / f"{a.dataset}_pi_settings.csv", index=False
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
