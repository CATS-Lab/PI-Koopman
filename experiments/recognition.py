"""ACC/HDV recognition from the first 15 s of the 30 s CATS-CF records."""

from pathlib import Path
import csv
import json
import traceback
import argparse
import numpy as np
import pandas as pd

import src.evaluation.recognition as helpers
import src.evaluation.statistics as stats
from src.models.koopman import acceleration
import src.models.koopman as koopman

OUT = helpers.OUT
N_BOOT = 5000
SEED = 20261015


def save_json(path, obj):
    path.write_text(
        json.dumps(helpers.to_jsonable(obj), indent=2, allow_nan=False) + "\n"
    )


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordinary_residuals(t, z, model):
    """One-second velocity-integral residuals, starting every 0.4 s.

    PI-CTH uses 5 intervals at 10 Hz and 13 at 25 Hz. PI-IDM uses 2 s.
    The evaluation width and stride are exact at both rates.
    """
    rate = round(1 / np.median(np.diff(t)))
    if rate not in helpers.RATES_HZ or not np.allclose(np.diff(t), 1 / rate, atol=1e-8):
        raise ValueError("Expected a uniform 10 or 25 Hz observation grid")
    width = round(helpers.RESIDUAL_SUPPORT_S * rate)
    stride = round(helpers.RESIDUAL_STRIDE_S * rate)
    field = acceleration(model, z)
    if not np.isfinite(field).all():
        raise ValueError("Nonfinite observed-state acceleration")
    rows = []
    for j in range(0, len(t) - width, stride):
        end = j + width
        duration = t[end] - t[j]
        if not np.isclose(duration, 1.0, atol=1e-8):
            raise ValueError("Expected exactly 1 s support")
        integral = np.trapezoid(field[j : end + 1], t[j : end + 1])
        residual = (z[end, 1] - z[j, 1] - integral) / duration
        if not np.isfinite(residual):
            raise ValueError("Nonfinite residual; no encounter may be dropped")
        rows.append(
            dict(
                window_index=len(rows),
                start_index=j,
                end_index=end,
                start_s=float(t[j]),
                end_s=float(t[end]),
                residual=float(residual),
            )
        )
    return rows


def choose_threshold(y, values):
    y, values = np.asarray(y), np.asarray(values, float)
    if not np.isfinite(values).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold requires finite scores and both classes")
    unique = np.unique(values)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        unique[:-1] + np.diff(unique) / 2,
        np.nextafter(unique[-1], np.inf),
    ]
    ba = np.array(
        [
            0.5 * (np.mean((values[y == 0] > c)) + np.mean((values[y == 1] <= c)))
            for c in candidates
        ]
    )
    tied = candidates[np.isclose(ba, ba.max(), rtol=0, atol=1e-14)]
    return float(tied[(len(tied) - 1) // 2])


def synthetic_checks():
    for rate in helpers.RATES_HZ:
        t = np.arange(15 * rate) / rate
        z = np.column_stack((np.full(len(t), 20.0), 10 + 1.7 * t, np.zeros(len(t))))
        model = dict(
            kind="physical", family="cth", coef=[1.0, 0.0, 18.3, 0.0], prior=None
        )
        residual = ordinary_residuals(t, z, model)
        assert len(residual) == 35
        np.testing.assert_allclose(
            [r["start_s"] for r in residual], np.arange(35) * 0.4, atol=1e-12
        )
        assert np.max(np.abs([r["residual"] for r in residual])) < 1e-12
        cth_support = {10: 0.5, 25: 0.52}[rate]
        A, y = koopman.weak_rows(
            t, np.ones((len(t), 1)), z[:, 1], "expweak", cth_support, 1.0
        )
        assert len(y) == {10: 73, 25: 61}[rate]
        np.testing.assert_allclose(A[:, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(y, 1.7, atol=1e-12)
    threshold = choose_threshold([1, 1, 0, 0], [1.0, 2.0, 3.0, 4.0])
    assert threshold == 2.5
    # Disconnected tied optima: their arithmetic median need not be optimal.
    y = np.array([1, 0, 0, 1, 1, 0])
    values = np.arange(1.0, 7.0)
    threshold = choose_threshold(y, values)
    ba = lambda c: 0.5 * (np.mean(values[y == 0] > c) + np.mean(values[y == 1] <= c))
    candidates = np.r_[0.0, np.arange(1.5, 6.0, 1.0), 7.0]
    assert np.isclose(ba(threshold), max(ba(c) for c in candidates))
    assert threshold == 1.5


def analyze(rows, score_key="residual_variance"):
    methods = list(dict.fromkeys(row["model"] for row in rows))
    reference = [r for r in rows if r["model"] == methods[0]]
    y = np.array([r["is_acc"] for r in reference])
    cases = np.array([r["case_id"] for r in reference])
    folds = np.array([r["run_index"] for r in reference])
    draws = stats.cluster_draws(y, cases, n=N_BOOT, seed=SEED)
    summary, oof, boot_values = {}, [], {}
    for method in methods:
        data = [r for r in rows if r["model"] == method]
        assert [r["trial_uid"] for r in data] == [r["trial_uid"] for r in reference]
        values = np.array([r[score_key] for r in data])
        pred = np.full(len(y), -1)
        for fold in np.unique(folds):
            train, test = folds != fold, folds == fold
            threshold = choose_threshold(y[train], values[train])
            pred[test] = values[test] <= threshold
            for i in np.flatnonzero(test):
                oof.append(
                    dict(
                        trial_uid=data[i]["trial_uid"],
                        case_id=cases[i],
                        model=method,
                        held_out_run_index=int(fold),
                        is_acc=int(y[i]),
                        score_name=score_key,
                        score_value=float(values[i]),
                        threshold=threshold,
                        predicted_acc=int(pred[i]),
                    )
                )
        result = stats.scores(y, pred, -values)
        boot_values[method], ci = stats.bootstrap_ba(y, pred, draws)
        acc, hdv = float(np.median(values[y == 1])), float(np.median(values[y == 0]))
        result.update(
            score_name=score_key,
            acc_median_score=acc,
            hdv_median_score=hdv,
            hdv_acc_median_ratio=hdv / acc if acc > 0 else None,
            balanced_accuracy_bootstrap_ci95=ci,
        )
        summary[method] = result
    differences = {}
    for first in methods:
        for second in methods:
            if first.startswith("PI-") and not second.startswith("PI-"):
                delta = boot_values[first] - boot_values[second]
                differences[first + " minus " + second] = dict(
                    balanced_accuracy_difference=summary[first]["balanced_accuracy"]
                    - summary[second]["balanced_accuracy"],
                    paired_bootstrap_ci95=np.quantile(delta, [0.025, 0.975]).tolist(),
                )
    return dict(
        score_name=score_key,
        models=summary,
        paired_comparisons=differences,
        bootstrap=dict(
            replicates=N_BOOT,
            seed=SEED,
            unit="case_id conditions within class, retaining all runs",
            scope="Conditional uncertainty for fixed out-of-fold predictions; thresholds not refitted in bootstrap",
        ),
    ), oof


def run_rate(rate, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, windows = [], []
    current_uid = None
    try:
        trials = helpers.load_trials(rate=rate)
        for index, trial in enumerate(trials):
            metadata = trial["metadata"]
            current_uid = metadata["trial_uid"]
            t, z, u = trial["t"], trial["z"], trial["u"]
            print(
                f"{rate} Hz [{index + 1}/{len(trials)}] fitting {current_uid}",
                flush=True,
            )
            fitted = helpers.fit_pi_models(t, z, u)
            for name, item in fitted.items():
                residual_rows = ordinary_residuals(t, z, item["model"])
                if len(residual_rows) != 35:
                    raise ValueError(
                        f"{current_uid}: expected 35 residual windows, found {len(residual_rows)}"
                    )
                residual = np.array([r["residual"] for r in residual_rows])
                row = dict(
                    metadata,
                    model=name,
                    n_windows=len(residual),
                    residual_mean=float(residual.mean()),
                    residual_variance=float(residual.var(ddof=1)),
                    residual_rmse=float(np.sqrt(np.mean(residual**2))),
                )
                if not np.isfinite(
                    [
                        row[k]
                        for k in ("residual_mean", "residual_variance", "residual_rmse")
                    ]
                ).all():
                    raise ValueError("Nonfinite encounter score")
                rows.append(row)
                windows.extend(dict(metadata, model=name, **r) for r in residual_rows)
            print(
                f"{rate} Hz [{index + 1}/{len(trials)}] completed {current_uid}",
                flush=True,
            )
        summary, oof = analyze(rows)
        save_csv(out / "rows.csv", rows)
        save_csv(out / "residual_windows.csv", windows)
        save_csv(out / "predictions.csv", oof)
        results = []
        for name, metric in summary["models"].items():
            cm = np.asarray(metric["confusion_manual_acc"])
            lo, hi = metric["balanced_accuracy_bootstrap_ci95"]
            results.append(
                dict(
                    rate_hz=rate,
                    model=name,
                    acc_median_variance=metric["acc_median_score"],
                    hdv_median_variance=metric["hdv_median_score"],
                    hdv_acc_variance_ratio=metric["hdv_acc_median_ratio"],
                    acc_correct=int(cm[1, 1]),
                    acc_total=int(cm[1].sum()),
                    hdv_correct=int(cm[0, 0]),
                    hdv_total=int(cm[0].sum()),
                    balanced_accuracy=metric["balanced_accuracy"],
                    auc=metric["auc"],
                    ba_ci_low=lo,
                    ba_ci_high=hi,
                )
            )
        return results
    except Exception as exc:
        save_json(
            out / "failure.json",
            dict(
                trial_uid=current_uid,
                error=str(exc),
                traceback=traceback.format_exc(),
                policy="Stopped; no failed encounter dropped",
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rates",
        type=int,
        nargs="+",
        choices=helpers.RATES_HZ,
        default=list(helpers.RATES_HZ),
    )
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if len(set(args.rates)) != len(args.rates):
        parser.error("Each rate should be listed once")
    synthetic_checks()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for rate in args.rates:
        results.extend(run_rate(rate, args.output / f"{rate}hz"))
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / "summary.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
