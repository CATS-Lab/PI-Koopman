"""Fit paired acceleration responses on 10 Hz recorded and simulated histories."""

from src.paths import OUTPUTS

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import replays
from src.data.openacc import crop, corrupt, openacc
from src.data.trials import load_trials
from src.evaluation.metrics import grid, metric
from src.training.prediction import PI
from src.models import physics, koopman as koop

METHODS = ["fd", "integral", "weak", "expweak"]


def physical_prior(x, family):
    coef, audit = physics.fit(x.t, x.z, x.u, family.upper())
    return dict(kind="physical", family=family, coef=coef, audit=audit)


def corrected(x, prior, dictionary, method, cfg):
    return koop.fit(
        x.t,
        x.z,
        x.u,
        method,
        dict(**cfg, prior=prior["family"], prior_model=prior, dictionary=dictionary),
    )


def source_scores(rec):
    ds, obj, tr, _ = rec
    x = crop(tr, 13, 20)
    future = crop(tr, 20, 25)
    rows = []
    for name, (dictionary, family) in PI.items():
        prior = physical_prior(x, family)
        for cfg in grid("expweak", 10, 7):
            m = corrected(x, prior, dictionary, "expweak", cfg)
            pred = physics.predict_field(m, future.t, future.u, future.z[0, :2])
            rows.append(
                (
                    obj,
                    name,
                    json.dumps(cfg, sort_keys=True),
                    metric(pred, future)["score"],
                )
            )
    return rows


def evaluate(task):
    (ds, obj, tr, truth), configs = task
    rows = []
    ref = {}
    for noise, seed in replays.NOISE:
        observed = corrupt(tr, noise, seed)
        for end in replays.ORIGINS:
            x = crop(observed, end - 7, end)
            probes = crop(tr, end - 7, end).z[10:]
            for name, (dictionary, family) in PI.items():
                prior = physical_prior(x, family)
                variants = [(family.upper(), prior)]
                for method in METHODS:
                    variants.append(
                        (
                            family.upper() + " " + method,
                            corrected(x, prior, dictionary, method, configs[name]),
                        )
                    )
                for model, m in variants:
                    a = koop.acceleration(m, probes)
                    if not np.isfinite(a).all():
                        raise ArithmeticError((ds, obj, tr.uid, model))
                    if noise == "clean":
                        ref[end, model] = a
                    delta = a - ref[end, model]
                    exact = (
                        np.nan
                        if truth is None
                        else np.sqrt(
                            np.mean(
                                (a - truth[(end - 7) * 10 + 10 : end * 10 + 1]) ** 2
                            )
                        )
                    )
                    rows.append(
                        (
                            ds,
                            obj,
                            tr.uid,
                            end,
                            noise,
                            seed,
                            model,
                            np.sqrt(np.mean(delta**2)),
                            delta.mean(),
                            exact,
                        )
                    )
    return rows


def select_simulated(records, pool):
    pool_records = [r for r in records if r[1] != "recorded"]
    scores = pd.DataFrame(
        [
            row
            for block in pool.imap_unordered(
                source_scores, pool_records, chunksize=4
            )
            for row in block
        ],
        columns=["object", "model", "config", "score"],
    )
    choices = {}
    for obj in scores.object.unique():
        by = scores[scores.object != obj].groupby(["model", "config"]).score.mean()
        for model in PI:
            choices[obj, model] = json.loads(by.loc[model].idxmin())
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        nargs="+",
        default=[
            (OUTPUTS / "summaries/local_pi_settings.csv"),
            (OUTPUTS / "summaries/openacc_pi_settings.csv"),
        ],
    )
    parser.add_argument("--output", type=Path, default=(OUTPUTS / "noise"))
    parser.add_argument("--simulation-profiles", type=Path, help="Previously selected simulation configurations")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    real = pd.concat(
        [pd.read_csv(f, dtype={"heldout_group": str}) for f in args.selection]
    ).query("budget_s==7")
    if "selection" in real:
        real = real[real.selection == "end20"]
    profiles = {
        (r.dataset, r.heldout_group, r.model): dict(T=r.T, ridge=r.ridge, kappa=r.kappa)
        for r in real.itertuples()
    }
    records = list(replays.records("local", load_trials(30, 125)[0])) + list(
        replays.records("openacc", openacc()[0])
    )
    with mp.Pool(args.workers) as pool:
        if args.simulation_profiles is None:
            choices = select_simulated(records, pool)
        else:
            profiles_used = pd.read_csv(args.simulation_profiles)
            choices = {
                (r.object, r.model): dict(T=r.T, kappa=r.kappa, ridge=r.ridge)
                for r in profiles_used.itertuples()
            }
        pd.DataFrame(
            [
                dict(object=obj, model=model, **cfg)
                for (obj, model), cfg in choices.items()
            ]
        ).to_csv(args.output / "synthetic_pi_choices.csv", index=False)

        def tasks():
            for rec in records:
                ds, obj, tr, _ = rec
                group = str(tr.run) if ds == "local" else tr.case
                yield (
                    rec,
                    {
                        m: (
                            profiles[ds, group, m]
                            if obj == "recorded"
                            else choices[obj, m]
                        )
                        for m in PI
                    },
                )

        tmp = args.output / "response_physics.partial"
        with tmp.open("w") as fp:
            for i, block in enumerate(
                pool.imap_unordered(evaluate, tasks(), chunksize=1)
            ):
                pd.DataFrame(
                    block,
                    columns=[
                        "dataset",
                        "object",
                        "uid",
                        "origin",
                        "noise",
                        "noise_seed",
                        "model",
                        "change",
                        "signed_change",
                        "truth_rmse",
                    ],
                ).to_csv(fp, index=False, header=i == 0)
                if i % 100 == 0:
                    print(i + 1, "/", len(records), flush=True)
        tmp.replace(args.output / "response_physics.csv")


if __name__ == "__main__":
    main()
