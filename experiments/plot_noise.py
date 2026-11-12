"""Draw the seven-model response comparison and the three PI estimation methods."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import OUTPUTS


def plot_noise(summary, out):
    d = pd.read_csv(summary)
    out.mkdir(exist_ok=True, parents=True)
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.8))
    short = [
        "CTH",
        "IDM",
        "DNN",
        "LSTM",
        "DeepKoopman-projected",
        "CTH expweak",
        "IDM expweak",
    ]
    for j, ds in enumerate(["local", "openacc"]):
        for i, cohort in enumerate(["recorded", "simulated"]):
            ax = axs[i, j]
            g = d[(d.dataset == ds) & (d.cohort == cohort)]
            for k, n in enumerate(["white", "correlated"]):
                a = g[g.noise == n].set_index("model").reindex(short)
                ax.scatter(
                    a.change if i == 0 else a.truth_rmse,
                    np.arange(7) + (k - 0.5) * 0.2,
                    marker=["o", "s"][k],
                    color=["#09517b", "#b45424"][k],
                    label=["Independent", "Time-correlated"][k],
                )
            ax.set_yticks(
                np.arange(7),
                ["CTH", "IDM", "DNN", "LSTM", "Deep Koopman", "PI-CTH (ours)", "PI-IDM (ours)"] if j == 0 else [],
            )
            ax.invert_yaxis()
            ax.set_xlim(left=0)
            ax.grid(axis="x", alpha=0.18)
            ax.set_title(("CATS-CF" if ds == "local" else "OpenACC") + " · " + cohort)
            ax.set_xlabel(
                ("Response change" if i == 0 else "RMSE against truth") + " (m/s²)"
            )
    for row in axs:
        limit = max(ax.get_xlim()[1] for ax in row)
        for ax in row:
            ax.set_xlim(0, limit)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(out / "claim_rate_consistency.pdf", bbox_inches="tight")
    plt.close(fig)
    methods = ["fd", "integral", "expweak"]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for j, family in enumerate(["CTH", "IDM"]):
        for i, kind in enumerate(["white", "correlated"]):
            ax = axs[i, j]
            ax.axhline(0, color="#666666", lw=1, ls="--")
            for ds, offset, marker, color in [
                ("local", -0.09, "o", "#09517b"),
                ("openacc", 0.09, "s", "#b45424"),
            ]:
                g = d[
                    (d.dataset == ds) & (d.noise == kind) & (d.cohort == "simulated")
                ].set_index("model")
                reference = g.loc[family + " weak", "truth_rmse"]
                differences = [g.loc[f"{family} {m}", "truth_rmse"] - reference
                               for m in methods]
                ax.scatter(np.arange(len(methods)) + offset, differences,
                           s=[30, 30, 54],
                           marker=marker, color=color,
                           label="CATS-CF" if ds == "local" else "OpenACC")
            ax.set_xticks(range(len(methods)), ["FD", "Integral", "Exp. weak (ours)"])
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.set_title(family + " · " + ["independent", "time-correlated"][i])
            ax.set_ylabel("RMSE minus ordinary weak (m/s²)")
            ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center",
               ncol=2, frameon=False)
    fig.savefig(out / "synthetic_field_error.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=OUTPUTS / "noise/summary.csv")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "noise")
    args = parser.parse_args()
    plot_noise(args.summary, args.output)
