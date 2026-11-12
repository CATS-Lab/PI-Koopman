"""Run a paper experiment and collect its tables from the repository root."""

import argparse
import subprocess
import sys

from src.paths import OUTPUTS, ROOT


def run(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["duration", "rate", "recognition", "noise", "timing"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.experiment in ("noise", "timing"):
        profiles = OUTPUTS / "summaries"
        kinds = ["pi", "neural"] if args.experiment == "noise" else ["pi"]
        if any(not (profiles / f"{ds}_{kind}_settings.csv").is_file()
               for ds in ["local", "openacc"] for kind in kinds):
            parser.error("First run: python -m scripts.reproduce duration")
    if args.experiment in ("duration", "rate"):
        datasets = ["local", "openacc"] if args.experiment == "duration" else ["local"]
        run("experiments.run_main", "--axis", args.experiment, "--dataset",
            "both" if len(datasets) == 2 else "local", "--device", args.device)
        for dataset in datasets:
            run("experiments.summarize_prediction", "--axis", args.experiment, "--dataset", dataset)
    elif args.experiment == "recognition":
        run("experiments.recognition")
    elif args.experiment == "noise":
        observations = str(OUTPUTS / "noise/observations")
        run("experiments.noise_response", "--simulation-profiles", str(ROOT / "configs/simulation_pi.csv"))
        run("experiments.noise_neural", "--prepare", "--input", observations)
        run("experiments.noise_neural", "--input", observations, "--device", args.device)
        run("experiments.summarize_noise")
        run("experiments.plot_noise")
    else:
        for dataset in ["local", "openacc"]:
            for budget in ["5", "7", "11"]:
                run("experiments.timing", dataset, budget, "--device", args.device)
        import pandas as pd
        file = "timing_cuda.csv" if args.device == "cuda" else "timing.csv"
        times = pd.read_csv(OUTPUTS / file).groupby("model")[["fit_s", "forecast_s"]].mean()
        print((times * 1000).set_axis(["fit_ms", "forecast_ms"], axis=1).round(2))


if __name__ == "__main__":
    main()
