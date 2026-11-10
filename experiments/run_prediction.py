"""Fit the physical and PI candidates for one prediction condition."""

from src.paths import OUTPUTS

import argparse, pathlib

root = pathlib.Path(__file__).resolve().parents[1]
from src.data.trials import load_trials
from src.data.openacc import openacc
import src.training.prediction as p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["local", "openacc"], default="local")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--rate", type=int, choices=[1, 5, 10, 25, 125], default=10)
    ap.add_argument("--origin", type=int, default=20)
    a = ap.parse_args()
    if a.dataset == "openacc" and a.rate != 10:
        ap.error("OpenACC is released at 10 Hz.")
    if not 0 < a.budget <= a.origin:
        ap.error("History must be positive and cannot extend before the record starts.")
    p.OUT = OUTPUTS / "physics"
    p.OUT.mkdir(parents=True, exist_ok=True)
    trials = load_trials(30, 125)[0] if a.dataset == "local" else openacc()[0]
    if not trials:
        raise SystemExit("Missing OpenACC: run python -m scripts.download_openacc.")
    p.run_cell(a.dataset, a.budget, a.rate, a.origin, trials)


if __name__ == "__main__":
    main()
