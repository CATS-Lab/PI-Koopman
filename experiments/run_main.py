"""Run a history-length or observation-rate prediction grid."""

import argparse, json, pathlib, subprocess, sys

root = pathlib.Path(__file__).resolve().parents[1]
from src.config import cells, DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["duration", "rate"], required=True)
    p.add_argument("--models", choices=["physics", "neural", "all"], default="all")
    p.add_argument("--dataset", choices=["local", "openacc", "both"], default="local")
    p.add_argument("--device", default="cpu")
    p.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    a = p.parse_args()
    if a.axis == "rate" and a.dataset != "local":
        p.error("Main-text rate axis uses CATS-CF only.")
    grid = cells(a.axis, a.dataset, a.config)
    reference_origin = json.loads(a.config.read_text())["duration"]["origin_s"]
    if a.axis == "rate":
        anchor = (a.dataset, grid[0][1], 10, reference_origin)
        if anchor not in grid:
            grid = [anchor, *grid]
    for ds, b, r, o in grid:
        args = [
            "--dataset",
            ds,
            "--budget",
            str(b),
            "--rate",
            str(r),
            "--origin",
            str(o),
        ]
        if a.models in ["physics", "all"]:
            subprocess.run(
                [sys.executable, "-m", "experiments.run_prediction"] + args,
                check=True,
                cwd=root,
            )
        if a.models in ["neural", "all"]:
            subprocess.run(
                [sys.executable, "-m", "experiments.neural_prediction"]
                + args
                + ["--device", a.device, "--reference-origin", str(reference_origin)]
                + (["--transfer"] if a.axis == "rate" else []),
                check=True,
                cwd=root,
            )


if __name__ == "__main__":
    main()
