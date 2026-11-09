"""Show the available OpenACC cases after downloading the trajectories."""
from src.data.openacc import openacc

if __name__ == "__main__":
    trials, records = openacc()
    if len(trials) != 284:
        raise RuntimeError(f"Expected 284 episodes, found {len(trials)}")
    print(records.groupby(["campaign", "mode"]).size().to_string())
