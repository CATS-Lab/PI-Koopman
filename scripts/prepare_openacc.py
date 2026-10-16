"""Build the main-text OpenACC episode view from downloaded JRC files."""
import sys
from pathlib import Path
from pikoopman.data.openacc import openacc

if __name__ == '__main__':
    trials, manifest = openacc(True)
    if len(trials) != 284:
        raise RuntimeError(f'Expected 284 episodes, found {len(trials)}')
    print(len(trials), manifest.groupby(['campaign', 'mode']).size().to_dict())
