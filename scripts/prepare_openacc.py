"""Build the main-text OpenACC episode view from downloaded JRC files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from revision_data import openacc

if __name__ == '__main__':
    trials, manifest = openacc(True)
    if len(trials) != 284:
        raise RuntimeError(f'Expected 284 episodes, found {len(trials)}')
    print(len(trials), manifest.groupby(['campaign', 'mode']).size().to_dict())
