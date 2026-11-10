"""Dataset paths in a repository checkout or the paper folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RELEASE_DIR = ROOT / "data/CATS-CF"
OPENACC_RAW = ROOT / "data/OpenACC"

if not (ROOT / "data").exists() and (ROOT.parent / "CATS-CF/public_release").is_dir():
    OUTPUTS = ROOT.parent / "experiments/work"
    RELEASE_DIR = ROOT.parent / "CATS-CF/public_release"
    OPENACC_RAW = ROOT.parent / "OpenACC"
