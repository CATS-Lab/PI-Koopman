"""Dataset paths in a repository checkout or the paper folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "data/release"
OPENACC_RAW = ROOT / "data/external/openacc/raw"

if not (ROOT / "data").exists() and (ROOT.parent / "CATS-CF/public_release").is_dir():
    RELEASE_DIR = ROOT.parent / "CATS-CF/public_release"
    OPENACC_RAW = ROOT.parent / "OpenACC"
