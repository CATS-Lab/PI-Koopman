"""Export 42 synchronized 30-second car-following CSV tables."""
import argparse
from pathlib import Path

from src.data.release import export_recordings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root of the paired vehicle log folders")
    parser.add_argument("--output", type=Path, default=Path("outputs/rebuilt_data"))
    args = parser.parse_args()
    print(f"{export_recordings(args.input, args.output)} trials")


if __name__ == "__main__":
    main()
