"""Rebuild only the main-text 15/30-s releases into a separate directory."""
import argparse, pathlib, sys
import pandas as pd
root=pathlib.Path(__file__).resolve().parents[1]
from pikoopman.data.release import select_all_runs, write_variant
from pikoopman.data.privacy import sanitize_release
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True,help='Directory containing private synchronized parent tables');p.add_argument('--output',type=pathlib.Path,default=root/'outputs/rebuilt_data');a=p.parse_args()
    if a.output.resolve()==(root/'data/release').resolve():
        raise SystemExit('Use a separate output directory to preserve frozen data.')
    for seconds in (15,30):
        table=pd.read_csv(a.input/f'synchronized_pairs_{seconds}s_125hz.csv.gz')
        selected,manifest=select_all_runs(table,seconds,optimize_subwindow=seconds!=15)
        write_variant(a.output,f'{seconds}s',selected,manifest,seconds)
    sanitize_release(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
