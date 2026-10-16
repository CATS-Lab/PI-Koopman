"""Aggregate complete seven-model main-text prediction axes without plots."""
import argparse,json,pathlib,sys
import pandas as pd
root=pathlib.Path(__file__).resolve().parents[1]
from src.config import cells, DEFAULT_CONFIG
def main():
    p=argparse.ArgumentParser();p.add_argument('--axis',choices=['duration','rate'],required=True)
    p.add_argument('--dataset',choices=['local','openacc'],default='local');p.add_argument('--config',type=pathlib.Path,default=DEFAULT_CONFIG);a=p.parse_args()
    if a.axis=='rate' and a.dataset!='local':p.error('Rate axis is local only.')
    grid=cells(a.axis,a.dataset,a.config)
    frames=[]
    for _,b,r,o in grid:
     name=f'{a.dataset}_B{b}_Hz{r}_end{o}'
     paths=[root/'outputs/physics/grid'/name/'selected_metrics.csv',root/'outputs/neural/grid'/name/'selected_encounter_metrics.csv']
     for path in paths:
      if not path.exists():raise SystemExit(f'Missing completed result: {path.relative_to(root)}')
      frames.append(pd.read_csv(path,dtype={'group':str}))
    d=pd.concat(frames,ignore_index=True);d['model']=d.model.replace({'DeepKoopman-projected':'Deep Koopman'})
    d=d[d['mode'].str.startswith('ACC')].copy()
    expected={'CTH','IDM','PI-Koopman-CTH','PI-Koopman-IDM','DNN','LSTM','Deep Koopman'}
    keys=['budget_s','rate_hz','origin_s','trial_uid','model']
    assert not d.duplicated(keys).any()
    for _,cell in d.groupby(['budget_s','rate_hz','origin_s']):
     assert set(cell.model)==expected
     assert cell.groupby('model').trial_uid.nunique().eq(24 if a.dataset=='local' else 236).all()
    metrics=['speed_rmse_mps','gap_rmse_m']
    # Equal origin weight within encounter, then equal encounter weight.
    enc=d.groupby(['budget_s','rate_hz','trial_uid','model'],as_index=False)[metrics].mean()
    summary=enc.groupby(['budget_s','rate_hz','model'],as_index=False)[metrics].mean()
    out=root/'outputs/summaries';out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/f'{a.dataset}_{a.axis}.csv',index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
