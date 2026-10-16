"""Main-text physical/PI cells with the original source-group selection."""
import argparse, json, pathlib, sys
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'experiments'));sys.path.insert(0,str(root/'src'))
from empirical_benchmark import load_trials
from revision_data import openacc
import protocol_revision_sensitivity as p
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['local','openacc'],default='local')
ap.add_argument('--budget',type=int,choices=[3,8,15],default=3)
ap.add_argument('--rate',type=int,choices=[1,5,10,25,125],default=10)
ap.add_argument('--origin',type=int,choices=[15,20,24],default=20);a=ap.parse_args()
if a.dataset=='openacc' and a.rate!=10:ap.error('OpenACC is released at 10 Hz.')
p.OUT=root/'outputs/physics';p.OUT.mkdir(parents=True,exist_ok=True)
protocol=dict(models=p.METHODS,selection='exclude target run or whole OpenACC source file',
 failure_penalties=dict(speed=1000,gap=10000),scope='main-text physical and PI candidates only')
(p.OUT/'protocol_frozen.json').write_text(json.dumps(protocol,indent=2))
trials=load_trials(30,125)[0] if a.dataset=='local' else openacc()[0]
if not trials:raise SystemExit('Missing OpenACC: run scripts/download_openacc.py then src/revision_data.py.')
p.run_cell(a.dataset,a.budget,a.rate,a.origin,trials)
