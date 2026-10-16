"""Launch exactly the main-text axes; full neural search can take many hours."""
import argparse,pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--axis',choices=['duration','rate'],required=True)
p.add_argument('--models',choices=['physics','neural','all'],default='all')
p.add_argument('--dataset',choices=['local','openacc','both'],default='local');p.add_argument('--device',default='cpu');a=p.parse_args()
if a.axis=='rate' and a.dataset!='local':p.error('Main-text rate axis uses CATS-CF only.')
cells=[(ds,b,10,20) for ds in (['local','openacc'] if a.dataset=='both' else [a.dataset]) for b in [3,8,15]] if a.axis=='duration' else [('local',8,r,o) for r in [1,5,10,25,125] for o in [15,20,24]]
for ds,b,r,o in cells:
 args=['--dataset',ds,'--budget',str(b),'--rate',str(r),'--origin',str(o)]
 if a.models in ['physics','all']:subprocess.run([sys.executable,str(root/'scripts/run_prediction.py')]+args,check=True,cwd=root)
 if a.models in ['neural','all']:subprocess.run([sys.executable,str(root/'experiments/unified_axes_neural.py')]+args+['--device',a.device],check=True,cwd=root)
