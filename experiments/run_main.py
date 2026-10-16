"""Launch exactly the main-text axes; full neural search can take many hours."""
import argparse,pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1]
from pikoopman.config import cells, DEFAULT_CONFIG
def main():
    p=argparse.ArgumentParser();p.add_argument('--axis',choices=['duration','rate'],required=True)
    p.add_argument('--models',choices=['physics','neural','all'],default='all')
    p.add_argument('--dataset',choices=['local','openacc','both'],default='local');p.add_argument('--device',default='cpu');p.add_argument('--config',type=pathlib.Path,default=DEFAULT_CONFIG);a=p.parse_args()
    if a.axis=='rate' and a.dataset!='local':p.error('Main-text rate axis uses CATS-CF only.')
    grid=cells(a.axis,a.dataset,a.config)
    for ds,b,r,o in grid:
     args=['--dataset',ds,'--budget',str(b),'--rate',str(r),'--origin',str(o)]
     if a.models in ['physics','all']:subprocess.run([sys.executable,'-m','experiments.run_prediction']+args,check=True,cwd=root)
     if a.models in ['neural','all']:subprocess.run([sys.executable,'-m','experiments.neural_prediction']+args+['--device',a.device],check=True,cwd=root)


if __name__ == "__main__":
    main()
