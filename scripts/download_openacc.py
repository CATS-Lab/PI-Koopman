"""Download official JRC campaigns without changing their contents."""
from pathlib import Path
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
import time

ROOT=Path(__file__).resolve().parents[1]
BASE='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/'
OUT=ROOT/'data/external/openacc/raw'
FILES={'Casale':['Casale_notes.txt','Veh_specifications.csv','copyright.txt']+[f'part{i}.csv' for i in range(3,12)],
       'AstaZero':['AstaZero_notes.txt','AstaZero_veh_specifications.csv','copyright.txt']+[f'ASta_040719_platoon{i}.csv' for i in range(3,11)]+[f'ASta_050719_platoon{i}.csv' for i in range(1,3)]}

def fetch(item):
    campaign,name=item;path=OUT/campaign/name;path.parent.mkdir(parents=True,exist_ok=True)
    url=BASE+campaign+'/'+name
    if not path.exists():
        for attempt in range(3):
            try:
                with urlopen(url,timeout=90) as response: body=response.read()
                path.write_bytes(body);break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    body=path.read_bytes()
    return dict(campaign=campaign,file=name,url=url,bytes=len(body),downloaded=True)

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fetch,[(c,n) for c,files in FILES.items() for n in files]))
    print('Downloaded',len(rows),'files;',sum(r['bytes'] for r in rows),'bytes')
