"""Prepare a bounded mixed TRAIN/DEV native panel using existing pilot assets."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.cache/'manifest.json').read_text());names={r['object_code'] for r in m['records']}
    base=json.loads((ROOT/'outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1/manifest.json').read_text())
    selected=[]
    for split in ('train','dev'):
        rows=[r for r in base['objects'] if r['object_code'] in names and r['split']==split]
        selected.extend(rows[:2])
    if len(selected)!=4:raise ValueError('pilot panel needs two represented objects per split')
    if a.output.exists():raise ValueError('fresh panel output')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(base,objects=selected,panel='2TRAIN+2DEV meshes;4 starts each at16env;seed213'),indent=2)+'\n')

if __name__=='__main__':main()
