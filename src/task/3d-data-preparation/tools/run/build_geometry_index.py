"""Refresh a geometry-only corpus index while immutable shards accumulate."""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[5]
TASK=Path(__file__).resolve().parents[2]


def write(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def build(root):
    base=REPO/'outputs/3d-data-preparation'
    sources=[base/name for name in ['unidexgrasp-index-20261014-r1','hot3d-posemesh-full-20261014-r1',
        'taco-posemesh-full-20261014-r1','hocap-posemesh-reuse-20261014-r1','dexycb-posemesh-reuse-20261014-r1',
        'h2o-posemesh-full-20261014-r1']]
    queue=base/'unidexgrasp-expansion-20261014-r1'
    for batch in sorted(queue.glob('batch*')):
        run=batch/'replay';index=batch/'posemesh-index'
        if (run/'audit-summary.json').exists() and (run/'trajectory-manifest.json').exists():
            if not (index/'manifest.json').exists():
                subprocess.run([sys.executable,str(TASK/'tools/run/index_unidexgrasp.py'),'--run',str(run),'--output',str(index)],
                    cwd=REPO,check=True,stdout=subprocess.DEVNULL)
            sources.append(index)
    combined=dict(schema='ref2dex.pose-mesh.v1',units='m',fps=30,training_allowed=False,objects=[],sequences=[],
        storage='immutable source/CAD references; no duplicated temporal windows',
        limitations=['Live engineering index; freeze before any scientific training run.',
            'TACO/HO-Cap/DexYCB source clocks nominal or unverified; not native30Hz admission.',
            'HO-Cap retained entirely in DEV; target semantics/profiles require source admission.'])
    seen={};counts=collections.Counter();groups={'train':set(),'dev':set()};used=[]
    for source in sources:
        file=source/'manifest.json'
        if not file.exists():continue
        meta=json.loads(file.read_text())
        if meta.get('schema')!=combined['schema'] or meta.get('training_allowed') is not False:
            raise ValueError('explicit provisional source required')
        for row in meta['objects']:
            identity=row['key'];path=(source/row['path']).resolve()
            if identity in seen:
                if seen[identity]!=path:raise ValueError('same geometry identity has different source files')
                continue
            combined['objects'].append({**row,'path':os.path.relpath(path,root)});seen[identity]=path
        for row in meta['sequences']:
            combined['sequences'].append({**row,'path':os.path.relpath((source/row['path']).resolve(),root)})
            counts[(int(row['source_id']),row['split'])]+=1;groups[row['split']].add(row['group_id'])
        used.append(str(source))
    if groups['train']&groups['dev']:raise ValueError('source group leaked between TRAIN and DEV')
    combined['source_manifests']=used;combined['checked_at']=time.time()
    write(root/'manifest.json',combined)
    report=dict(status='LIVE_ENGINEERING_INDEX',pid=os.getpid(),sequences=len(combined['sequences']),geometries=len(seen),
        source_counts=[dict(source_id=s,split=sp,sequences=n) for (s,sp),n in sorted(counts.items())],checked_at=time.time(),training_allowed=False)
    write(root/'progress.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--seconds',type=int,default=86400);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    while True:
        print(json.dumps(build(root)),flush=True)
        if not a.watch or (root/'STOP').exists() or time.monotonic()-started>a.seconds:break
        time.sleep(60)


if __name__=='__main__':main()
