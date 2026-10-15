"""Bounded full UniDexGrasp++ motion scan; no GPU or automatic data rewriting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
TASK=Path(__file__).resolve().parents[2]
REPO=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.motion_quality import motion_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists() or REPO/'outputs'/TASK.name not in out.parents or not 1<=a.workers<=4:
        raise ValueError('fresh bounded Task output required')
    out.mkdir(parents=True)
    meta=json.loads((root/'manifest.json').read_text())
    rows=[(i,r) for i,r in enumerate(meta['sequences']) if r['source_id']==4]
    started=time.monotonic();records=[]
    def read(item):
        i,r=item;path=root/r['path'];stamp=path.stat()
        if (stamp.st_size,stamp.st_mtime_ns)!=(r['payload_size_bytes'],r['payload_mtime_ns']):
            raise ValueError('frozen payload changed')
        with np.load(path,allow_pickle=False) as z:f=motion_features(z['time'],z['object'],z['hand'])
        return dict(row=i,split=r['split'],sequence=r['source_sequence'],source_manifest=r['source_manifest'],
            official_success=r.get('official_success'),maxima={k:float(v.max()) for k,v in f.items()},
            peaks={k:int(v.argmax()) for k,v in f.items()},
            # Save all framewise diagnostics so view policies need not rescan source data.
            features={k:v.tolist() for k,v in f.items()})
    progress=dict(status='RUNNING',run_id=out.name,input_manifest=str(root/'manifest.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        device='CPU array diagnostics only; no neural computation',workers=a.workers,max_seconds=1200,
        artifact_budget_bytes=300_000_000)
    def checkpoint():
        progress.update(processed=len(records),total=len(rows),elapsed_s=time.monotonic()-started)
        (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        if progress['elapsed_s']>1200 or shutil.disk_usage(out).free<20*2**30 or (out/'STOP').exists():
            raise RuntimeError('motion scan deadline/disk/STOP')
        if (out/'records.jsonl').exists() and (out/'records.jsonl').stat().st_size>300_000_000:
            raise RuntimeError('motion scan artifact budget')
    try:
        checkpoint()
        with (out/'records.jsonl').open('x') as handle,ThreadPoolExecutor(max_workers=a.workers) as pool:
            for first in range(0,len(rows),512):
                for record in pool.map(read,rows[first:first+512]):
                    handle.write(json.dumps(record,separators=(',',':'))+'\n')
                    records.append({k:v for k,v in record.items() if k!='features'})
                handle.flush();checkpoint()
                if len(records)%4096==0:print(json.dumps(progress),flush=True)
        train=[r for r in records if r['split']=='train']
        keys=list(train[0]['maxima']);quantiles=[0,.5,.9,.95,.99,.999,1]
        distribution={k:dict(zip(map(str,quantiles),np.quantile([r['maxima'][k] for r in train],quantiles).tolist())) for k in keys}
        summary=dict(status='COMPLETED',sequences=len(records),train_sequences=len(train),
            train_quantiles=distribution,
            train_candidate_counts={k:{str(t):sum(r['maxima'][k]>t for r in train) for t in thresholds}
                for k,thresholds in dict(speed_mps=[3,5,10,15,20],acceleration_mps2=[50,100,300,1000],
                    step_m=[.1,.25,.5],angular_speed_radps=[20,50,100]).items()},
            top_speed=sorted(train,key=lambda r:r['maxima']['speed_mps'],reverse=True)[:20])
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        progress['status']='COMPLETED';checkpoint()
        print(json.dumps({k:v for k,v in summary.items() if k!='top_speed'}),flush=True)
    except BaseException as error:
        progress.update(status='FAILED',reason=repr(error));(out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');raise


if __name__=='__main__':main()
