"""Exhaustive admitted TRAIN label replay using the frozen official state actor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6);a=p.parse_args();root=Path(__file__).resolve().parents[5]
    if a.output.exists() or root/'outputs/shadow-finetune' not in a.output.resolve().parents:
        raise ValueError('fresh Task output required')
    def pids(gpu):
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        return {int(x) for x in s.split() if x.isdigit()}
    if pids(a.gpu):raise RuntimeError('selected GPU occupied')
    if a.gpu==6 and pids(7):raise RuntimeError('temporary test requires GPU7 reserve')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    widths=[300,1024,1024,512,512,24];layers=[]
    for i,(lo,hi) in enumerate(zip(widths,widths[1:])):
        layers.append(torch.nn.Linear(lo,hi))
        if i<4:layers.append(torch.nn.ELU())
    actor=torch.nn.Sequential(*layers).cuda().eval()
    weights=root/'outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1/plus/dexgrasp/example_model/state_based_model.pt'
    state=torch.load(weights,map_location='cuda',weights_only=True)
    actor.load_state_dict({k[len('actor.'):]:v for k,v in state.items() if k.startswith('actor.')},strict=True)
    actor.requires_grad_(False)
    meta=json.loads((a.cache/'manifest.json').read_text())
    if meta.get('schema')!='ref2dex.shadow-native-policy.v1' or meta.get('status')!='COMPLETED':
        raise ValueError('completed native cache required')
    rows=[row for row in meta['records'] if row['split']=='train']
    if not rows:raise ValueError('nonempty TRAIN cache required')
    a.output.mkdir(parents=True);started=time.monotonic();total=bad_values=bad_windows=0
    error_sum=maximum=0.;worst=None;obs=[];targets=[];origins=[]
    def flush():
        nonlocal total,bad_values,bad_windows,error_sum,maximum,worst
        if not obs:return
        x=torch.as_tensor(np.concatenate(obs),device='cuda')
        y=torch.as_tensor(np.concatenate(targets),device='cuda')
        with torch.inference_mode():pred=actor(x.clamp(-5,5)).clamp(-1,1)
        if not torch.isfinite(pred).all():raise ValueError('nonfinite teacher replay')
        error=(pred-y).abs();value=float(error.max());error_sum+=float(error.sum(dtype=torch.float64))
        bad_values+=int((error>1e-4).sum());bad_windows+=int((error>1e-4).any(1).sum());total+=len(x)
        if value>maximum:
            index=int(error.argmax());example,dimension=divmod(index,24);rowid,tick=origins[example]
            maximum=value;worst=dict(record=rowid,tick=tick,dimension=dimension,
                path=rows[rowid]['path'],raw_episode=rows[rowid]['raw_episode'],env=rows[rowid]['env'],
                expected=float(y[example,dimension]),replayed=float(pred[example,dimension]))
        obs.clear();targets.clear();origins.clear()
        if torch.cuda.max_memory_allocated()>2*2**30:raise RuntimeError('2GiB model allocation budget')
    for i,row in enumerate(rows):
        if time.monotonic()-started>600:raise TimeoutError('ten-minute teacher-label audit')
        ticks=np.asarray(row['ticks'],dtype=np.int64)
        with np.load(a.cache/row['path'],allow_pickle=False) as z:
            obs.append(z['observation'][ticks].astype(np.float32));targets.append(z['action'][ticks].astype(np.float32))
        origins.extend((i,int(t)) for t in ticks)
        if len(origins)>=4096:flush()
        if (i+1)%2000==0:
            if pids(a.gpu)-{os.getpid()}:raise RuntimeError('foreign selected-GPU process')
            if a.gpu==6 and pids(7):raise RuntimeError('GPU7 reserve occupied')
            print(json.dumps(dict(sequences=i+1,windows=total,max_abs=maximum,elapsed_s=time.monotonic()-started)),flush=True)
    flush();expected=sum(len(r['ticks']) for r in rows)
    if total!=expected:raise ValueError('label audit coverage mismatch')
    result=dict(status='PASS' if not bad_values else 'FAIL',split='train',sequences=len(rows),windows=total,
        max_abs_error=maximum,mean_abs_error=error_sum/max(total*24,1),tolerance=1e-4,
        mismatched_values=bad_values,mismatched_windows=bad_windows,worst=worst,
        teacher_weights=str(weights),actor='300-1024-1024-512-512-24 ELU; strict official weights; clip input5/action1',
        cache=str(a.cache.resolve()),gpu=a.gpu,elapsed_s=time.monotonic()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        scope='all admitted TRAIN first commands; no BC training, future-action replay or closed-loop success certification')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
