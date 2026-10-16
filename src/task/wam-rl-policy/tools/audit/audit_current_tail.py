"""Measure whether forecast-window admission leaves current BC's holding tail weak."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2]))
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2);a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Assigned GPU occupied')
    out.mkdir(parents=True);os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Tail audit budget')));signal.alarm(600)
    import numpy as np
    import torch
    from wam_rl_policy.bc_mlp import CurrentMLP,SCHEMA
    torch.set_num_threads(2);started=time.monotonic()
    packet=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert packet['schema']==SCHEMA
    model=CurrentMLP().cuda().eval();model.load_state_dict(packet['model'],strict=True)
    meta=json.loads((a.cache/'manifest.json').read_text());groups=defaultdict(list)
    for record in meta['records']:
        if record['split']=='dev':groups[record['object_code']].append(record)
    rng=np.random.default_rng(244);samples=[];labels=[];stages=[];source=[]
    for code,records in sorted(groups.items()):
        record=records[int(rng.integers(len(records)))];limit=min(199,int(record['unsafe_onset'])-1)
        with np.load(a.cache/record['path']) as z:
            ids=np.arange(0,limit,4);samples.append(z['observation'][ids]);labels.append(z['action'][ids])
        stages.extend('startup' if t<8 else 'admitted_period' if t<152 else 'late_unseen_period' for t in ids)
        source.append(dict(object_code=code,sequence=record['path'],safe_limit=limit))
    x=torch.as_tensor(np.concatenate(samples),device='cuda');y=torch.as_tensor(np.concatenate(labels),device='cuda');errors=[]
    with torch.no_grad():
        for off in range(0,len(x),2048):errors.append((model(x[off:off+2048])-y[off:off+2048]).abs().cpu().numpy())
    error=np.concatenate(errors);stages=np.asarray(stages);metrics={}
    for stage in sorted(set(stages)):
        selected=error[stages==stage];metrics[stage]=dict(windows=len(selected),command_mae=float(selected.mean()),root_mae=float(selected[:,:6].mean()),finger_mae=float(selected[:,6:].mean()))
    manifest=dict(status='COMPLETED',schema='ref2dex.shadow-current-tail-audit.v1',model_compute='GPU',gpu=a.gpu,
        checkpoint=str(a.checkpoint.resolve()),dev_objects=len(groups),selection='one DEV sequence per object; seed244; every fourth current safe tick',
        metric='current canonical command MAE, not closed-loop performance',metrics=metrics,sources=source,elapsed_s=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(metrics=metrics,elapsed_s=manifest['elapsed_s'])))


if __name__=='__main__':main()
