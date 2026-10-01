#!/usr/bin/env python3
"""Audit force-proxy-trigger geometry from archived actual pre-pulse states."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import PANELS,admission,sha
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge


def summarize(gaps):
    return dict(states=len(gaps),near20mm_fraction=float((gaps<=.020).float().mean()),
                near5mm_fraction=float((gaps<=.005).float().mean()),
                gap_mm_quantiles={str(q):float(torch.quantile(gaps,q)*1000) for q in (0.,.1,.5,.9,1.)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=4)
    args=parser.parse_args()
    if ROOT not in args.output.resolve().parents or args.output.exists():
        raise ValueError('unique owned output required')
    gpu=admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu['uuid']:
        raise ValueError('visible GPU does not match admission')
    torch.set_num_threads(2)
    started=time.monotonic()
    asset=ROOT/'third_party/DExplore/dexplore/data/assets'
    hand=asset/'inspire_hand_new/inspire_hand_right.urdf'
    obj=asset/'mjcf/airplane.urdf'
    inputs={str(p.resolve()):sha(p) for p in (hand,obj,Path(__file__).resolve())}
    args.output.mkdir(parents=True)
    manifest=dict(experiment_id='P-20261001-contact-trigger-geometry',run_status='RUNNING',
                  pid=os.getpid(),command=sys.argv,gpu=gpu,input_sha256=inputs,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds']=time.monotonic()-started
        (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        bridge=DExploreCmv2GeometryBridge(hand_urdf=hand,object_urdf=obj,device='cuda:0',seed=42)
        panels=[]; all_gaps=[]; archived=[]
        with torch.inference_mode():
            for t,s in PANELS:
                path=args.input/f't{t}_s{s}_zero_a/response.pt'
                data=torch.load(path,map_location='cpu',weights_only=False)
                record=json.loads((path.parent/'results.json').read_text())
                if sha(path)!=record['output_sha256']:raise ValueError('physical output drift')
                inputs[str(path.resolve())]=sha(path)
                triggers=data['triggers']; env=torch.arange(len(triggers))
                state=data['state_before'][triggers,env].cuda()
                chunks=[]
                for block in state.split(16):
                    if time.monotonic()-started>115:raise TimeoutError('geometry audit budget')
                    geometry=bridge.current(block[:,:18],block[:,36:49])
                    minimum=torch.full((len(block),),float('inf'),device='cuda')
                    for points in geometry.hand_points.split(256,dim=1):
                        minimum=torch.minimum(minimum,torch.cdist(points,geometry.object_points).amin(dim=(1,2)))
                    if not torch.isfinite(minimum).all():raise FloatingPointError('nonfinite distance')
                    chunks.append(minimum.cpu())
                gaps=torch.cat(chunks); all_gaps.append(gaps)
                panels.append(dict(panel=f't{t}_s{s}',summary=summarize(gaps),
                                   by_motion={str(m):summarize(gaps[data['motion_id']==m]) for m in range(3)},
                                   current_force_proxy_fraction=float(state[:,49:51].bool().all(-1).float().mean())))
                archived.append(dict(panel=f't{t}_s{s}',gaps_m=gaps,state=state.cpu(),motion=data['motion_id']))
        total=summarize(torch.cat(all_gaps))
        report=dict(experiment_id=manifest['experiment_id'],run_status='COMPLETED',
                    label='PROMISING' if total['near20mm_fraction']>=.8 else 'UNPROMISING',
                    decision='PRESERVE_TRIGGER_FOR_COVERAGE_DESIGN' if total['near20mm_fraction']>=.8 else 'REPLACE_FORCE_PROXY_TRIGGER_WITH_GEOMETRIC_PROXIMITY',
                    summary=total,panels=panels,
                    boundary='sampled surface gap only; reused data; not actual pairwise contact or policy utility')
        torch.save(archived,args.output/'current_geometry.pt')
        if any(sha(Path(p))!=value for p,value in inputs.items()):raise ValueError('input drift')
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',input_sha256=inputs,inputs_unchanged=True)
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
