#!/usr/bin/env python3
"""GPU FK/live audit and archive actual corresponding hand flows for fixed panels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
import torch
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
from oracle_hand_flow import measured_flow_inputs
from consequence_sufficiency import readouts
from oracle_y_utility import stable_grasp_z


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batches',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists(): raise ValueError('unique artifact required')
    begin=time.monotonic();torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise ValueError('GPU FK required')
    asset=ROOT/'third_party/DExplore/dexplore/data/assets'
    bridge=DExploreCmv2GeometryBridge(hand_urdf=asset/'inspire_hand_new/inspire_hand_right.urdf',
        object_urdf=asset/'mjcf/airplane.urdf',device='cuda:0')
    batches=[];checks=[];hashes={}
    for batch in args.batches:
        result=json.loads(batch.read_text());screen=result['screen'];rows=torch.tensor(screen['indices'],dtype=torch.long)
        paths=[Path(p) for p in result['input_sha256'] if p.endswith('panel.pt')]
        if len(paths)!=7: raise ValueError('complete candidate panel required')
        flows=[];targets=[];outcomes=[];all_z=[]
        for path in paths:
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            if h!=result['input_sha256'][str(path)]: raise ValueError('panel input drift')
            hashes[str(path)]=h
            raw=torch.load(path,map_location='cpu',weights_only=False)
            p={k:(v[rows] if isinstance(v,torch.Tensor) and len(v)==len(raw['triggers']) else v)
               for k,v in raw.items() if isinstance(v,torch.Tensor) and v.ndim>0}
            geometry,action,error=measured_flow_inputs(p,bridge)
            e,i,_,y,_=readouts(p);z,_=stable_grasp_z(p['height'],p['pair'],p['rest_height'])
            flows.append(action['Chunk'].cpu());targets.append(torch.cat((e,i),-1));outcomes.append(y);all_z.append(z)
            checks.append(dict(panel=str(path),candidate=raw['candidate'],rows=len(rows),fk_checks=error))
        value=dict(flow=torch.stack(flows,1),ei=torch.stack(targets,1),y=torch.stack(outcomes,1),
            z=torch.stack(all_z,1),rows=rows,batch_result=str(batch.resolve()))
        batches.append(value)
    y=torch.cat([b['y'] for b in batches]).numpy();z=torch.cat([b['z'] for b in batches]).numpy()
    torch.save(dict(schema='ref2dex.paired_actual_flow.v1',batches=batches,
        input_sha256=hashes,flow_contract='Actual0→4/4→8 corresponding120handpoints inCURRENT object frame, divided by.02m; not attainable desiredflow'),args.output)
    report=dict(status='PASS',physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        input_sha256=hashes,checks=checks,elapsed_seconds=time.monotonic()-begin,
        artifact_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],panels=len(checks),anchors=len(y),elapsed=report['elapsed_seconds'])))
if __name__=='__main__':main()
