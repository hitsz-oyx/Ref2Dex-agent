"""Matched small dev check for destructive warm-start updates, not model efficacy."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','candidate','source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam' not in out.parents:raise ValueError('fresh owned output required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import torch
    from ptv3_shared_wam.checkpoint import load_checkpoint,atomic_json
    from ptv3_shared_wam.rl_robot import RLRobotWindows
    from ptv3_shared_wam.model import SharedWAM
    from ptv3_shared_wam.robot import RobotPair,robot_loss
    from ptv3_shared_wam.loading import to_device
    from ptv3_shared_wam.data import shared_collate
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    dataset=RLRobotWindows(a.source,'val')
    ids=np.random.default_rng(126).choice(len(dataset),64,replace=False)
    samples=[dataset[int(i)] for i in ids];result={}
    for name,path in (('parent',a.parent),('candidate',a.candidate)):
        packet=load_checkpoint(path)
        model=RobotPair(SharedWAM(packet['stats'],packet['config']))
        model.load_state_dict(packet['model'],strict=True);model=model.cuda().eval()
        totals={}
        with torch.no_grad():
            for first in range(0,64,16):
                batch=to_device(shared_collate(samples[first:first+16]),torch.device('cuda'))
                prediction=model(batch);loss,terms=robot_loss(prediction,batch,packet['config'],model.retargeter)
                for key,value in dict(objective=loss,**terms).items():totals[key]=totals.get(key,0)+float(value)/4
        result[name]=totals
        del model,prediction,loss,terms,batch,packet
        gc.collect();torch.cuda.empty_cache()
    passed=all(np.isfinite(result['candidate'][key]) and result['candidate'][key]<=1.2*result['parent'][key]
               for key in ('objective','hand','action'))
    out.mkdir(parents=True)
    report=dict(status='PASSED' if passed else 'FAILED',fixed_dev_indices=ids.tolist(),samples=64,results=result,
        gate='objective/hand/action <=1.2x parent after four engineering updates',
        limitation='warm-start stability check; does not establish forecast or grasp improvement')
    atomic_json(out/'summary.json',report);print(json.dumps(report),flush=True)
    if not passed:raise AssertionError('early expanded updates disrupt existing forecasts')


if __name__=='__main__':main()
