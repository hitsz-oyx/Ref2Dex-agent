#!/usr/bin/env python3
"""Evaluate frozen checkpoint on sequence-held-out test panels without fitting."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.data import Windows, balanced_indices
from oakink_wm.pointworld_temporal import model_from_config
spec = importlib.util.spec_from_file_location('trainer', TASK/'tools/run/train_oakink2_pointworld_temporal.py')
trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=256)
    p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--split',choices=['val','test'],default='test')
    a=p.parse_args();torch.set_num_threads(2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if state['dataset_hash']!=trainer.digest(a.data/'processed/manifest.json'):raise ValueError('dataset mismatch')
    sources=state['identity'].get('implementation_sources', {})
    if not sources or any(trainer.digest(TASK/key)!=value for key,value in sources.items()):
        raise ValueError('checkpoint implementation drift')
    from oakink_wm.pointworld_temporal import VENDOR
    if any(trainer.digest(VENDOR/key)!=value for key,value in state['identity']['vendor_sources'].items()):
        raise ValueError('checkpoint vendor drift')
    c=state['config'];arm=state['identity']['arm']
    model=model_from_config(state['identity']['stats'], c).cuda()
    model.load_state_dict(state['model'])
    test=Windows(a.data,a.split)
    balanced=balanced_indices(test,a.samples,214 if a.split=='test' else 212)
    natural=np.random.default_rng(215 if a.split=='test' else 213).choice(len(test),a.samples,replace=len(test)<a.samples)
    result=dict(split=a.split,checkpoint_sha256=trainer.digest(a.checkpoint),step=state['step'],arm=arm,
                engineering_only=a.engineering_only,
                balanced=trainer.evaluate(model,test,balanced,arm,c['microbatch'],c['amp']),
                natural=trainer.evaluate(model,test,natural,arm,c['microbatch'],c['amp']))
    if arm=='action':result['shuffle']=trainer.evaluate(model,test,balanced,arm,c['microbatch'],c['amp'],True)
    a.output.parent.mkdir(parents=True,exist_ok=True);trainer.atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('balanced','natural','shuffle')}))


if __name__=='__main__':main()
