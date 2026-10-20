"""Restore world4 explicitly from a preserved world3 checkpoint after repair."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.topology import restore_four_gpu_checkpoint
from ptv3_shared_wam_v5.storage import read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','config','manifest','stats','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--new-rank-device',default='cuda:3')
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(2)
    if a.output.exists():raise ValueError('fresh transition output required')
    validate_recipe(a.manifest,a.stats);config=read_json(a.config)
    identity=hashlib.sha256(a.manifest.read_bytes()+a.stats.read_bytes()).hexdigest()
    try:saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    except PermissionError:
        a.checkpoint.chmod(0o664);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    seed=int(config['seed'])*1000003+int(saved['step'])*97+3
    # Dedicated generators do not alter any retained rank's RNG payload.
    state=dict(python=random.Random(seed).getstate(),numpy=np.random.RandomState(seed%(2**32)).get_state(),
        torch=torch.Generator().manual_seed(seed).get_state(),
        cuda=torch.Generator(device=a.new_rank_device).manual_seed(seed).get_state().cpu())
    result=restore_four_gpu_checkpoint(saved,config,identity,state,seed)
    result['topology_transition'].update(parent_checkpoint=str(a.checkpoint.resolve()),git_commit=a.git_commit)
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=a.output/('step%06d.pt'%result['step']);pending=checkpoint.with_suffix('.pt.part')
    torch.save(result,pending);pending.replace(checkpoint);checkpoint.chmod(0o664)
    report=dict(status='COMPLETED',checkpoint=str(checkpoint),step=result['step'],
        transition=result['topology_transition'],optimizer_states=len(result['optimizer']['state']),
        retained_weights_and_optimizer=True,original_elapsed_s=result['elapsed_s'],
        elapsed_s=time.monotonic()-started,python=sys.version,torch=torch.__version__,
        device_note='CPU checkpoint conversion; GPU only creates a fresh dedicated rank3 CUDA RNG state')
    out=a.output/'report.json';out.write_text(json.dumps(report,indent=2)+'\n');out.chmod(0o664)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
