"""Publish a distinct v6 world3 checkpoint; never weaken normal resume checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.topology import transition_checkpoint
from ptv3_shared_wam_v5.storage import read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','config','manifest','stats','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    torch.set_num_threads(2);started=time.monotonic()
    validate_recipe(a.manifest,a.stats);config=read_json(a.config)
    identity=hashlib.sha256(a.manifest.read_bytes()+a.stats.read_bytes()).hexdigest()
    try:saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    except PermissionError:
        a.checkpoint.chmod(0o664);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    result=transition_checkpoint(saved,config,identity)
    result['topology_transition'].update(parent_checkpoint=str(a.checkpoint.resolve()),git_commit=a.git_commit)
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=a.output/('step%06d.pt'%result['step']);pending=checkpoint.with_suffix('.pt.part')
    torch.save(result,pending);pending.replace(checkpoint);checkpoint.chmod(0o664)
    report=dict(status='COMPLETED',checkpoint=str(checkpoint),step=result['step'],
        transition=result['topology_transition'],optimizer_states=len(result['optimizer']['state']),
        original_elapsed_s=result['elapsed_s'],elapsed_s=time.monotonic()-started,
        python=sys.version,torch=torch.__version__,device='CPU checkpoint transformation only')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'report.json').chmod(0o664);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
