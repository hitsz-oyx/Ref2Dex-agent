"""Publish a distinct fully bound v6 checkpoint from one complete v5 parent."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.migration import migrate_checkpoint
from ptv3_shared_wam_v5.storage import read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','manifest','stats','config','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    started=time.time();torch.set_num_threads(2);torch.manual_seed(170)
    a.output.mkdir(parents=True,exist_ok=False)
    recipe=validate_recipe(a.manifest,a.stats);config=read_json(a.config)
    a.checkpoint.chmod(0o664)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=SharedWAMV6(read_json(a.stats),config)
    original_manifest=a.checkpoint.parent/'run-manifest.json'
    legacy=read_json(original_manifest) if saved.get('execution_contract') is None else None
    result=migrate_checkpoint(saved,model,a.manifest,a.stats,recipe,config,saved['world_size'],legacy)
    result['migration'].update(parent_checkpoint=str(a.checkpoint.resolve()),git_commit=a.git_commit,
        parent_run_manifest=str(original_manifest),branch_initialization_seed=170)
    target=a.output/('step%06d.pt'%result['step']);pending=target.with_suffix('.pt.part')
    torch.save(result,pending);pending.replace(target);target.chmod(0o664)
    report=dict(status='COMPLETED',checkpoint=str(target.resolve()),step=result['step'],
        world_size=result['world_size'],migration=result['migration'],elapsed_s=time.time()-started,
        original_elapsed_s=result['elapsed_s'],python=sys.version,torch=torch.__version__,
        manifest=str(a.manifest.resolve()),config=config,git_commit=a.git_commit,
        device='CPU checkpoint transformation only; model inference/training uses GPU')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
