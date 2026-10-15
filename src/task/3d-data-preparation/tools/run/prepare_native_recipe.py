"""Materialize an engineering sampling recipe and one shared TRAIN normalizer."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

TASK=Path(__file__).resolve().parents[2]
REPO=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.native_recipe import recipe_for_snapshot
from clean_native_corpus import write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=TASK/'configs/native_sampling_engineering_v1.json')
    p.add_argument('--artifact-root',type=Path,default=REPO,help='Explicit artifact workspace when code runs in a detached snapshot')
    a=p.parse_args();out=a.output.resolve()
    if a.artifact_root.resolve()/'outputs'/TASK.name not in out.parents or out.exists():
        raise ValueError('fresh Task recipe directory required')
    if shutil.disk_usage(out.parent).free<20*2**30:
        raise RuntimeError('less than 20GiB free')
    recipe,stats=recipe_for_snapshot(a.root,json.loads(a.policy.read_text()))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'recipe.json',recipe)
    write_json(out/'horizon-stats.json',stats)
    write_json(out/'manifest.json',dict(status='COMPLETED',training_allowed=False,
        recipe='recipe.json',horizon_stats='horizon-stats.json',code_root=str(REPO),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    print(json.dumps(recipe['source_probabilities']),flush=True)
