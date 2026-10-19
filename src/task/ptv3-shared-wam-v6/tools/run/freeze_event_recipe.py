"""Admit a separate v6 engineering-continuation recipe without editing v5/cache."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.contracts import DATA_SCHEMA
from ptv3_shared_wam_v6.events import EVENT_ENCODING
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.storage import read_json, load_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent-manifest','events','stats','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    parent=read_json(a.parent_manifest)
    recipe=dict(schema=DATA_SCHEMA,status='FROZEN',training_allowed=True,
        parent_manifest=str(a.parent_manifest.resolve()),event_manifest=str(a.events.resolve()),
        stats=str(a.stats.resolve()),parent_manifest_identity=metadata_identity(a.parent_manifest),
        event_manifest_identity=metadata_identity(a.events),stats_identity=metadata_identity(a.stats),
        event_encoding=EVENT_ENCODING,event_dropout=.15,source_ids=parent['source_ids'],
        windows={s:int(len(load_array(parent['indices'][s],mmap_mode='r'))) for s in ('train','dev')},
        git_commit=a.git_commit,scope='user-authorized v6 engineering continuation Probe',
        limitation='demonstration-derived proximity event annotations; not force contact or certified task intent')
    path=a.output/'manifest.json';path.write_text(json.dumps(recipe,indent=2)+'\n');path.chmod(0o664)
    (a.output/'stats.json').symlink_to(a.stats.resolve())
    validate_recipe(path,a.stats)
    print(json.dumps(recipe),flush=True)


if __name__=='__main__':main()
