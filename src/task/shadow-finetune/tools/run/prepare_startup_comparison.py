"""Freeze a startup-sampling intervention against an existing uniform full-data run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from shadow_finetune import paths
from shadow_finetune.scale_controls import STARTUP_SCHEMA,load_controls,stamp,window_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();reference=a.reference_run.resolve()
    if out.exists() or paths.ROOT/'outputs/shadow-finetune' not in out.parents:
        raise ValueError('fresh Task output required')
    old=json.loads((reference/'controls.json').read_text());settings=old['settings']
    if settings['updates']!=1000 or settings['batch_size']!=64:
        raise ValueError('bounded fixed1000/batch64 reference required')
    c,dev,validation,_=load_controls(reference/'controls.json',old['training_caches']['full'],
        settings['parent'],settings['seed'],settings['updates'],settings['batch_size'])
    manifest=json.loads((reference/'train-full/manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['comparison']!=c or manifest.get('sampler',{}).get('startup_fraction',0.)!=0:
        raise ValueError('completed original uniform reference required')
    sys.path.insert(0,str(TASK/'tools/audit'))
    from summarize_startup import implementation_check
    code_audit=implementation_check(Path(manifest['code_root']),TASK.parents[2])
    startup_ids=[i for i,key in enumerate(dev.keys) if key[1]==0]
    if len(startup_ids)<64:raise ValueError('64 admitted DEV starts required')
    selected=np.random.default_rng(218).choice(startup_ids,64,replace=False).tolist()
    controls=dict(c,schema=STARTUP_SCHEMA,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
        arms=['startup'],summary_entry='summarize_startup.py',startup_fraction=.5,
        artifact_budget_bytes=8*10**9,
        training_caches=dict(startup=c['training_caches']['full']),reference_run=str(reference),
        reference_code_commit=manifest['git_commit'],startup_dev_seed=218,
        reference_implementation_check=code_audit,
        startup_validation_ids=selected,startup_validation_keys=[window_key(dev,i) for i in selected],
        comparison='TRAIN sampling only:50% mixture of sequence-uniform admitted query0..7; original all-query component retained',
        execution='prepared; one temporary GPU6, total<=1hour/8GB, preserve Inspire GPU0..5 and reserve GPU7')
    extra=[reference/'controls.json',reference/'train-full/step1000.pt',reference/'train-full/initial-actor.npz',
           reference/'train-full/manifest.json',reference/'native-final-full/comparison.json']
    controls['input_stamps']=[*c['input_stamps'],*[dict(path=str(f),stamp=stamp(f)) for f in extra]]
    out.mkdir(parents=True)
    file=out/'controls.json';file.write_text(json.dumps(controls,indent=2)+'\n')
    train=[sys.executable,str(TASK/'tools/run/train.py'),'--data',c['training_caches']['full'],
        '--controls',str(file),'--parent',settings['parent'],'--seed',str(settings['seed']),
        '--steps','1000','--batch-size','64','--startup-fraction','.5','--output',str(out/'train-startup'),'--gpu','6']
    native=[sys.executable,str(TASK/'tools/run/evaluate.py'),'--cache',c['validation_cache'],
        '--checkpoint',str(out/'train-startup/step1000.pt'),'--object-manifest',c['native_panel'],
        '--seed',str(c['native_reset_seed']),'--envs','16','--output',str(out/'native-final-startup'),'--gpu','6']
    (out/'commands.json').write_text(json.dumps(dict(startup=dict(train=train,evaluate_final=native)),indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED',reference=str(reference),startup_dev_queries=len(selected),output=str(out))))


if __name__=='__main__':main()
