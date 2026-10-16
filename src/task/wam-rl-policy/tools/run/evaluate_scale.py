"""Run every predeclared DEV shard and persist an aggregate native score."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panels',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--panel',choices=('dev','core_dev','train'),default='dev');p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--bc-kind',choices=('wam','mlp'),required=True);p.add_argument('--rl-checkpoint',type=Path)
    p.add_argument('--force-zero-residual',action='store_true')
    p.add_argument('--shard-indices',help='Comma-separated indices in the frozen panel, for parallel evaluation')
    p.add_argument('--seconds',type=int,default=7200);p.add_argument('--seed',type=int,default=249)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output')
    out.mkdir(parents=True);started=time.monotonic();panel=json.loads(a.panels.read_text())[a.panel]
    indices=[int(i) for i in a.shard_indices.split(',')] if a.shard_indices else list(range(len(panel)))
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(panel) for i in indices):raise ValueError('Unique valid panel shard indices required')
    manifest=dict(status='RUNNING',checkpoint=str(a.checkpoint.resolve()),bc_kind=a.bc_kind,seed=a.seed,
        rl_checkpoint=str(a.rl_checkpoint.resolve()) if a.rl_checkpoint else None,force_zero_residual=a.force_zero_residual,panel=a.panel,
        shards=[panel[i] for i in indices],source_shard_indices=indices,
        gpu=a.gpu,max_seconds=a.seconds,results=[],totals={k:0 for k in ('episodes','official_success','final_within5cm','held60_with_lift10cm')})
    def save():
        part=out/'manifest.json.part';part.write_text(json.dumps(manifest,indent=2)+'\n');part.replace(out/'manifest.json')
    save()
    try:
        for index in indices:
            path=panel[index]
            remaining=a.seconds-(time.monotonic()-started)
            if remaining<30:raise TimeoutError('Suite budget')
            objects=json.loads(Path(path).read_text())['objects'];target=out/('shard%02d'%index)
            command=[sys.executable,'-u',str(TASK/'tools/run/shadow_ppo.py'),'--checkpoint',str(a.checkpoint.resolve()),
                '--cache',str(a.cache.resolve()),'--object-manifest',path,'--output',str(target),'--mode','evaluate' if a.rl_checkpoint else 'bc',
                '--gpu',str(a.gpu),'--envs',str(len(objects)*4),'--seed',str(a.seed),'--seconds',str(min(1800,int(remaining))),
                '--bc-kind',a.bc_kind,'--experiment-id','P-20261016-shadow-scale-continuous']
            if a.rl_checkpoint:command+=['--rl-checkpoint',str(a.rl_checkpoint.resolve())]
            if a.force_zero_residual:command+=['--force-zero-residual']
            with (out/('shard%02d-supervisor.log'%index)).open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            native=json.loads((target/'result.json').read_text())['final']['native']
            for key in manifest['totals']:manifest['totals'][key]+=native[key]
            manifest['results'].append(dict(shard=index,objects=len(objects),result=str(target/'result.json'),**{k:native[k] for k in manifest['totals']}))
            manifest['elapsed_s']=time.monotonic()-started;save();print(json.dumps(manifest['totals']),flush=True)
        manifest['status']='COMPLETED'
    except BaseException as error:manifest.update(status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_s']=time.monotonic()-started;save()


if __name__=='__main__':main()
