"""Run frozen-final Shadow BC/PPO pairs on predeclared reset panels."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs=3,default=[1,3,4]);a=p.parse_args()
    a.run=a.run.resolve()
    source=json.loads((a.run/'train/manifest.json').read_text())
    if source['status']!='COMPLETED':raise ValueError('Completed final PPO required')
    final=a.run/'train/latest.pt';started=time.monotonic()
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()
    def pair(seed,gpu):
        import numpy as np
        out=a.run/('eval-seed%d'%seed);out.mkdir(exist_ok=False)
        reports={};initial={}
        for mode in ('bc','evaluate'):
            name='bc' if mode=='bc' else 'ppo'
            cmd=[sys.executable,'-u',str(TASK/'tools/run/shadow_ppo.py'),
                '--checkpoint',source['base_checkpoint'],'--cache',str(ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1'),
                '--object-manifest',str(ROOT/'outputs/shadow-finetune/pilot-panel-20261015-r1.json'),
                '--output',str(out/name),'--mode',mode,'--gpu',str(gpu),'--envs','16','--seed',str(seed),'--seconds','300']
            if mode=='evaluate':cmd+=['--rl-checkpoint',str(final)]
            with (out/(name+'-launch.log')).open('x') as stream:
                r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=360)
            if r.returncode:raise RuntimeError(name+' pair failed: '+str(out))
            reports[name]=json.loads((out/name/'result.json').read_text())['final']
            with np.load(out/name/'last-episode.npz') as z:
                initial[name]={k:z[k][0].astype(float) for k in ('q','dq','object','hand','hand_root','observation','goal','success','reset')}
        error={k:float(abs(initial['bc'][k]-initial['ppo'][k]).max()) for k in initial['bc']}
        if max(error.values())!=0:raise ValueError('BC/PPO initial states not exact: '+str(error))
        result=dict(seed=seed,gpu=gpu,roles=reports,initial_state_max_error=error,
            held_delta=reports['ppo']['native']['held60_with_lift10cm']-reports['bc']['native']['held60_with_lift10cm'])
        (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    process=dict(status='RUNNING',pid=__import__('os').getpid(),git_commit=code,seeds=[216,233,234],gpus=a.gpus)
    (a.run/'evaluation-process.json').write_text(json.dumps(process,indent=2)+'\n')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(pair,seed,gpu) for seed,gpu in zip((216,233,234),a.gpus)]
            result=[job.result() for job in jobs]
        (a.run/'evaluation-pairs.json').write_text(json.dumps(result,indent=2)+'\n')
        process.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    except BaseException as error:
        process.update(status='FAILED',error=repr(error));raise
    finally:(a.run/'evaluation-process.json').write_text(json.dumps(process,indent=2)+'\n')


if __name__=='__main__':main()
