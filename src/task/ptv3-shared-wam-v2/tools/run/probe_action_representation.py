"""Own matched target-parameterization arms and fixed-endpoint native readout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='6,7')
    p.add_argument('--detach',action='store_true');a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:raise ValueError('Fresh Task output')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        cmd=[sys.executable,str(Path(__file__).resolve())]+[x for x in sys.argv[1:] if x!='--detach']
        with (scratch/'driver.log').open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1'))
        print(json.dumps(dict(supervisor_pid=child.pid,output=str(out))));return
    import numpy as np
    gpus=a.gpus.split(',')
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Two distinct authorized GPUs')
    for gpu in gpus:
        if subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Selected GPU occupied')
    out.mkdir(parents=True);started=time.monotonic()
    status=dict(status='RUNNING',supervisor_pid=os.getpid(),code_root=str(TASK.parents[2]),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
        gpus=dict(zip(('native','geometry'),gpus)),stages=[])
    def write():
        part=out/'process.json.part';part.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');part.replace(out/'process.json')
    write()
    card=TASK/'docs/experiments/probes/P-20261015-inspire-action-representation.md'
    parent=ROOT/'outputs/ptv3-shared-wam-v2/remote-parent-20261015-r1'
    source=ROOT/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1'
    env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shared-wam-v2-extensions'))
    def train(item,smoke):
        arm,gpu=item;name=('smoke-' if smoke else 'train-')+arm
        cmd=[sys.executable,str(TASK/'tools/run/launch_robot_finetune.py'),
            '--output',str(out/name),'--config',str(TASK/('configs/action_repr_'+arm+'.json')),
            '--stats',str(parent/'stats_step80000.json'),'--source',str(source),
            '--init-shared',str(parent/'step80000.pt'),'--experiment-card',str(card),'--gpus',gpu,
            '--loader-workers','2','--sequence-cache','128','--pin-memory','--poll-seconds','20',
            '--artifact-budget-bytes','9000000000']
        if smoke:cmd+=['--smoke','--smoke-batch-size','256']
        with (out/(name+'.log')).open('x') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Training stage failed: '+name)
        return name
    try:
        arms=list(zip(('native','geometry'),gpus))
        with ThreadPoolExecutor(max_workers=2) as pool:status['stages']+=list(pool.map(lambda x:train(x,True),arms))
        write()
        runs=[out/('smoke-'+arm) for arm,_ in arms]
        manifests=[json.loads((r/'manifest.json').read_text()) for r in runs]
        configs=[{k:v for k,v in m['config'].items() if k!='action_representation'} for m in manifests]
        if configs[0]!=configs[1]:raise ValueError('Unmatched config outside representation')
        for name in ('validation.npy',):
            if not np.array_equal(np.load(runs[0]/name),np.load(runs[1]/name)):raise ValueError('DEV indices differ')
        if json.loads((runs[0]/'rollout_admission.json').read_text())!=json.loads((runs[1]/'rollout_admission.json').read_text()):
            raise ValueError('Training membership differs')
        with np.load(runs[0]/'initial-actor.npz') as n,np.load(runs[1]/'initial-actor.npz') as g:
            excluded={'head.weight','head.bias','action_mean','action_scale'}
            checked=[k for k in n.files if k not in excluded]
            if set(n.files)!=set(g.files) or any(not np.array_equal(n[k],g[k]) for k in checked):
                raise ValueError('Matched actor trunk or current-state normalization differs')
            if any(np.any(x[k]) for x in (n,g) for k in ('head.weight','head.bias')):raise ValueError('Readout not zero initialized')
        audits=[json.loads((r/'engineering_audit.json').read_text()) for r in runs]
        first={k:abs(audits[0]['terms'][k]-audits[1]['terms'][k]) for k in ('hand','scene')}
        if max(first.values())>1e-5:raise ValueError('Matched initial forecasting differs: '+str(first))
        match=dict(status='PASS',common_initial_tensors=checked,initial_forecast_abs_differences=first,
            same_DEV=True,same_admission=True,only_config_difference='action_representation',
            output_dimensions=dict(native=12,geometry=15))
        (out/'matched-smoke.json').write_text(json.dumps(match,indent=2)+'\n')
        with ThreadPoolExecutor(max_workers=2) as pool:status['stages']+=list(pool.map(lambda x:train(x,False),arms))
        write()
        def evaluate(item):
            arm,gpu=item;rows=[]
            for short,sequence,controls in [('airplane','s10_airplane_pass_1',149),('sphere','s4_spheresmall_pass_1',141),('mug','s10_mug_pass_1',172)]:
                native=out/('native-'+arm+'-'+short)
                cmd=[sys.executable,str(TASK.parent/'ptv3-shared-wam/tools/run/launch_native.py'),
                    '--model-version','v2','--shared-actor','--checkpoint',str(out/('train-'+arm)/'step2000.pt'),
                    '--output',str(native),'--gpu',gpu,'--controls',str(controls),'--execution-steps','1',
                    '--seed','148','--sequence',sequence,'--experiment-card',str(card)]
                with (out/('native-'+arm+'-'+short+'.log')).open('x') as f:
                    result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError('Native stage failed: '+str(native))
                    result=subprocess.run([sys.executable,str(TASK.parent/'ptv3-shared-wam/tools/audit/audit_native.py'),
                        '--native',str(native/'native')],env=env,stdout=f,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError('Native independent audit failed: '+str(native))
                rows.append(dict(sequence=sequence,result=json.loads((native/'native/result.json').read_text())))
            return arm,rows
        with ThreadPoolExecutor(max_workers=2) as pool:native=dict(pool.map(evaluate,arms))
        curves={arm:json.loads((out/('train-'+arm)/'curve.json').read_text()) for arm,_ in arms}
        final={arm:c[-1] for arm,c in curves.items()}
        if any(v['step']!=2000 for v in final.values()):raise ValueError('Missing common final update')
        (out/'comparison.json').write_text(json.dumps(dict(status='COMPLETED',match=match,final=final,native=native,
            scope='Single-seed target-representation package Probe, fixed endpoint; no cross-hand causal claim'),indent=2,allow_nan=False)+'\n')
        status.update(status='COMPLETED')
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));raise
    finally:
        status['elapsed_s']=time.monotonic()-started;write()


if __name__=='__main__':main()
