"""Own a matched shared7k current-input continuation and fixed-endpoint grasp screen."""
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
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:raise ValueError('Fresh Task output required')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        command=[sys.executable,str(Path(__file__).resolve())]+[s for s in sys.argv[1:] if s!='--detach']
        with (scratch/'driver.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        print(json.dumps(dict(supervisor_pid=child.pid,output=str(out),log=str(scratch/'driver.log'))));return
    import numpy as np
    for gpu in (6,7):
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Assigned GPU occupied')
    out.mkdir();started=time.monotonic()
    old=ROOT/'outputs/ptv3-shared-wam-v2/shared-actor-shared-20261015-r1'
    parent=ROOT/'outputs/ptv3-shared-wam-v2/remote-parent-20261015-r1'
    robot=ROOT/'outputs/ptv3-shared-wam-v2/shared-native-grasp-20261015-r1/inputs/control-step7000.pt'
    card=TASK/'docs/experiments/probes/P-20261015-shared-current-continuation.md'
    config=json.loads((old/'manifest.json').read_text())['config']
    config.update(seed=198,training_rng_seed=198,updates=1000,batch_size=256,validation_seed=192,
        validation_samples=256,validation_interval=500,checkpoint_interval=500,checkpoint_keep_recent=1,
        checkpoint_milestone_interval=500,save_initial_actor=True,max_seconds=4200,artifact_budget_bytes=7_000_000_000)
    for arm in ('masked','delta0'):
        (out/(arm+'.json')).write_text(json.dumps(dict(config,current_observation_input=arm),indent=2)+'\n')
    status=dict(status='RUNNING',supervisor_pid=os.getpid(),code_root=str(TASK.parents[2]),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),gpus=[6,7],stage='smoke')
    def write():
        part=out/'process.json.part';part.write_text(json.dumps(status,indent=2)+'\n');part.replace(out/'process.json')
    write()
    env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='')
    arms=[('masked',6),('delta0',7)]
    def train(item,smoke):
        arm,gpu=item;name=('smoke-' if smoke else 'train-')+arm
        cmd=[sys.executable,str(TASK/'tools/run/launch_robot_finetune.py'),'--output',str(out/name),
            '--config',str(out/(arm+'.json')),'--stats',str(parent/'stats_step80000.json'),
            '--source',str(ROOT/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1'),
            '--init-shared',str(parent/'step80000.pt'),'--init-robot',str(robot),
            '--experiment-card',str(card),'--gpus',str(gpu),'--loader-workers','2','--sequence-cache','128',
            '--pin-memory','--poll-seconds','20','--artifact-budget-bytes','7000000000']
        if smoke:cmd+=['--smoke','--smoke-batch-size','256']
        with (out/(name+'.log')).open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda x:train(x,True),arms))
        runs=[out/('smoke-'+arm) for arm,_ in arms]
        manifests=[json.loads((r/'manifest.json').read_text()) for r in runs]
        configs=[{k:v for k,v in m['config'].items() if k!='current_observation_input'} for m in manifests]
        assert configs[0]==configs[1]
        for file in ('rollout_admission.json','encoder_workspace_admission.json'):
            assert json.loads((runs[0]/file).read_text())==json.loads((runs[1]/file).read_text())
        snapshot=json.loads((runs[0]/'encoder_workspace_admission.json').read_text())
        previous=json.loads((old/'encoder_workspace_admission.json').read_text())
        assert snapshot==previous
        assert [snapshot[s]['retained_windows'] for s in ('train','val')]==[93680,10376]
        for r in runs:assert np.array_equal(np.load(r/'validation.npy'),np.load(old/'validation.npy'))
        with np.load(runs[0]/'initial-actor.npz') as a0,np.load(runs[1]/'initial-actor.npz') as b0:
            assert set(a0.files)==set(b0.files) and all(np.array_equal(a0[k],b0[k]) for k in a0.files)
            assert not a0['current_encoder.2.weight'].any() and not a0['current_encoder.2.bias'].any()
        audits=[json.loads((r/'engineering_audit.json').read_text()) for r in runs]
        assert abs(audits[0]['loss']-audits[1]['loss'])<1e-5
        (out/'matched-smoke.json').write_text(json.dumps(dict(status='PASS',same_initial_actor=True,
            same_initial_loss=True,same_admission_and_original_DEV=True,optimizer_reset_both=True),indent=2)+'\n')
        # Verify live delta0 request/role/clock/PD before spending the continuation budget.
        legacy=TASK.parent/'ptv3-shared-wam'
        def native(item,sequence,controls,label,smoke=False):
            arm,gpu=item;name='native-'+arm+'-'+label;target=out/name
            checkpoint=out/(('smoke-' if smoke else 'train-')+arm)/('step4.pt' if smoke else 'step1000.pt')
            cmd=[sys.executable,str(legacy/'tools/run/launch_native.py'),'--model-version','v2','--shared-actor',
                '--checkpoint',str(checkpoint),'--output',str(target),'--gpu',str(gpu),'--controls',str(controls),
                '--execution-steps','1','--seed','148','--sequence',sequence,'--experiment-card',str(card)]
            with (out/(name+'.log')).open('x') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,str(legacy/'tools/audit/audit_native.py'),'--native',str(target/'native')],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
            return json.loads((target/'native/result.json').read_text())
        status['stage']='native-smoke';write()
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda x:native(x,'s10_airplane_pass_1',4,'smoke',True),arms))
        status['stage']='train1000';write()
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda x:train(x,False),arms))
        status['stage']='fixed-endpoint-native';write()
        def qualify(item):
            return item[0],{label:native(item,seq,n,label) for label,seq,n in
                [('airplane','s10_airplane_pass_1',149),('sphere','s4_spheresmall_pass_1',141),('mug','s10_mug_pass_1',172)]}
        with ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(qualify,arms))
        final={arm:json.loads((out/('train-'+arm)/'curve.json').read_text())[-1] for arm,_ in arms}
        assert all(m['step']==1000 for m in final.values())
        (out/'comparison.json').write_text(json.dumps(dict(status='COMPLETED',final=final,native=results,
            scope='One-seed shared7k+1000 paired current-input Probe; no formal success rate'),indent=2)+'\n')
        status.update(status='COMPLETED')
    except BaseException as e:
        status.update(status='FAILED',error=repr(e));raise
    finally:status['elapsed_s']=time.monotonic()-started;write()


if __name__=='__main__':main()
