#!/usr/bin/env python3
"""One authorized exact-config resume, with a separate finite execution budget."""
import argparse, hashlib, importlib.util, json, math, os, signal, subprocess, time
from pathlib import Path

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
spec=importlib.util.spec_from_file_location('tau_guard',TASK/'tools/run/launch_tau_no_goal_no_event.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
OUT_FOR_RECEIPT=None
OWNED_RUN=False


def main():
    global OUT_FOR_RECEIPT,OWNED_RUN
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checks',type=Path,required=True);p.add_argument('--queue',action='store_true')
    p.add_argument('--target-step',type=int,default=3000);p.add_argument('--run-seconds',type=int,default=7200)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    OUT_FOR_RECEIPT=out
    if not 0<a.run_seconds<=7200 or not 1768<a.target_step<=3000:
        raise ValueError('this authorization is bounded at step3000 / 7200 new seconds')
    config_path=TASK/'configs/tau_no_goal_no_event_train_v1.json';config=G.read(config_path)
    policy=G.read(config['condition_recipe']);checks=G.read(a.checks)
    with G.recipe_guard(config['condition_recipe_sha256'],out):
        with (out/'ONE_SHOT.lock').open('x') as f:f.write(str(os.getpid()))
        OWNED_RUN=True
        pg=subprocess.run(['pgrep','-af','train_pretrain.py'],capture_output=True,text=True)
        (out/'pgrep-before-launch.txt').write_text(pg.stdout)
        if pg.returncode!=1 or pg.stdout:raise RuntimeError('existing training; never launch a second job')
        import torch
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if G.sha(config_path)!=G.sha(a.resume.parent/'config.json'):
            raise ValueError('frozen config file SHA changed')
        parent_manifest=ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'
        stats=parent_manifest.parent/'stats.json'
        data_sha=hashlib.sha256(parent_manifest.read_bytes()+stats.read_bytes()).hexdigest()
        if saved['config']!=config or saved['world_size']!=3 or saved['step']!=1768 or saved['data_identity']!=data_sha:
            raise ValueError('exact config/data/world/checkpoint continuation required')
        if len(saved['random_state'])!=3 or not saved.get('optimizer',{}).get('state'):
            raise ValueError('optimizer and all rank RNG states required')
        admission=dict(at=G.stamp(),checkpoint=str(a.resume.resolve()),saved_step=saved['step'],
            previous_elapsed=saved['elapsed_s'],config_sha256=G.sha(config_path),config_exact=True,
            optimizer_states=len(saved['optimizer']['state']),optimizer_groups=len(saved['optimizer']['param_groups']),
            rng_ranks=len(saved['random_state']),data_identity=data_sha,curated_identity=policy['data_identity'])
        G.write(out/'resume-admission.json',admission);del saved
        queue_deadline=time.time()+1800
        while True:
            # Budget-only admission view; frozen config is never modified.
            view=dict(config,hard_stop_epoch=time.time()+a.run_seconds)
            r=G.preflight(checks,policy,view);G.write(out/'preflight.json',r)
            if r['pass_all']:break
            bad=[k for k,v in r['gates'].items() if not v]
            G.write(out/'queue-status.json',dict(status='WAITING' if a.queue else 'BLOCKED',bad=bad,at=G.stamp()))
            if not a.queue or any(k!='authorized_gpus_free' for k in bad) or time.time()>queue_deadline:
                G.write(out/'queue-status.json',dict(status='BLOCKED',bad=bad,at=G.stamp()));return
            time.sleep(15)
        deadline=time.time()+a.run_seconds
        budget=dict(last_step=a.target_step,current_run_seconds=a.run_seconds,hard_stop_epoch=deadline,
                    previous_elapsed=admission['previous_elapsed'],config_sha256=G.sha(config_path))
        G.write(out/'execution-budget.json',budget)
        training=out/'training';head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        cmd=[str(G.PYTHON),'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',
            str(TASK/'tools/run/train_pretrain.py'),'--config',str(config_path),'--manifest',str(parent_manifest),
            '--stats',str(stats),'--output',str(training),'--resume',str(a.resume.resolve()),'--steps',str(a.target_step),
            '--resume-run-seconds',str(a.run_seconds),'--resume-hard-stop-epoch',str(deadline),
            '--workers','4','--git-commit',head,'--preserve-all-checkpoints','--checkpoint-budget-gib','32']
        env=G.runtime_environment(out)
        env.update(PYTHONPATH=str(out/'bootstrap')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''),
                   REF2DEX_B43_RUN_ROOT=str(out),REF2DEX_RESUME_FIRST_STEP='1769')
        with (out/'train-launch.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            G.write(out/'launch.json',dict(at=G.stamp(),pid=child.pid,supervisor_pid=os.getpid(),head=head,
                command=cmd,physical_gpus=[4,5,6],runtime_root=env['REF2DEX_RUN_ROOT'],tmpdir=env['TMPDIR'],
                config_sha256=G.sha(config_path),effective_stop_budget=budget,automatic_restart=False))
            reason=None
            while child.poll() is None:
                now=time.time();free=G.shutil.disk_usage('/home2').free;n=G.size(out)
                progress=G.read(training/'progress.json') if (training/'progress.json').exists() else {}
                loss=progress.get('loss');finite=loss is None or math.isfinite(loss)
                if free<20*2**30:reason='DISK_STOP_20_GIB'
                elif n>32*2**30:reason='ARTIFACT_STOP_32_GIB'
                elif not finite:reason='NONFINITE_LOSS'
                elif now>=deadline-30:reason=reason or 'HARD_STOP_SAVE_MARGIN_30_SECONDS'
                row=dict(at=G.stamp(),step=progress.get('step'),loss=loss,finite_loss=finite,free_bytes=free,
                         artifact_bytes=n,seconds_to_hard_stop=deadline-now,stop_reason=reason)
                G.write(out/'watchdog-latest.json',row)
                with (out/'watchdog.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if reason:training.mkdir(exist_ok=True);(training/'STOP').touch(exist_ok=True)
                if now>=deadline:
                    if os.getpgid(child.pid)!=child.pid:raise RuntimeError('unsafe own process group')
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(5)
            code=child.wait()
        result=G.read(training/'result.json') if (training/'result.json').exists() else None
        G.write(out/'stop-receipt.json',dict(at=G.stamp(),pid=child.pid,returncode=code,
            reason=G.terminal_reason(code,result,reason),result=result,artifact_bytes=G.size(out),
            effective_stop_budget=budget,automatic_restart=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OWNED_RUN and OUT_FOR_RECEIPT is not None and not (OUT_FOR_RECEIPT/'stop-receipt.json').exists():
            G.write(OUT_FOR_RECEIPT/'stop-receipt.json',dict(at=G.stamp(),returncode=1,
                reason='FAILED_ADMISSION_NO_RESTART',error=repr(exc),automatic_restart=False,
                result=dict(status='FAILED_ADMISSION',last_step=1768,optimizer_updates=0)))
        raise
