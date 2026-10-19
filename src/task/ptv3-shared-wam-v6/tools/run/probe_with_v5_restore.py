"""Pause owned v5, test v6 migration/update/recovery, then restore v5 in finally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.migration import validate_expansion_config
from ptv3_shared_wam_v5.storage import read_json


def write(path,value):
    pending=path.with_suffix('.json.part');pending.write_text(json.dumps(value,indent=2)+'\n')
    pending.replace(path);path.chmod(0o664)


def gpu_free(gpus):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    rows={int(s.split(',')[0]):int(s.split(',')[1]) for s in raw.splitlines()}
    return all(rows[g]<256 for g in gpus)


def wait_free(gpus,seconds=180):
    end=time.monotonic()+seconds
    while not gpu_free(gpus):
        if time.monotonic()>end:raise TimeoutError('GPUs not released; do not overlap original training')
        time.sleep(5)


def confirmed_command(pid,needle):
    try:return needle in Path('/proc/%d/cmdline'%pid).read_bytes().decode().replace('\0',' ')
    except FileNotFoundError:return False


def terminate_owned(child,output):
    """Only our launcher and its recorded, path-verified torchrun group."""
    marker=output/'process.json'
    if marker.exists():
        pid=read_json(marker).get('torchrun_pid')
        if pid and confirmed_command(pid,str(output)):
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    if child.poll() is None:
        child.terminate()
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:child.kill();child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('v5-run','v5-code-root','base-code-root','manifest','stats','config','output','restore-output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--git-commit',required=True);p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 500<=a.updates<=2000:raise ValueError('sustained engineering check requires500..2000 updates')
    a.output.mkdir(parents=True,exist_ok=False)
    recipe=validate_recipe(a.manifest,a.stats);config=read_json(a.config);old_config=read_json(a.v5_run/'config.json')
    validate_expansion_config(old_config,config)
    original=read_json(a.v5_run/'process.json');original_manifest=read_json(a.v5_run/'run-manifest.json')
    parent_identity=hashlib.sha256(Path(recipe['parent_manifest']).read_bytes()+Path(recipe['stats']).read_bytes()).hexdigest()
    if (original_manifest.get('data_identity')!=parent_identity or original_manifest.get('world_size')!=4
            or original_manifest.get('config')!=old_config):
        raise ValueError('v6 parent must match the active v5 data/config/world before pausing it')
    if (original.get('status')!='RUNNING' or (a.v5_run/'STOP').exists()
            or not confirmed_command(original['torchrun_pid'],str(a.v5_run))
            or str(a.v5_code_root/'src/task/ptv3-shared-wam-v5/tools/run/train_pretrain.py') not in original['command']):
        raise ValueError('only the confirmed active v5 run may be paused')
    if a.restore_output.exists():raise ValueError('fresh v5 restoration output required')
    gpus=list(map(int,a.gpus.split(',')));started=time.monotonic();baseline=None;error=None;pause_requested=False
    state=dict(status='PREPARED',git_commit=a.git_commit,v5_run=str(a.v5_run),v5_code_root=str(a.v5_code_root),
        original_git_commit=original_manifest['git_commit'],updates=a.updates,phases={},
        restore_output=str(a.restore_output),user_preference='restore v5 after the v6 probe',
        environment=sys.executable,max_probe_seconds=3600,workers_per_rank=a.workers)
    write(a.output/'run-manifest.json',state)

    def progress(stage):
        state.update(status=stage,elapsed_s=time.monotonic()-started)
        write(a.output/'progress.json',state);print(json.dumps(dict(stage=stage,elapsed_s=state['elapsed_s'])),flush=True)

    def run_command(name,command,output,seconds):
        progress(name);child=None
        with (a.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            deadline=time.monotonic()+seconds
            try:
                while child.poll() is None:
                    if time.monotonic()>deadline or time.monotonic()-started>3600:
                        raise TimeoutError('owned phase/total probe deadline: '+name)
                    time.sleep(10)
                if child.returncode:raise RuntimeError('%s exited%d; see owned phase log'%(name,child.returncode))
            finally:
                if child.poll() is None:
                    if output.exists():
                        (output/'STOP').touch()
                        try:child.wait(timeout=200)
                        except subprocess.TimeoutExpired:terminate_owned(child,output)

    def phase(name,checkpoint,extra,expected):
        output=a.output/name
        command=[sys.executable,str(TASK/'tools/run/launch_pretrain.py'),'--config',str(a.config),
            '--manifest',str(a.manifest),'--stats',str(a.stats),'--output',str(output),
            '--base-code-root',str(a.base_code_root),'--gpus',a.gpus,'--workers',str(a.workers),
            '--git-commit',a.git_commit,'--resume',str(checkpoint)]+extra
        run_command(name,command,output,2700 if expected>16 else 1000)
        result=read_json(output/'result.json')
        if result['status']!='COMPLETED' or result['optimizer_updates']!=expected:
            raise ValueError('phase did not complete all planned updates: '+name)
        signatures=result.get('ddp_parameter_signatures')
        if not signatures or len(signatures)!=4 or any(x!=signatures[0] for x in signatures):
            raise ValueError('four-rank parameter signatures failed: '+name)
        log=output/'train.jsonl';log.chmod(0o664);rows=[json.loads(line) for line in log.read_text().splitlines()]
        if not any(row.get('event_goal_gradient_norm_after_clip',0)>0 for row in rows):
            raise ValueError('event branch has no measured gradient: '+name)
        if rows[-1].get('event_goal_output_weight_norm',0)<=0:
            raise ValueError('event branch did not update: '+name)
        state['phases'][name]=dict(result=result,event_first=rows[0],event_last=rows[-1],output=str(output))
        wait_free(gpus)
        return output/('step%06d.pt'%result['last_step'])

    def restore():
        nonlocal baseline
        if not pause_requested:return
        if baseline is None:
            # A failed phase must not leave the baseline stopped just because
            # its stop-result read failed. Only published complete files count.
            checkpoints=sorted(a.v5_run.glob('step*.pt'))
            if not checkpoints:raise RuntimeError('no complete v5 checkpoint available for restoration')
            baseline=checkpoints[-1]
            state.update(baseline_step=int(baseline.stem[4:]),baseline_checkpoint=str(baseline),
                         restoration_fallback='latest published checkpoint after abnormal stop; inspect original result')
        progress('RESTORING_V5');wait_free(gpus)
        launch=a.v5_code_root/'src/task/ptv3-shared-wam-v5/tools/run/launch_pretrain.py'
        command=[sys.executable,str(launch),'--config',str(a.v5_run/'config.json'),
            '--manifest',recipe['parent_manifest'],'--stats',recipe['stats'],'--output',str(a.restore_output),
            '--base-code-root',str(a.base_code_root),'--gpus',a.gpus,'--workers',str(a.workers),
            '--git-commit',original_manifest['git_commit'],'--resume',str(baseline)]
        with (a.output/'restore-v5-launch.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state['v5_restore']=dict(supervisor_pid=child.pid,command=command,parent_checkpoint=str(baseline))
        end=time.monotonic()+300
        while time.monotonic()<end:
            if child.poll() is not None:raise RuntimeError('v5 restoration launcher exited%d'%child.returncode)
            path=a.restore_output/'progress.json'
            if path.exists():
                row=read_json(path)
                if row['status']=='RUNNING' and row['step']>state['baseline_step']:
                    state['v5_restore']['progress']=row;state['v5_restore']['verified']=True;return
            time.sleep(10)
        raise TimeoutError('v5 restoration has not produced a completed update within300s')

    try:
        progress('STOPPING_V5');(a.v5_run/'STOP').touch(exist_ok=False);pause_requested=True
        end=time.monotonic()+300
        while not (a.v5_run/'result.json').exists():
            if time.monotonic()>end:raise TimeoutError('original v5 did not publish controlled-stop result')
            time.sleep(5)
        stopped=read_json(a.v5_run/'result.json')
        if stopped['status'] not in ('STOPPED','COMPLETED'):raise ValueError('original v5 stop did not save normally')
        baseline=a.v5_run/('step%06d.pt'%stopped['last_step'])
        if not baseline.is_file():raise ValueError('original v5 final checkpoint missing')
        state.update(baseline_step=stopped['last_step'],baseline_checkpoint=str(baseline),v5_stop_result=stopped)
        wait_free(gpus)
        migrate=a.output/'migration'
        run_command('MIGRATING_V5_TO_V6',[sys.executable,str(TASK/'tools/run/migrate_v5_checkpoint.py'),
            '--checkpoint',str(baseline),'--manifest',str(a.manifest),'--stats',str(a.stats),
            '--config',str(a.config),'--output',str(migrate),'--git-commit',a.git_commit],migrate,300)
        report=read_json(migrate/'report.json');state['migration']=report
        first=phase('initial-recovery',Path(report['checkpoint']),['--recovery-probe-updates','8'],8)
        sustained=phase('sustained',first,['--continuation-updates',str(a.updates)],a.updates)
        final=phase('resume-recovery',sustained,['--recovery-probe-updates','8'],8)
        state['v6_final_checkpoint']=str(final);state['v6_probe_status']='COMPLETED'
    except BaseException as exc:
        error=exc;state.update(v6_probe_status='FAILED',error=repr(exc))
    finally:
        try:restore()
        except BaseException as exc:
            state.update(restore_error=repr(exc));error=error or exc
        state.update(status='FAILED' if error else 'COMPLETED',elapsed_s=time.monotonic()-started)
        write(a.output/'report.json',state);write(a.output/'progress.json',state)
    if error:raise error


if __name__=='__main__':main()
