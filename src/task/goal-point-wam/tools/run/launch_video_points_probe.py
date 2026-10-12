#!/usr/bin/env python3
"""Run the fixed short mixed Probe after qualified video expansion."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('plan','expansion','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[5]; task = repo/'src/task/goal-point-wam'
    spec = importlib.util.spec_from_file_location('owned_helpers',repo/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    plan = json.loads(a.plan.read_text()); expansion = json.loads((a.expansion/'stage_status.json').read_text())
    if expansion['status'] != 'COMPLETED' or not expansion['training_allowed']:
        raise ValueError('completed multi-session qualified expansion required')
    video = Path(expansion['qualified']); data_meta = json.loads((video/'manifest.json').read_text())
    if helper.digest(video/'manifest.json') != expansion['qualified_manifest_sha256']:
        raise ValueError('qualified view changed')
    if a.output.exists() or helper.gpu_processes([a.gpu]):
        raise RuntimeError('new output and idle assigned GPU required')
    parent_root = Path(plan['original_training']); parent = parent_root/'train/final.pt'
    if json.loads((parent_root/'group_status.json').read_text())['status'] != 'COMPLETED':
        raise ValueError('native parent incomplete')
    a.output.mkdir(parents=True)
    state = dict(status='SMOKE',pid=os.getpid(),run_id=a.output.name,started_at=time.time(),assigned_gpu=a.gpu,
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                 train_windows=expansion['train_windows'],val_windows=expansion['val_windows'],sessions=expansion['sessions'],
                 expansion_run=a.expansion.name,scope_cap_bytes=25_000_000_000,artifact_cap_bytes=1_500_000_000,
                 steps=1000,parent=str(parent),parent_sha256=helper.digest(parent))
    stop = [False]
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    controls = {str(f):helper.digest(f) for f in (a.plan,video/'manifest.json',Path(__file__),task/'tools/run/train_video_points.py',task/'tools/run/video_validation_panel.py')}
    cache = repo/'tmp/video-short-fit'; cache.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(cache),PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
               TRITON_CACHE_DIR=str(cache/'triton'),TORCH_HOME=str(cache/'torch'),HF_HOME=str(cache/'hf'))

    def persist():
        helper.record(a.output/'stage_status.json',state)

    def run_owned(command,log_path,seconds):
        if helper.gpu_processes([a.gpu]):
            raise RuntimeError('foreign GPU before launch')
        start = time.time()
        with log_path.open('w') as log:
            proc = subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(child_pid=proc.pid,command=command,phase_started_at=start); persist()
            try:
                while proc.poll() is None:
                    reason = None
                    scope_bytes = int(subprocess.check_output(['du','-sb',str(a.plan.parent)],text=True).split()[0])
                    run_bytes = int(subprocess.check_output(['du','-sb',str(a.output)],text=True).split()[0])
                    state.update(scope_bytes=scope_bytes,run_bytes=run_bytes,checked_at=time.time())
                    if scope_bytes >= state['scope_cap_bytes'] or run_bytes >= state['artifact_cap_bytes']: reason='storage cap'
                    if stop[0] or time.time()-start > seconds or time.time()-state['started_at'] > 1500: reason='user stop/deadline'
                    if any(helper.digest(Path(f)) != sha for f,sha in controls.items()): reason='control input/code drift'
                    if any(not helper.belongs_to(pid,proc.pid) for pid in helper.gpu_processes([a.gpu])): reason='foreign GPU'
                    validation = a.output/'train/validation.jsonl'
                    if validation.exists():
                        rows = [json.loads(line) for line in validation.read_text().splitlines()]
                        if len(rows)>1:
                            first,last = rows[0]['main'],rows[-1]['main']
                            if last['macro/trajectory_epe_m'] > first['macro/trajectory_epe_m']*1.10: reason='native hand >110%'
                            if last['macro/world_point_epe_m_h24'] > first['macro/world_point_epe_m_h24']*1.20: reason='native world >120%'
                    if reason:
                        state['stop_reason']=reason; persist()
                        os.killpg(proc.pid,signal.SIGUSR1)
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGTERM)
                            try:proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                        raise RuntimeError(reason)
                    snap = subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
                    state['gpu_snapshot']=snap
                    with (a.output/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),snapshot=snap))+'\n')
                    persist()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:pass
                return proc.returncode
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)

    common = [plan['train_python'],str(task/'tools/run/train_video_points.py'),'--data',plan['native_data'],'--stats',plan['stats'],
              '--video-data',str(video),'--init-wam',str(parent),'--main-batch','16','--video-batch','4','--seed','287',
              '--validation-indices',plan['validation_indices'],'--artifact-cap-bytes','1500000000']
    persist()
    try:
        if run_owned(common+['--smoke','--output',str(a.output/'smoke'),'--seconds','180'],a.output/'smoke.log',240) != 0:
            raise RuntimeError('mixed engineering smoke failed; see log')
        state.update(status='TRAINING');persist()
        cmd = common+['--output',str(a.output/'train'),'--steps','1000','--seconds','900','--eval-every','250','--checkpoint-every','10000']
        if run_owned(cmd,a.output/'train.log',1080) != 0:
            raise RuntimeError('mixed fit failed; see log')
        result = json.loads((a.output/'train/result.json').read_text())
        vals = [json.loads(line) for line in (a.output/'train/validation.jsonl').read_text().splitlines()]
        before,after = vals[0],vals[-1]; old,new = before['video'],after['video']
        improved = sum(new['sessions'][s]['point_epe_mm'] < v['point_epe_mm'] for s,v in old['sessions'].items())
        gates = dict(fixed_steps_complete=result['steps']==1000 and result['status']=='COMPLETED',
                     held_point_five_percent=new['point_epe_mm'] <= .95*min(old['point_epe_mm'],new['stationary_point_epe_mm']),
                     at_least_three_quarters_sessions_improve=improved/len(old['sessions']) >= .75,
                     native_hand_retained=after['main']['macro/trajectory_epe_m'] <= 1.05*before['main']['macro/trajectory_epe_m'],
                     native_world_retained=after['main']['macro/world_point_epe_m_h24'] <= 1.10*before['main']['macro/world_point_epe_m_h24'])
        helper.record(a.output/'decision.json',dict(conclusion='PROMISING' if all(gates.values()) else 'UNCLEAR',gates=gates,
            improved_sessions=improved,session_count=len(old['sessions']),before=before,after=after,
            limitation='Single fixed1000-step Probe. Numerical pseudo-labels; no physical accuracy, long-train or robot utility claim.'))
        state.update(status='COMPLETED',result_status=result['status'],decision_gates=gates)
    except Exception as exc:
        state.update(status='STOPPED' if state.get('stop_reason') else 'FAILED',error=repr(exc))
        raise
    finally:
        state.update(ended_at=time.time());persist()


if __name__ == '__main__':
    main()
