#!/usr/bin/env python3
"""Supervise a fixed-step three-rank mixed fit, with an enlarged-batch smoke."""
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
    for key in ('plan', 'expansion', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs=3, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--main-batch', type=int, default=128)
    p.add_argument('--video-batch', type=int, default=32)
    p.add_argument('--seconds', type=int, default=172800)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[5]
    task = repo/'src/task/goal-point-wam'
    spec = importlib.util.spec_from_file_location('owned_helpers', repo/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    plan = json.loads(a.plan.read_text())
    expansion = json.loads((a.expansion/'stage_status.json').read_text())
    if expansion['status'] != 'COMPLETED' or not expansion['training_allowed']:
        raise ValueError('completed qualified expansion required')
    if (a.output.exists() or len(set(a.gpus)) != 3 or helper.gpu_processes(a.gpus)
            or a.steps < 4 or min(a.main_batch, a.video_batch, a.seconds) <= 0):
        raise ValueError('new bounded output and three idle distinct GPUs required')
    video = Path(expansion['qualified'])
    parent_root = Path(plan['original_training'])
    parent = parent_root/'train/final.pt'
    if json.loads((parent_root/'group_status.json').read_text())['status'] != 'COMPLETED':
        raise ValueError('completed native parent required')
    a.output.mkdir(parents=True)
    state = dict(status='SMOKE', pid=os.getpid(), run_id=a.output.name,
                 started_at=time.time(), assigned_gpus=a.gpus, planned_steps=a.steps,
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
                 train_windows=expansion['train_windows'], val_windows=expansion['val_windows'],
                 artifact_cap_bytes=4_000_000_000, scope_cap_bytes=25_000_000_000,
                 parent=str(parent), parent_step=10000, resumed_from=str(a.resume) if a.resume else None,
                 semantic_early_stopping=False, main_per_rank_batch=a.main_batch, video_per_rank_batch=a.video_batch)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    cache = repo/'tmp/video-long-fit'; cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,a.gpus)), TMPDIR=str(cache),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='2', TRITON_CACHE_DIR=str(cache/'triton'), TORCH_HOME=str(cache/'torch'),
               HF_HOME=str(cache/'hf'), TORCH_NCCL_ASYNC_ERROR_HANDLING='1')

    def persist():
        helper.record(a.output/'stage_status.json', state)

    def graceful_stop(proc, child_output):
        manifest = child_output/'input_manifest.json'
        if manifest.exists():
            for row in json.loads(manifest.read_text())['ranks']:
                if helper.belongs_to(row['pid'], proc.pid):
                    try: os.kill(row['pid'], signal.SIGUSR1)
                    except ProcessLookupError: pass
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)

    def run_owned(command, child_output, seconds):
        if helper.gpu_processes(a.gpus):
            raise RuntimeError('assigned GPU occupied before launch')
        started = time.time()
        with (a.output/(child_output.name+'.log')).open('w') as log:
            proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            state.update(child_pid=proc.pid, command=command, phase_started_at=started); persist()
            try:
                while proc.poll() is None:
                    reason = None
                    if stop[0] or time.time()-started > seconds: reason = 'user stop or wall-time bound'
                    if any(not helper.belongs_to(pid,proc.pid) for pid in helper.gpu_processes(a.gpus)):
                        reason = 'foreign process on assigned GPU'
                    # Metadata accounting only, once per monitor interval; no repeated data hashes.
                    run_bytes = int(subprocess.check_output(['du','-sb',str(a.output)],text=True).split()[0])
                    if run_bytes >= state['artifact_cap_bytes']: reason = 'artifact budget'
                    state.update(run_bytes=run_bytes, checked_at=time.time())
                    progress = child_output/'progress.json'
                    if progress.exists(): state['progress'] = json.loads(progress.read_text())
                    snap = subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
                    state['gpu_snapshot'] = snap
                    with (a.output/'gpu_samples.jsonl').open('a') as f:
                        f.write(json.dumps(dict(time=time.time(), snapshot=snap))+'\n')
                    persist()
                    if reason:
                        state['stop_reason'] = reason; persist(); graceful_stop(proc,child_output)
                        raise RuntimeError(reason)
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: pass
                return proc.returncode
            finally:
                if proc.poll() is None: graceful_stop(proc,child_output)

    def command(output, main_batch, video_batch, smoke=False):
        cmd = [plan['train_python'],'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',
               str(task/'tools/run/train_video_points.py'),'--data',plan['native_data'],'--stats',plan['stats'],
               '--video-data',str(video),'--init-wam',str(parent),'--output',str(output),
               '--main-batch',str(main_batch),'--video-batch',str(video_batch),'--seed','287',
               '--validation-indices',plan['validation_indices'],'--steps',str(a.steps),
               '--seconds',str(600 if smoke else a.seconds),'--artifact-cap-bytes','4000000000',
               '--eval-every','1000','--checkpoint-every','1000','--telemetry-every','50']
        if smoke: cmd += ['--smoke']
        elif a.resume: cmd += ['--resume',str(a.resume)]
        return cmd

    persist()
    try:
        main_batch, video_batch = a.main_batch, a.video_batch
        smoke = a.output/'smoke'
        code = run_owned(command(smoke,main_batch,video_batch,True),smoke,720)
        if code != 0:
            if a.resume or 'out of memory' not in (a.output/'smoke.log').read_text().lower():
                raise RuntimeError('three-rank smoke failed; see smoke.log')
            main_batch, video_batch = max(1,main_batch//2), max(1,video_batch//2)
            state.update(batch_fallback='OOM: halve both per-rank batches'); persist()
            smoke = a.output/'smoke-half'
            if run_owned(command(smoke,main_batch,video_batch,True),smoke,720) != 0:
                raise RuntimeError('smoke at halved batch failed')
        state.update(status='TRAINING', main_per_rank_batch=main_batch, video_per_rank_batch=video_batch,
                     main_global_batch=3*main_batch, video_global_batch=3*video_batch,
                     smoke_result=json.loads((smoke/'result.json').read_text())); persist()
        train = a.output/'train'
        if run_owned(command(train,main_batch,video_batch),train,a.seconds+900) != 0:
            raise RuntimeError('distributed fit failed; see train.log; existing checkpoints preserved')
        result = json.loads((train/'result.json').read_text())
        state.update(status=result['status'], completed_steps=result['steps'])
    except Exception as exc:
        state.update(status='STOPPED' if state.get('stop_reason') else 'FAILED', error=repr(exc))
        raise
    finally:
        state.update(ended_at=time.time()); persist()


if __name__ == '__main__':
    main()
