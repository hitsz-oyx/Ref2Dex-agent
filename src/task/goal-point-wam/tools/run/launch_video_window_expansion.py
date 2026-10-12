#!/usr/bin/env python3
"""Bounded per-record whole geometry and short-window extraction on idle GPUs.

Raw acquisitions and old caches are read-only. A new qualified symlink view
contains only windows passing every existing quality gate, with frozen splits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--reuse-windows', type=Path)
    p.add_argument('--reuse-expansion', type=Path, help='Reuse completed records from an earlier owned batch')
    p.add_argument('--seconds', type=int, default=2400)
    p.add_argument('--scope-cap-bytes', type=int, default=25_000_000_000)
    p.add_argument('--artifact-cap-bytes', type=int, default=10_000_000_000)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[5]
    task = repo/'src/task/goal-point-wam'
    spec = importlib.util.spec_from_file_location('owned_helpers', repo/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    if a.output.exists() or not 1 <= len(a.gpus) <= 4 or len(set(a.gpus)) != len(a.gpus) or not 60 <= a.seconds <= 3600:
        raise ValueError('new bounded output and one to four unique GPUs required')
    if helper.gpu_processes(a.gpus):
        raise RuntimeError('assigned GPUs occupied')
    plan = json.loads(a.plan.read_text())
    records = plan['records']
    if len(records) > 24 or len({r['id'] for r in records}) != len(records):
        raise ValueError('bounded unique frozen recordings required')
    a.output.mkdir(parents=True)
    scope = a.plan.parent
    assets = Path(plan['depth_checkpoint']).parents[1]
    fixed = {str(f): helper.digest(f) for f in (a.plan, Path(__file__), task/'tools/audit/probe_long_video_depth.py', task/'tools/audit/probe_window_video_tracks.py', task/'tools/audit/audit_video_tracks.py', task/'tools/audit/trajectory_geometry.py')}
    state = dict(status='RUNNING', pid=os.getpid(), started_at=time.time(), run_id=a.output.name,
                 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                 assigned_gpus=a.gpus, input_digests=fixed, scope_cap_bytes=a.scope_cap_bytes,
                 artifact_cap_bytes=a.artifact_cap_bytes, records=[], workers={})
    stop = threading.Event(); lock = threading.Lock()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.set())

    def persist():
        with lock:
            helper.record(a.output/'stage_status.json', state)

    def safety():
        if stop.is_set() or time.time()-state['started_at'] > a.seconds:
            raise RuntimeError('user stop or expansion deadline')
        if any(helper.digest(Path(f)) != sha for f, sha in fixed.items()):
            raise RuntimeError('input/code drift')
        scope_bytes = int(subprocess.check_output(['du','-sb',str(scope)],text=True).split()[0])
        run_bytes = int(subprocess.check_output(['du','-sb',str(a.output)],text=True).split()[0])
        with lock:
            state.update(scope_bytes=scope_bytes, run_bytes=run_bytes, checked_at=time.time())
        if scope_bytes >= a.scope_cap_bytes or run_bytes >= a.artifact_cap_bytes:
            raise RuntimeError('storage envelope exhausted')

    def run_owned(command, log_path, gpu, seconds):
        safety()
        if helper.gpu_processes([gpu]):
            stop.set(); raise RuntimeError('foreign GPU before launch')
        cache = repo/'tmp'/('video-expansion-gpu%d'%gpu); cache.mkdir(parents=True,exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(cache), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
                   TRITON_CACHE_DIR=str(cache/'triton'), TORCH_HOME=str(cache/'torch'), HF_HOME=str(cache/'hf'))
        start = time.time()
        with log_path.open('w') as log:
            proc = subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:
                state['workers'][str(gpu)] = dict(child_pid=proc.pid, command=command, started_at=start)
            persist()
            try:
                while proc.poll() is None:
                    try:
                        safety()
                        if time.time()-start > seconds:
                            raise RuntimeError('per-stage deadline')
                        if any(not helper.belongs_to(pid,proc.pid) for pid in helper.gpu_processes([gpu])):
                            raise RuntimeError('foreign GPU during run')
                    except Exception:
                        stop.set(); raise
                    snapshot = subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
                    with lock:
                        state['gpu_snapshot'] = snapshot
                        with (a.output/'gpu_samples.jsonl').open('a') as f:
                            f.write(json.dumps(dict(time=time.time(),gpu=gpu,child_pid=proc.pid,snapshot=snapshot))+'\n')
                    persist()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: pass
                return proc.returncode
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=10)

    reused = {}
    if a.reuse_expansion:
        previous = json.loads((a.reuse_expansion/'stage_status.json').read_text())
        if previous['status'] not in ('STOPPED','COMPLETED') or previous['input_digests'][str(a.plan)] != helper.digest(a.plan):
            raise ValueError('prior batch must be stopped/completed with the same frozen plan')
        allowed_ids = {r['id'] for r in records}
        for receipt in previous['records']:
            if receipt['status'] == 'COMPLETED' and receipt['id'] in allowed_ids:
                reused[receipt['id']] = Path(receipt['windows'])
    if a.reuse_windows:
        meta = json.loads((a.reuse_windows/'manifest.json').read_text())
        report = json.loads((a.reuse_windows/'result.json').read_text())
        if report['status'] != 'COMPLETED' or helper.digest(a.reuse_windows/'manifest.json') != report['manifest_sha256']:
            raise ValueError('reused extraction incomplete or changed')
        identity = meta['records'][0]['raw_hdf5_sha256']
        matches = [r for r in records if helper.digest(Path(r['raw_hdf5'])) == identity]
        if len(matches) != 1:
            raise ValueError('reused recording absent/ambiguous in frozen plan')
        reused[matches[0]['id']] = a.reuse_windows
    jobs = queue.Queue()
    for r in records: jobs.put(r)

    def worker(gpu):
        while not jobs.empty() and not stop.is_set():
            try: r = jobs.get_nowait()
            except queue.Empty: break
            folder = a.output/('record%02d'%r['id']); folder.mkdir()
            receipt = dict(id=r['id'], member=r['member'], split=r['split'], session=r['session'], gpu=gpu, status='RUNNING')
            try:
                windows = reused.get(r['id'])
                if windows is None:
                    depth = folder/'geometry'
                    cmd = [plan['prepare_python'],str(task/'tools/audit/probe_long_video_depth.py'), '--plan',str(a.plan), '--member',r['member'], '--output',str(depth), '--seconds','300']
                    if run_owned(cmd,folder/'depth.log',gpu,360) != 0:
                        raise ValueError('depth failed; see preserved depth.log')
                    dm = json.loads((depth/'result.json').read_text())
                    receipt.update(geometry=str(depth/'depth.npz'), calibration=dm['calibration'])
                    if not dm['calibration_screen_pass']:
                        raise ValueError('whole-record scale gate failed')
                    windows = folder/'windows'
                    cmd = [plan['prepare_python'],str(task/'tools/audit/probe_window_video_tracks.py'), '--geometry',str(depth/'depth.npz'), '--tracker-vendor',str(assets/'vendor/co-tracker'), '--tracker-checkpoint',str(assets/'cotracker3-direct/scaled_offline.pth'), '--output',str(windows), '--all-windows','--seconds','600']
                    if run_owned(cmd,folder/'windows.log',gpu,660) != 0:
                        raise ValueError('window extraction failed; see preserved windows.log')
                report = json.loads((windows/'result.json').read_text())
                receipt.update(status='COMPLETED', windows=str(windows), reused=r['id'] in reused,
                               total=report['loader_windows'], admitted=report['admission_pass'])
            except Exception as exc:
                receipt.update(status='FAILED', error=repr(exc))
            finally:
                helper.record(folder/'receipt.json',receipt)
                with lock: state['records'].append(receipt)
                persist(); jobs.task_done()

    persist()
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures = [pool.submit(worker,gpu) for gpu in a.gpus]
        for future in futures: future.result()
    if stop.is_set():
        state.update(status='STOPPED',ended_at=time.time()); persist(); return
    safety()
    view = a.output/'qualified'; (view/'clips').mkdir(parents=True)
    admitted, excluded, calibrations = [], [], []
    for receipt in sorted(state['records'],key=lambda r:r['id']):
        if receipt['status'] != 'COMPLETED':
            excluded.append(receipt); continue
        windows = Path(receipt['windows'])
        meta = json.loads((windows/'manifest.json').read_text())
        report = json.loads((windows/'result.json').read_text())
        quality = json.loads((windows/'quality.json').read_text())
        if helper.digest(windows/'manifest.json') != report['manifest_sha256'] or len(quality['records']) != len(meta['records']):
            raise ValueError('completed extraction drift/quality count mismatch')
        calibrations.append(meta['calibration'])
        for record, q in zip(meta['records'],quality['records']):
            if not q['admission_screen_pass']:
                excluded.append(dict(id=receipt['id'],path=record['path'],gates=q['admission_screen'])); continue
            source = (windows/record['path']).resolve()
            if helper.digest(source) != record['sha256'] or record['split'] != receipt['split'] or record['session'].removesuffix('.mov') != receipt['session'].removesuffix('.mov'):
                raise ValueError('clip identity/frozen split drift')
            target = view/'clips'/('r%02d-'%receipt['id']+source.parent.name+'.npz')
            target.symlink_to(source)
            admitted.append(dict(record,path=str(target.relative_to(view)),session=receipt['session'].removesuffix('.mov'), recording_id=receipt['id']))
    sessions = {s:sorted({r['session'] for r in admitted if r['split']==s}) for s in ('train','val')}
    if set(sessions['train']) & set(sessions['val']):
        raise ValueError('normalized session leakage')
    eligible = len(sessions['train']) >= 6 and len(sessions['val']) >= 3
    manifest = dict(schema='goal-point-wam.video-points.v1',status='COMPLETED',fps=30,history=4,horizon=24,
        engineering_only=not eligible,training_allowed=eligible,causal_observation=False,
        observation_policy='offline_clip_reconstruction',offline_pretraining_only=True,label_depth_backend='offline-teacher',
        records=admitted,label_calibrations=calibrations,session_split=sessions,
        quality_policy='All original seven numerical gates; frozen recording/session split. Not physical ground truth.',
        input_digests=fixed,geometry_context='whole_recording',excluded=excluded)
    helper.record(view/'manifest.json',manifest)
    state.update(status='COMPLETED', ended_at=time.time(), qualified=str(view), qualified_manifest_sha256=helper.digest(view/'manifest.json'),
                 training_allowed=eligible, sessions=sessions, train_windows=sum(r['split']=='train' for r in admitted),
                 val_windows=sum(r['split']=='val' for r in admitted), exclusions=len(excluded))
    persist()


if __name__ == '__main__':
    main()
