#!/usr/bin/env python3
"""Run the frozen four-rank full-batch preflight, then one bounded remote model."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[5]
TASK = REPO/'src/task/goal-point-wam'
helper_path = REPO/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py'
spec = importlib.util.spec_from_file_location('owned_remote_helpers', helper_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'stats', 'init-pw', 'validation-indices', 'output', 'scope-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--artifact-cap-bytes', type=int, default=5_000_000_000)
    args = p.parse_args()
    scope, output = args.scope_root.resolve(), args.output.resolve()
    if scope not in REPO.parents or scope not in output.parents:
        raise ValueError('isolated code and outputs must belong to the declared storage scope')
    if output.exists(): raise FileExistsError(output)
    if helpers.gpu_processes([0,1,2,3]): raise RuntimeError('remote GPUs are occupied')
    output.mkdir(parents=True)
    stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    config = json.loads(args.config.read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', TMPDIR=str(REPO/'tmp'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
               TRITON_CACHE_DIR=str(REPO/'tmp/triton-remote'))
    common = []
    for name in ('data','stats','init-pw','config','validation-indices'):
        common += ['--'+name, str(getattr(args,name.replace('-','_')).resolve())]
    smoke = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(TASK/'tools/run/train_distributed.py'), *common,
             '--output',str(output/'preflight'), '--smoke','--smoke-full-batch']
    train = [sys.executable,str(TASK/'tools/run/launch_distributed.py'), *common,
             '--gpus','0,1,2,3','--output',str(output/'training')]
    state = dict(pid=os.getpid(),status='PREFLIGHT',gpus=[0,1,2,3],
                 started_at=time.time(),microbatch=config['microbatch'],global_batch=4*config['microbatch'])
    for phase, command in [('PREFLIGHT',smoke),('TRAINING',train)]:
        if stop[0]: break
        with (output/(phase.lower()+'_console.log')).open('w') as log:
            proc = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status=phase,child_pid=proc.pid,command=command)
            stopped = None
            deadline = time.time()+(900 if phase=='PREFLIGHT' else config['group_seconds']+240)
            while proc.poll() is None:
                size = int(subprocess.check_output(['du','-sb',str(scope)],text=True).split()[0])
                foreign = [pid for pid in helpers.gpu_processes([0,1,2,3]) if not helpers.belongs_to(pid,proc.pid)]
                reason = ('user_stop' if stop[0] else 'artifact_cap' if size>=args.artifact_cap_bytes else
                          'foreign_gpu_process' if foreign else 'deadline' if time.time()>=deadline else None)
                state.update(artifact_bytes=size,checked_at=time.time())
                if reason and stopped is None:
                    state.update(stop_reason=reason)
                    stopped = time.time()
                    if phase=='TRAINING': os.kill(proc.pid,signal.SIGUSR1)
                    else:
                        manifest = output/'preflight/input_manifest.json'
                        if manifest.exists():
                            for rank in json.loads(manifest.read_text())['ranks']:
                                if helpers.belongs_to(rank['pid'],proc.pid):
                                    try: os.kill(rank['pid'],signal.SIGUSR1)
                                    except ProcessLookupError: pass
                        else: proc.terminate()
                if stopped and time.time()-stopped>240: proc.terminate()
                helpers.record(output/'remote_status.json',state)
                time.sleep(30)
            state.update(exit_code=proc.returncode)
            if proc.returncode or stopped:
                state.update(status='STOPPED' if stopped else 'FAILED',ended_at=time.time())
                helpers.record(output/'remote_status.json',state)
                return proc.returncode or 1
        if phase=='PREFLIGHT':
            result = json.loads((output/'preflight/result.json').read_text())
            if result['status']!='COMPLETED' or result['step']!=4 or len(set(result['rank_parameter_sha256']))!=1:
                raise RuntimeError('four-rank full-batch preflight failed')
            state['preflight_result'] = result
            # Keep numerical/identity evidence; these four-update engineering
            # checkpoints are reproducible and need not consume NAS budget.
            state['preflight_checkpoint_sha256'] = {}
            for name in ('latest.pt','best.pt','final.pt'):
                checkpoint = output/'preflight'/name
                if checkpoint.exists():
                    state['preflight_checkpoint_sha256'][name] = helpers.digest(checkpoint)
            helpers.record(output/'remote_status.json',state)
            for name in state['preflight_checkpoint_sha256']:
                (output/'preflight'/name).unlink()
    state.update(status='STOPPED' if stop[0] else 'COMPLETED',ended_at=time.time())
    helpers.record(output/'remote_status.json',state)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
