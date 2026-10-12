#!/usr/bin/env python3
"""Guard one native EgoDex auxiliary run; never alter another GPU job."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]
HELPER = TASK.parent/'cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py'
spec = importlib.util.spec_from_file_location('egodex_owned_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'stats', 'ego-data', 'init-wam', 'config', 'validation-indices', 'output', 'runtime-tmp']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--scope-root', type=Path)
    p.add_argument('--artifact-cap-bytes', type=int, default=298_000_000_000)
    p.add_argument('--engineering-only', action='store_true')
    p.add_argument('--no-checkpoint', action='store_true')
    a = p.parse_args()
    root, temp = a.output.resolve(), a.runtime_tmp.resolve()
    config = json.loads(a.config.read_text())
    gpus = list(map(int, a.gpus.split(',')))
    if len(gpus) != config['world_size'] or len(set(gpus)) != len(gpus) or not 1 <= len(gpus) <= 4:
        raise ValueError('world size and assigned GPU set mismatch')
    if helpers.gpu_processes(gpus):
        raise RuntimeError('assigned GPU has existing processes')
    if len(os.fsencode(str(temp))) > 75:
        raise ValueError('runtime tmp too long for DataLoader socket')
    if a.scope_root:
        scope = a.scope_root.resolve()
        if scope not in root.parents or scope not in temp.parents or scope not in REPO.parents:
            raise ValueError('NAS code/outputs/runtime must belong to declared scope')
    elif REPO not in root.parents or REPO not in temp.parents:
        raise ValueError('local outputs/runtime must belong to repo')
    if root.exists():
        raise FileExistsError('new launcher output required')
    root.mkdir(parents=True); temp.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), HELPER, TASK/'tools/run/train_egodex_auxiliary.py',
             TASK/'tools/run/train_three_domain.py', a.config, a.stats, a.data/'processed/manifest.json',
             a.ego_data/'manifest.json', a.ego_data/'index_train.npy', a.ego_data/'index_val.npy',
             a.validation_indices, a.validation_indices.parent/'panel.json']
    files += list((TASK/'src/goal_point_wam').glob('*.py'))
    sources = {str(f.resolve()):helpers.digest(f) for f in files}
    command = [sys.executable]
    if len(gpus) > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node='+str(len(gpus))]
    command += [str(TASK/'tools/run/train_egodex_auxiliary.py')]
    for name in ['data', 'stats', 'ego-data', 'init-wam', 'config', 'validation-indices']:
        command += ['--'+name, str(getattr(a, name.replace('-', '_')).resolve())]
    command += ['--output', str(root/'train')]
    if a.engineering_only:
        command += ['--engineering-only']
    if a.no_checkpoint:
        command += ['--no-checkpoint']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, TMPDIR=str(temp),
               TRITON_CACHE_DIR=str(REPO/'tmp/triton-ego'), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    stop = [False]
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_:stop.__setitem__(0, True))
    start = time.time()
    with (root/'console.log').open('w') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state = dict(status='RUNNING', pid=os.getpid(), child_pid=proc.pid, gpus=gpus, command=command,
                     git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                     sources=sources, started_at=start, deadline=start+config['group_seconds'],
                     artifact_cap_bytes=a.artifact_cap_bytes)
        stopping = None
        while proc.poll() is None:
            foreign = [pid for pid in helpers.gpu_processes(gpus) if not helpers.belongs_to(pid, proc.pid)]
            drift = any(not Path(f).exists() or helpers.digest(Path(f)) != h for f,h in sources.items())
            roots = [str(a.scope_root.resolve())] if a.scope_root else [str(REPO/'outputs'), str(REPO/'tmp')]
            size = sum(int(x.split()[0]) for x in subprocess.check_output(['du','-sb',*roots],text=True).splitlines())
            reason = ('user_stop' if stop[0] else 'source_drift' if drift else 'foreign_gpu' if foreign else
                      'disk_cap' if size >= a.artifact_cap_bytes else 'deadline' if time.time() >= state['deadline'] else None)
            if reason and stopping is None:
                stopping = time.time(); state.update(status='STOP_REQUESTED', stop_reason=reason)
                identity = root/'train/input_manifest.json'
                if identity.exists():
                    for rank in json.loads(identity.read_text())['ranks']:
                        if helpers.belongs_to(rank['pid'], proc.pid):
                            try:
                                os.kill(rank['pid'], signal.SIGUSR1)
                            except ProcessLookupError:
                                pass
                else:
                    proc.terminate()
            if stopping and time.time()-stopping > 240:
                proc.terminate()
            state.update(artifact_bytes=size, checked_at=time.time())
            progress = root/'train/progress.json'
            if progress.exists():
                state['progress'] = json.loads(progress.read_text())
            helpers.record(root/'group_status.json', state)
            time.sleep(10 if a.engineering_only else 30)
        state.update(status='STOPPED' if stopping else 'COMPLETED' if proc.returncode == 0 else 'FAILED',
                     exit_code=proc.returncode, ended_at=time.time())
        helpers.record(root/'group_status.json', state)
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
