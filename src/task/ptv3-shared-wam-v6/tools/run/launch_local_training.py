"""Save/stop the identified PPO, check actual WAM recovery, supervise local adaptation."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.local_subset import initialize_weights, validate_local_execution
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.recipe import validate_recipe


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    pending = path.with_suffix('.json.part')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def alive(pid):
    path = Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z'


def verify_owned_process(pid, marker):
    path = Path('/proc')/str(pid)
    if path.stat().st_uid != os.getuid() or marker not in (path/'cmdline').read_bytes().replace(b'\0', b' ').decode():
        raise ValueError('PPO process ownership/run identity mismatch: '+str(pid))


def gpu_state():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    return {int(p[0]): dict(memory_mib=int(p[1]), utilization=int(p[2]))
            for p in (line.split(',') for line in text.splitlines())}


def released(gpus):
    state = gpu_state()
    if any(state[g]['memory_mib'] > 256 for g in gpus):
        raise ValueError('requested/reserved GPUs are not released: '+str(state))
    return state


def audit_checkpoint(path, expected, initialization=None):
    packet = torch.load(path, map_location='cpu', weights_only=False)
    if packet.get('schema') != 'ref2dex.ptv3-shared-wam.v6' or packet['step'] != expected:
        raise ValueError('unexpected local checkpoint schema/step')
    if (packet['world_size'] != 1 or packet['execution_contract']['effective_global_batch'] != 64
            or not packet['initialization_binding'] or not packet['optimizer']['state']):
        raise ValueError('incomplete local weights/batch/optimizer binding')
    def finite(value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('nonfinite checkpoint tensor')
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value: finite(child)
    finite(packet['model']); finite(packet['optimizer'])
    steps = sorted(set(int(state['step']) for state in packet['optimizer']['state'].values()))
    if steps != [expected] or len(packet['random_state']) != 1:
        raise ValueError('optimizer/RNG did not recover correctly: '+str(steps))
    if initialization is not None:
        for key in ('hand_mean', 'hand_std', 'scene_mean', 'scene_std'):
            if not torch.equal(packet['model']['normalizer.'+key], initialization[key]):
                raise ValueError('normalizer changed during training')
    result = dict(path=str(path), step=expected, optimizer_states=len(packet['optimizer']['state']),
                  optimizer_steps=steps, bytes=path.stat().st_size, all_finite=True,
                  initialization_binding=packet['initialization_binding'])
    del packet; gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe-root', 'checkpoint', 'ppo-output', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False)
    out = a.output.resolve(); recipe_root = a.recipe_root.resolve(); ppo = a.ppo_output.resolve()
    state = dict(status='PREFLIGHT', supervisor_pid=os.getpid(), git_commit=a.git_commit,
                 run_id=out.name, work_version='v6.local-subset-supervisor.v1', physical_gpus=[4],
                 reserved_idle_gpus=[5, 6], stages=[], started_at=time.time())
    def publish(): write(out/'process.json', state)
    publish(); torch.set_num_threads(2)
    if read(recipe_root/'preflight.json')['status'] != 'PASS':
        raise ValueError('complete local data preflight required before PPO stop')
    recipe = validate_recipe(recipe_root/'manifest.json', recipe_root/'stats.json')
    config = read(recipe_root/'config.json')
    validate_local_execution(config, recipe, 1, config['updates'], config['batch_size'])
    model = SharedWAMV6(read(recipe_root/'stats.json'), config)
    packet = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    state['initialization'] = initialize_weights(model, packet, a.checkpoint, recipe)
    normalizer = {key: value.cpu().clone() for key, value in model.normalizer.state_dict().items()}
    del model, packet; gc.collect()
    if shutil.disk_usage(out).free < 35*2**30:
        raise ValueError('insufficient disk peak headroom before switch')
    metadata = read(ppo/'manifest.json')
    if (metadata['run_id'] != 'inspire-measured-official-20261020-r3'
            or metadata['status'] not in ('RUNNING', 'STOPPED') or metadata['gpus'] != [4, 5, 6]):
        raise ValueError('only the user-authorized active PPO r3 may be stopped')
    marker = 'inspire-measured-official-20261020-r3'
    pids = [int(metadata['pid'])]+[int(row['pid']) for row in metadata['stages']]
    for pid in pids:
        if alive(pid): verify_owned_process(pid, marker)
    state.update(status='STOPPING_PPO', ppo_output=str(ppo), ppo_pids=pids,
                 ppo_before=[read(ppo/('rank-'+str(r))/'progress.json') for r in range(3)],
                 existing_stop_requested=(ppo/'STOP').exists())
    publish(); (ppo/'STOP').touch(exist_ok=True)
    deadline = time.monotonic()+300
    while any(alive(pid) for pid in pids):
        if time.monotonic() > deadline:
            raise TimeoutError('PPO graceful stop timed out; no unknown process killed')
        time.sleep(5)
    ranks = [read(ppo/('rank-'+str(r))/'manifest.json') for r in range(3)]
    if any(row['status'] != 'STOPPED' for row in ranks):
        raise ValueError('PPO did not stop normally')
    iteration = ranks[0]['iteration']
    if any(row['iteration'] != iteration for row in ranks):
        raise ValueError('PPO ranks stopped at different iterations')
    checkpoint = ppo/'rank-0'/('official-iteration-%d.pt' % iteration)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['iteration'] != iteration or not saved['optimizer']['state']:
        raise ValueError('PPO final checkpoint incomplete')
    for section in ('model', 'optimizer'):
        def finite(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('nonfinite PPO final checkpoint')
            if isinstance(value, dict):
                for item in value.values(): finite(item)
            elif isinstance(value, (list, tuple)):
                for item in value: finite(item)
        finite(saved[section])
    del saved; gc.collect()
    state['ppo_stop'] = dict(status='VERIFIED', iteration=iteration, checkpoint=str(checkpoint),
                             bytes=checkpoint.stat().st_size, worker_statuses=[r['status'] for r in ranks])
    released([4, 5, 6]); publish()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(out), TRITON_CACHE_DIR=str(out/'triton'))
    child = None; current = None; requested = []
    signal.signal(signal.SIGTERM, lambda *_: requested.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: requested.append('SIGINT'))
    try:
        source = None
        for name, extra, target in (
                ('readiness', ['--steps', '8', '--initialize-from', str(a.checkpoint.resolve())], 8),
                ('recovery', ['--recovery-probe-updates', '2'], 10),
                ('training', [], 2000)):
            released([4, 5, 6])
            current = out/name
            command = [sys.executable, '-u', str(Path(__file__).with_name('train_pretrain.py')),
                       '--config', str(recipe_root/'config.json'), '--manifest', str(recipe_root/'manifest.json'),
                       '--stats', str(recipe_root/'stats.json'), '--output', str(current),
                       '--workers', '4', '--git-commit', a.git_commit]+extra
            if source: command += ['--resume', str(source)]
            record = dict(name=name, command=command, output=str(current), started_at=time.time())
            state.update(status='RUNNING', phase=name); state['stages'].append(record)
            with (out/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                record['pid'] = child.pid; publish(); stop_deadline = None; last_step = -1
                last_progress = time.monotonic(); stage_start = last_progress
                while child.poll() is None:
                    gpu = gpu_state()
                    reason = requested or ((out/'STOP').exists() and 'owned STOP')
                    if shutil.disk_usage(out).free < 20*2**30: reason = 'disk floor'
                    if any(gpu[g]['memory_mib'] > 256 for g in [5, 6]): reason = 'reserved GPU occupied'
                    if time.monotonic()-stage_start > (1800 if name != 'training' else 14700): reason = 'stage time budget'
                    progress = current/'progress.json'
                    if progress.exists():
                        state['progress'] = read(progress)
                        if state['progress']['step'] != last_step:
                            last_step = state['progress']['step']; last_progress = time.monotonic()
                    if time.monotonic()-last_progress > 900: reason = 'no progress for15min'
                    if reason and stop_deadline is None:
                        current.mkdir(exist_ok=True); (current/'STOP').touch()
                        stop_deadline = time.monotonic()+180
                        state.update(status='STOPPING', stop_reason=reason)
                    if stop_deadline and time.monotonic() > stop_deadline:
                        raise TimeoutError('owned WAM worker did not honor STOP')
                    state.update(gpu=gpu, free_gib=shutil.disk_usage(out).free/2**30,
                                 elapsed_s=time.time()-state['started_at'])
                    publish()
                    with (out/'gpu.jsonl').open('a') as log_gpu:
                        log_gpu.write(json.dumps(dict(time=time.time(), phase=name, gpus=gpu))+'\n')
                    time.sleep(15)
            record['exit_code'] = child.returncode
            if child.returncode:
                raise RuntimeError('WAM '+name+' failed; inspect '+str(out/(name+'.log')))
            result = read(current/'result.json'); record['result'] = result
            if result['status'] != 'COMPLETED':
                state.update(status=result['status']); publish(); return
            source = current/('step%06d.pt' % target)
            record['checkpoint_audit'] = audit_checkpoint(source, target, normalizer)
            publish()
        state.update(status='COMPLETED'); publish()
    except BaseException as error:
        state.update(status='FAILED', error=repr(error)); publish(); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()


if __name__ == '__main__':
    main()
