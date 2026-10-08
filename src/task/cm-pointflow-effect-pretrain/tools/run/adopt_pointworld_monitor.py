"""Restore supervision of an existing PointWorld job without relaunching it."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('existing_pointworld_launcher',
                                             Path(__file__).with_name('launch_pointworld_action_ddp.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def process_identity(pid):
    """Pin a live process, including start time to reject recycled PIDs."""
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = (path / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'uid': path.stat().st_uid,
                'command': [os.fsdecode(item) for item in command]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved):
    return saved['uid'] == os.getuid() and process_identity(saved['pid']) == saved


def request_save(workers, parent):
    """Signal only the pinned owned workers still below the pinned torchrun."""
    if not same_process(parent):
        return []
    signaled = []
    for worker in workers:
        if same_process(worker) and launcher.belongs_to(worker['pid'], parent['pid']):
            try:
                os.kill(worker['pid'], signal.SIGUSR1)
                signaled.append(worker['pid'])
            except ProcessLookupError:
                pass
    return signaled


def source_drift(sources):
    return [name for name, expected in sources.items()
            if not Path(name).is_file() or launcher.digest(Path(name)) != expected]


def adopt(run):
    run = run.resolve()
    run.relative_to((ROOT / 'outputs/cm-pointflow-effect-pretrain').resolve())
    state_path = run / 'group_status.json'
    original = json.loads(state_path.read_text())
    if process_identity(original['pid']) is not None:
        raise ValueError('existing supervisor is still live; refuse duplicate supervision')
    if not time.time() < original['deadline'] <= time.time() + 86400:
        raise ValueError('adoption must preserve a live authorized deadline')
    parent = process_identity(original['torchrun_pid'])
    if (parent is None or parent['uid'] != os.getuid()
            or parent['command'] != original['command']):
        raise ValueError('live owned torchrun identity does not match the original launch')
    packet = json.loads((run / 'train-action/input_manifest.json').read_text())
    workers = [process_identity(rank['pid']) for rank in packet['ranks']]
    trainer = str(Path(__file__).with_name('train_oakink2_pointworld_ddp.py').resolve())
    output = str(run / 'train-action')
    if (len(workers) != packet['world_size'] or len(workers) != len(original['gpus'])
            or any(worker is None or worker['uid'] != os.getuid()
                   or trainer not in worker['command'] or output not in worker['command']
                   or not launcher.belongs_to(worker['pid'], parent['pid']) for worker in workers)):
        raise ValueError('live owned training ranks do not match the original launch')
    drift = source_drift(original['sources'])
    if drift:
        raise ValueError('cannot adopt already drifted inputs: ' + ', '.join(drift))
    backup = run / 'group_status.before_adoption.json'
    with backup.open('x') as stream:
        stream.write(json.dumps(original, indent=2) + '\n')
    state = dict(original, pid=os.getpid(), original_supervisor_pid=original['pid'],
                 adopted_at=time.time(), monitor_script_sha256=launcher.digest(Path(__file__)),
                 monitor_git_commit=os.environ.get('POINTWORLD_MONITOR_COMMIT'),
                 pinned_parent=parent, pinned_workers=workers, supervision_recovered=True)
    user_stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: user_stop.__setitem__(0, True))
    stopping_at = None
    try:
        while same_process(parent):
            progress = json.loads((run / 'train-action/progress.json').read_text())
            state.update(status='TRAINING', progress=progress, checked_at=time.time())
            drift = source_drift(original['sources'])
            foreign = [pid for pid in launcher.gpu_processes(original['gpus'])
                       if not launcher.belongs_to(pid, parent['pid'])]
            if stopping_at is None and (drift or foreign or user_stop[0]
                                       or time.time() >= original['deadline']):
                state['stop_reason'] = ('source_drift' if drift else 'foreign_gpu_process' if foreign
                                        else 'user_stop' if user_stop[0] else 'deadline')
                state['drifted_sources'] = drift
                state['foreign_gpu_pids'] = foreign
                state['save_signaled_pids'] = request_save(workers, parent)
                stopping_at = time.time()
            if stopping_at is not None and time.time() - stopping_at > 180 and same_process(parent):
                os.kill(parent['pid'], signal.SIGTERM)
            launcher.record(state_path, state)
            time.sleep(30)
        progress = json.loads((run / 'train-action/progress.json').read_text())
        # An adopted process is not our child; do not invent an exit code.
        terminal = progress.get('status')
        state.update(status='COMPLETED' if terminal == 'COMPLETED' else
                     'STOPPED' if terminal == 'BUDGET_STOP' else 'FAILED',
                     progress=progress, ended_at=time.time(), exit_code=None,
                     completion_evidence='trainer progress/result; adopted process has exited')
        launcher.record(state_path, state)
    except BaseException as error:
        state.update(status='MONITOR_FAILED', error=repr(error), ended_at=time.time())
        launcher.record(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    adopt(args.run_dir)


if __name__ == '__main__':
    main()
