"""Independent, one-second storage floor for task-owned preparation sessions.

Receipt is fsynced before SIGINT. A five-second grace period is followed by
SIGSTOP for any surviving writer session: no automatic restart or deletion.
Only same-user session leaders with this exact run in argv are eligible.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time

FLOOR = 30 * 2**30
TOOLS = {'prepare_full_rigid_data.py', 'summarize_full_rigid_data.py',
         'audit_nine_source_readiness.py'}


def owned_sessions(run):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            if entry.stat().st_uid != os.getuid() or os.getsid(pid) != pid:
                continue
            if not any(Path(a).name in TOOLS for a in argv):
                continue
            if not any(a == str(run) or a == str(run.relative_to(Path.cwd())) for a in argv):
                continue
            found.append(pid)
        except (OSError, ValueError):
            continue
    return found


def durable(path, value):
    pending = path.with_suffix(path.suffix + '.part')
    with pending.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def stop(run, free, targets, reason='free_below_30GiB'):
    receipt = run / ('storage-stop-receipt-%d.json' % time.time_ns())
    durable(receipt, dict(status='STOP_REQUESTED', reason=reason,
        measured_at=datetime.now(timezone.utc).isoformat(), free_bytes=free,
        floor_bytes=FLOOR, sessions=targets, resumable=True,
        completed_batches_preserved=True, automatic_restart=False,
        signal_policy='SIGINT then SIGSTOP survivors after 5 seconds'))
    durable(run / 'STOP', dict(reason=reason, receipt=receipt.name))
    # Existing supervisor checks this terminal marker before launching summary.
    if not (run / 'stop-receipt.json').exists():
        durable(run / 'stop-receipt.json', dict(status='STOPPED', reason=reason,
            receipt=receipt.name, disk_free_bytes=free, resumable=True,
            training_allowed=False, automatic_restart=False))
    for pid in targets:
        try:
            os.killpg(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=21600)
    args = parser.parse_args()
    run = args.run.resolve()
    deadline = time.monotonic() + args.max_seconds
    while True:
        free = shutil.disk_usage(run).free
        targets = owned_sessions(run)
        reason = ('free_below_30GiB' if free < FLOOR else
                  'watcher_deadline' if time.monotonic() >= deadline else None)
        durable(run / 'storage-watch-status.json', dict(status='WATCHING',
            measured_at=datetime.now(timezone.utc).isoformat(), free_bytes=free,
            floor_bytes=FLOOR, sessions=targets, poll_seconds=1,
            automatic_restart=False, deadline_remaining_s=max(0, deadline-time.monotonic())))
        if reason:
            receipt = stop(run, free, targets, reason)
            time.sleep(5)
            survivors = []
            for pid in targets:
                if pid not in owned_sessions(run):
                    continue
                try:
                    os.killpg(pid, signal.SIGSTOP)
                    survivors.append(pid)
                except ProcessLookupError:
                    pass
            durable(run / 'storage-watch-status.json', dict(status='STOPPED',
                receipt=str(receipt), paused_survivors=survivors,
                automatic_restart=False))
            return
        time.sleep(1)


if __name__ == '__main__':
    main()
