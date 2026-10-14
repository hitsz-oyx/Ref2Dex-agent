#!/usr/bin/env python3
"""Own one bounded consumer and preserve external GPU/resource monitoring."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/"src"))
from video_data_preparation.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=3900)
    args = parser.parse_args()
    scope = args.scope.resolve()
    if not 60 <= args.seconds <= 7200 or scope == Path("/tmp") or Path("/tmp") in scope.parents:
        raise ValueError("owned scope and bounded deadline required")
    scope.mkdir(parents=True, exist_ok=True)
    if (scope/"supervisor.json").exists():
        raise FileExistsError("fresh supervisor scope required")
    config = json.loads(args.config.read_text())
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    child_script = TASK/"tools/run/prepare_egodex.py"
    command = [str(args.python), "-u", str(child_script), "--ready", str(args.ready),
               "--config", str(args.config), "--output", str(scope/"corpus"), "--gpu", str(args.gpu), "--once"]
    began = time.monotonic()
    reason, peak_mib = "CHILD_EXIT", 0
    with (scope/"consumer.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, TMPDIR=str(scope/"tmp"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
        identity = dict(supervisor_pid=os.getpid(), child_pid=child.pid, uid=os.getuid(), gpu=args.gpu,
                        command=command, git_commit=config["git_commit"], work_version=config["git_commit"],
                        started_at=time.time(), max_seconds=args.seconds)
        atomic_json(scope/"launch.json", identity, exclusive=True)
        try:
            while child.poll() is None:
                metrics = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                    "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
                peak_mib = max(peak_mib, int(metrics.split(",")[1].strip()))
                progress = scope/"corpus/progress.json"
                state = json.loads(progress.read_text()) if progress.exists() else {}
                row = dict(identity, status="RUNNING", elapsed_s=time.monotonic()-began,
                           gpu_metrics=metrics, peak_memory_mib=peak_mib, consumer=state)
                atomic_json(scope/"supervisor.json", row)
                with (scope/"resources.jsonl").open("a") as events:
                    events.write(json.dumps(row)+"\n")
                if stop[0] or (scope/"STOP").exists():
                    reason = "USER_STOP"
                    break
                if time.monotonic()-began >= args.seconds:
                    reason = "HARD_DEADLINE"
                    break
                if (scope/"consumer.log").stat().st_size > 100*2**20:
                    reason = "LOG_BUDGET"
                    break
                # Responsive signals and child completion without busy tool polling.
                for _ in range(30):
                    if child.poll() is not None or stop[0] or (scope/"STOP").exists():
                        break
                    time.sleep(1)
        finally:
            if child.poll() is None:
                # Popen created this session; no unknown PID/group is targeted.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            atomic_json(scope/"supervisor.json", dict(identity, status="COMPLETED" if child.returncode == 0 else "FAILED",
                reason=reason, returncode=child.returncode, elapsed_s=time.monotonic()-began,
                peak_memory_mib=peak_mib, ended_at=time.time()))


if __name__ == "__main__":
    main()
