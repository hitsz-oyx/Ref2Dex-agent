"""At an owned checkpoint boundary, apply the verified runtime patch and resume."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
SOURCE = "src/task/ptv3-shared-wam/src/ptv3_shared_wam/training.py"


def write(path, value):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_matches(pid, marker):
    path = Path("/proc") / str(pid) / "cmdline"
    return path.exists() and marker.encode() in path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--new-output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2000)
    parser.add_argument("--base-ref", default="63bde79a")
    args = parser.parse_args()
    old, new = args.old_output.resolve(), args.new_output.resolve()
    owner = ROOT / "outputs/ptv3-shared-wam"
    if owner not in old.parents or owner not in new.parents or new.exists():
        raise ValueError("owned old run and fresh new run required")
    status_path = ROOT / "tmp" / old.name / "prefetch_migration.json"
    group = json.loads((old / "process.json").read_text())
    original = subprocess.check_output(["git", "show", args.base_ref + ":" + SOURCE], cwd=ROOT)
    patch = TASK / "tools/run/training-prefetch.patch"
    status = dict(status="WAITING_CHECKPOINT", old_output=str(old), new_output=str(new),
                  step=args.step, controller_pid=os.getpid(), old_torchrun_pid=group["torchrun_pid"],
                  old_supervisor_pid=group["supervisor_pid"])
    checkpoint = old / ("step%d.pt" % args.step)
    started = time.monotonic()
    try:
        while not checkpoint.exists():
            if time.monotonic() - started > 7200:
                raise TimeoutError("two-hour checkpoint wait budget")
            if not process_matches(group["torchrun_pid"], str(old)):
                raise RuntimeError("old torchrun exited before the requested checkpoint")
            progress = json.loads((old / "progress.json").read_text())
            status["old_step"] = progress["step"]
            write(status_path, status)
            time.sleep(20)
        if (ROOT / SOURCE).read_bytes() != original:
            raise RuntimeError("training source changed independently; refuse to overwrite")
        subprocess.run(["git", "apply", "--unidiff-zero", "--check", str(patch)], cwd=ROOT, check=True)
        sys.path.insert(0, str(TASK / "src"))
        from ptv3_shared_wam.checkpoint import load_checkpoint
        packet = load_checkpoint(checkpoint)
        if (packet["step"] != args.step or packet["stage"] != "pretrain"
                or packet["provenance"].get("engineering_only")
                or packet["provenance"]["world_size"] != 4):
            raise ValueError("matching scientific four-rank pretraining checkpoint required")
        status["status"] = "STOPPING_OWNED_GROUP"
        write(status_path, status)
        pid = group["torchrun_pid"]
        if not process_matches(pid, str(old)) or os.getpgid(pid) != pid:
            raise RuntimeError("torchrun process-group ownership no longer matches")
        os.killpg(pid, signal.SIGTERM)
        if process_matches(group["supervisor_pid"], str(old)):
            os.kill(group["supervisor_pid"], signal.SIGTERM)
        gpus = ",".join(group["physical_gpus"])
        deadline = time.monotonic() + 90
        while subprocess.check_output(["nvidia-smi", "-i", gpus, "--query-compute-apps=pid",
                                       "--format=csv,noheader"], text=True).strip():
            if time.monotonic() > deadline:
                raise TimeoutError("owned GPU contexts did not release; no foreign processes stopped")
            time.sleep(2)
        # Stop precedes any edit of the watched canonical source.
        subprocess.run(["git", "apply", "--unidiff-zero", str(patch)], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "--only", "-m",
                        "perf(ptv3-shared-wam): overlap CPU batch preparation with GPU training",
                        "--", SOURCE], cwd=ROOT, check=True)
        status["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        new.mkdir(parents=True)
        if (old / "curve.json").exists():
            shutil.copy2(old / "curve.json", new / "curve.json")
        old_elapsed = json.loads((old / "process.json").read_text()).get("elapsed_s", 0.)
        remaining = max(1, int(packet["config"]["max_seconds"] - old_elapsed - 60))
        del packet
        command = [sys.executable, str(TASK / "tools/run/launch_pretrain.py"), "--detach",
            "--gpus", gpus, "--config", group["config"], "--experiment-card", group["experiment_card"],
            "--output", str(new), "--resume", str(checkpoint), "--loader-workers", "4",
            "--loader-prefetch", "2", "--sequence-cache", "512", "--pin-memory",
            "--deadline-seconds", str(remaining)]
        receipt = subprocess.check_output(command, cwd=ROOT, text=True)
        status.update(status="RESUMING", receipt=json.loads(receipt), checkpoint=str(checkpoint))
        write(status_path, status)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            progress_path = new / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                if progress["step"] >= args.step + 40:
                    status.update(status="RUNNING_OPTIMIZED", progress=progress)
                    write(status_path, status)
                    return
            process = new / "process.json"
            if process.exists() and json.loads(process.read_text()).get("status") == "FAILED":
                raise RuntimeError("resumed group failed; original checkpoint is preserved")
            time.sleep(20)
        raise TimeoutError("resumed group did not reach40 updates within10 minutes")
    except BaseException as error:
        status.update(status="FAILED", error=repr(error))
        write(status_path, status)
        raise


if __name__ == "__main__":
    main()
