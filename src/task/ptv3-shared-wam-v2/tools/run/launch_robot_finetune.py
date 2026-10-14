"""Launch and supervise one owned v2 robot fine-tuning process group.

The launcher is deliberately separate from the v2 pretraining launcher: it
passes the RL660 source and an immutable remote pretraining parent to
``train_robot.py`` and never reuses a pretraining output.  The launcher only
terminates the process group it created, records a machine-readable status,
and checks the selected GPUs, progress, disk headroom, and this run's artifact
budget at a bounded polling interval.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
V2_OUTPUTS = ROOT / "outputs" / "ptv3-shared-wam-v2"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_gpus(value):
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not 1 <= len(gpus) <= 6 or len(set(gpus)) != len(gpus):
        raise ValueError("one to six distinct physical GPU indices required")
    if not all(item.isdigit() for item in gpus):
        raise ValueError("GPU indices must be physical decimal indices")
    return gpus


def check_gpu_free(gpus):
    for gpu in gpus:
        rows = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
             "--format=csv,noheader"], text=True).strip()
        if rows:
            raise RuntimeError("assigned GPU occupied: " + gpu)


def gpu_snapshot(gpus):
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-i", ",".join(gpus),
             "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return "ERROR " + repr(error)


def owned_bytes(paths):
    total = 0
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            total += root.stat().st_size
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def set_nofile(limit):
    try:
        current = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(current[0], limit)
        hard = max(current[1], target)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError):
        # The training code remains valid on hosts that disallow raising this
        # child-only limit; the result is recorded by the launcher command.
        pass


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--init-shared", type=Path, required=True)
    p.add_argument("--experiment-card", type=Path, required=True)
    p.add_argument("--gpus", default="4,6")
    p.add_argument("--resume", type=Path)
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--loader-prefetch", type=int, default=2)
    p.add_argument("--sequence-cache", type=int, default=16)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--deadline-seconds", type=int)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--stall-seconds", type=int, default=1800)
    p.add_argument("--min-free-bytes", type=int, default=20 * 1024**3)
    p.add_argument("--artifact-budget-bytes", type=int, default=8_000_000_000)
    p.add_argument("--nofile-limit", type=int, default=65536)
    p.add_argument("--detach", action="store_true",
                   help="detach this owned supervisor and return immediately")
    return p


def validate_args(args, gpus):
    out = args.output.resolve()
    if out.exists() or V2_OUTPUTS not in out.parents:
        raise ValueError("fresh v2 Task output required")
    card = args.experiment_card.resolve()
    if TASK / "docs" / "experiments" not in card.parents or not card.is_file():
        raise ValueError("existing Task-local experiment card required")
    for path in (args.config, args.stats, args.source, args.init_shared):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.resume and not args.resume.exists():
        raise FileNotFoundError(args.resume)
    if args.poll_seconds < 1 or args.poll_seconds > 1200:
        raise ValueError("poll interval must be between 1 and 1200 seconds")
    if args.stall_seconds < args.poll_seconds:
        raise ValueError("stall timeout must be at least one poll interval")
    if args.deadline_seconds is not None and args.deadline_seconds <= 0:
        raise ValueError("deadline must be positive")
    if args.min_free_bytes <= 0 or args.artifact_budget_bytes <= 0:
        raise ValueError("resource limits must be positive")
    return out, card


def child_command(args, out, card, gpus):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc_per_node=" + str(len(gpus)),
               str(TASK / "tools/run/train_robot.py"), "--config",
               str(args.config.resolve()), "--stats", str(args.stats.resolve()),
               "--source", str(args.source.resolve()), "--init-shared",
               str(args.init_shared.resolve()), "--output", str(out),
               "--experiment-card", str(card), "--gpu", gpus[0],
               "--loader-workers", str(args.loader_workers),
               "--loader-prefetch", str(args.loader_prefetch),
               "--sequence-cache", str(args.sequence_cache)]
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    if args.pin_memory:
        command.append("--pin-memory")
    return command


def main():
    args = parser().parse_args()
    gpus = parse_gpus(args.gpus)
    out, card = validate_args(args, gpus)
    scratch = ROOT / "tmp" / out.name
    scratch.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve())]
        command.extend(item for item in sys.argv[1:] if item != "--detach")
        log_path = scratch / "supervisor.log"
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        receipt = dict(supervisor_pid=child.pid, output=str(out), scratch=str(scratch),
                       command=command, physical_gpus=gpus)
        # Keep the parent receipt separate from the foreground supervisor's
        # live launch/status record; detached startup must not race its child.
        write_json(scratch / "detach.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return

    check_gpu_free(gpus)
    config = json.loads(args.config.read_text())
    deadline = args.deadline_seconds or int(config.get("max_seconds", 172800))
    if deadline <= 0 or deadline > int(config.get("max_seconds", deadline)):
        raise ValueError("deadline must stay within the v2 robot config budget")
    command = child_command(args, out, card, gpus)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
    environment.update(TMPDIR=str(scratch), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                       NCCL_DEBUG="WARN", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    status = dict(status="STARTING", supervisor_pid=os.getpid(), command=command,
                  physical_gpus=gpus, config=str(args.config.resolve()),
                  stats=str(args.stats.resolve()), source=str(args.source.resolve()),
                  parent_checkpoint=str(args.init_shared.resolve()),
                  experiment_card=str(card), output=str(out),
                  target_updates=config.get("updates"),
                  global_batch=config.get("batch_size", 0) * len(gpus),
                  poll_seconds=args.poll_seconds, stall_seconds=args.stall_seconds,
                  artifact_budget_bytes=args.artifact_budget_bytes,
                  min_free_bytes=args.min_free_bytes)
    write_json(scratch / "launch.json", status)
    stop_reason = []

    def on_signal(number, _frame):
        stop_reason.append("supervisor received signal%d" % number)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    process = None
    last_step = -1
    last_progress = started
    last_poll = 0.0
    try:
        with (scratch / "train.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True,
                                       preexec_fn=lambda: set_nofile(args.nofile_limit))
        status["torchrun_pid"] = process.pid
        while process.poll() is None:
            now = time.monotonic()
            if stop_reason:
                raise RuntimeError(stop_reason[0])
            if now - started > deadline:
                raise TimeoutError("owned group wall-time budget exceeded")
            if now - last_poll >= args.poll_seconds or last_poll == 0.0:
                progress = out / "progress.json"
                if progress.is_file():
                    row = json.loads(progress.read_text())
                    status["progress"] = row
                    if row.get("step") != last_step:
                        last_step = row.get("step", last_step)
                        last_progress = now
                if now - last_progress > args.stall_seconds:
                    raise TimeoutError("no robot training progress for configured stall timeout")
                free = shutil.disk_usage(ROOT).free
                status["free_bytes"] = free
                if free < args.min_free_bytes:
                    raise RuntimeError("filesystem free space below configured safety floor")
                status["owned_bytes"] = owned_bytes((out, scratch))
                if status["owned_bytes"] > args.artifact_budget_bytes:
                    raise RuntimeError("v2 robot artifact budget exceeded")
                status["gpu"] = gpu_snapshot(gpus)
                status.update(status="RUNNING" if last_step >= 0 else "STARTING",
                              elapsed_s=now - started)
                with (scratch / "gpu.jsonl").open("a") as metrics:
                    metrics.write(json.dumps(dict(checked_at=time.time(),
                                                  elapsed_s=now - started,
                                                  gpu=status["gpu"],
                                                  free_bytes=free,
                                                  owned_bytes=status["owned_bytes"],
                                                  step=last_step)) + "\n")
                write_json(scratch / "process.json", status)
                if out.exists():
                    for name, target in (("launcher.log", scratch / "train.log"),
                                         ("gpu.jsonl", scratch / "gpu.jsonl")):
                        link = out / name
                        if not link.exists():
                            link.symlink_to(target)
                    write_json(out / "process.json", status)
                last_poll = now
            time.sleep(min(1.0, args.poll_seconds))
        status["exit_code"] = process.returncode
        status["status"] = "COMPLETED" if process.returncode == 0 else "FAILED"
    except BaseException as error:
        status.update(status="FAILED", error=repr(error))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status["exit_code"] = process.returncode
        status["elapsed_s"] = time.monotonic() - started
        write_json(scratch / "process.json", status)
        if out.exists():
            write_json(out / "process.json", status)
    if status["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
