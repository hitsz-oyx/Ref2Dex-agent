"""Detach and supervise one owned torchrun group, with persistent status/logs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--loader-prefetch", type=int, default=2)
    parser.add_argument("--sequence-cache", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, help="remaining original experiment wall-time budget")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--stage", choices=("pretrain", "robot"), default="pretrain")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--init-shared", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    config_path, card = args.config.resolve(), args.experiment_card.resolve()
    if (out.exists() and not args.resume) or ROOT / "outputs/ptv3-shared-wam" not in out.parents:
        raise ValueError("fresh Task-local training output required")
    if TASK / "docs/experiments" not in card.parents or not card.is_file():
        raise ValueError("existing Task-local experiment card required")
    config = json.loads(config_path.read_text())
    deadline = args.deadline_seconds or config["max_seconds"]
    if not 1 <= deadline <= config["max_seconds"]:
        raise ValueError("remaining deadline must stay inside the original budget")
    gpus = args.gpus.split(",")
    if not 1 <= len(gpus) <= 6 or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError("one to six distinct physical GPU indices required")
    for gpu in gpus:
        if subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                    "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("assigned GPU occupied: " + gpu)
    scratch = ROOT / "tmp" / out.name
    scratch.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(out),
                   "--config", str(config_path), "--experiment-card", str(card), "--gpus", args.gpus,
                   "--loader-workers", str(args.loader_workers), "--loader-prefetch", str(args.loader_prefetch),
                   "--sequence-cache", str(args.sequence_cache)]
        command.extend(["--stage", args.stage])
        if args.source:
            command.extend(["--source", str(args.source.resolve())])
        if args.init_shared:
            command.extend(["--init-shared", str(args.init_shared.resolve())])
        if args.resume:
            command.extend(["--resume", str(args.resume.resolve())])
        if args.pin_memory:
            command.append("--pin-memory")
        command.extend(["--deadline-seconds", str(deadline)])
        with (scratch / "supervisor.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1"))
        receipt = dict(supervisor_pid=child.pid, output=str(out), scratch=str(scratch), command=command)
        write_json(scratch / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)
        return
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, TMPDIR=str(scratch),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       TORCH_NCCL_ASYNC_ERROR_HANDLING="1", NCCL_DEBUG="WARN",
                       PYTHONDONTWRITEBYTECODE="1")
    sys.path.insert(0, str(TASK / "src"))
    from ptv3_shared_wam.checkpoint import directory_bytes
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--nproc_per_node=" + str(len(gpus)), str(TASK / ("tools/run/train_%s.py" % args.stage)),
               "--config", str(config_path), "--experiment-card", str(card), "--output", str(out),
               "--gpu", gpus[0]]
    if args.source:
        command.extend(["--source", str(args.source.resolve())])
    if args.init_shared:
        command.extend(["--init-shared", str(args.init_shared.resolve())])
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    if args.loader_workers:
        command.extend(["--loader-workers", str(args.loader_workers), "--loader-prefetch", str(args.loader_prefetch),
                        "--sequence-cache", str(args.sequence_cache)])
    if args.pin_memory:
        command.append("--pin-memory")
    started = time.monotonic()
    status = dict(status="STARTING", supervisor_pid=os.getpid(), command=command, physical_gpus=gpus,
                  config=str(config_path), experiment_card=str(card), output=str(out),
                  target_updates=config["updates"], global_batch=config["batch_size"] * len(gpus))
    requested_stop = []
    def on_signal(number, frame):
        requested_stop.append("supervisor received signal%d" % number)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    process = None
    last_step, last_progress = -1, started
    storage_checked = 0.
    try:
        with (scratch / "train.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            status["torchrun_pid"] = process.pid
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if requested_stop:
                    raise RuntimeError(requested_stop[0])
                if elapsed > deadline:
                    raise TimeoutError("owned group wall-time budget exceeded")
                if elapsed - storage_checked >= 120 or storage_checked == 0:
                    if directory_bytes(ROOT / "outputs", ROOT / "tmp") >= 400_000_000_000:
                        raise RuntimeError("shared400GB artifact budget reached")
                    owned = [scratch] + ([out] if out.exists() else [])
                    if directory_bytes(*owned) > 4_000_000_000:
                        raise RuntimeError("owned run artifact budget4GB reached")
                    storage_checked = elapsed
                progress = out / "progress.json"
                if progress.exists():
                    row = json.loads(progress.read_text())
                    status["progress"] = row
                    if row["step"] != last_step:
                        last_step, last_progress = row["step"], time.monotonic()
                if time.monotonic() - last_progress > 1800:
                    raise TimeoutError("no training progress for30 minutes")
                status.update(status="RUNNING" if last_step >= 0 else "STARTING", elapsed_s=elapsed)
                status["gpu"] = subprocess.check_output(["nvidia-smi", "-i", args.gpus,
                    "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                with (scratch / "gpu.jsonl").open("a") as metrics:
                    metrics.write(json.dumps(dict(elapsed_s=elapsed, gpu=status["gpu"])) + "\n")
                write_json(scratch / "process.json", status)
                if out.exists():
                    for name, target in (("launcher.log", scratch / "train.log"),
                                         ("gpu.jsonl", scratch / "gpu.jsonl")):
                        link = out / name
                        if not link.exists():
                            link.symlink_to(target)
                    write_json(out / "process.json", status)
                time.sleep(30)
            status["exit_code"] = process.returncode
            status["status"] = "COMPLETED" if process.returncode == 0 else "FAILED"
    except BaseException as error:
        status.update(status="FAILED", error=repr(error))
    finally:
        if process is not None and process.poll() is None:
            # This new process group was created above and belongs to this run.
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
