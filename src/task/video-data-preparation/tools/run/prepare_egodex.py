#!/usr/bin/env python3
"""Bounded, restartable consumer of the existing raw READY stream."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/"src"))

from video_data_preparation.backends import AnnotatedMasks, CoTrackerBackend, OmegaDepthBackend, load_backend
from video_data_preparation.geometry import resize_intrinsic
from video_data_preparation.io import atomic_json
from video_data_preparation.job_ledger import JobLedger, stable_split
from video_data_preparation.native_hand import read_native
from video_data_preparation.pipeline import ProcessingConfig, RejectedWindow, process_window
from video_data_preparation.shards import ShardWriter, recover_publications
from video_data_preparation.video import VideoFrames


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=TASK/"configs/geometry_only.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    for name in ("depth-vendor", "depth-checkpoint", "tracker-vendor", "tracker-checkpoint"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--once", action="store_true", help="drain currently available jobs, then exit")
    parser.add_argument("--index-only", action="store_true", help="import READY events without loading GPU models")
    return parser


def directory_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def gpu_state(gpu):
    row = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    pids = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).split()
    return dict(gpu=gpu, utilization_memory=row, pids=list(map(int, pids)))


def main(argv=None):
    args = args_parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    for key in ("depth_vendor", "depth_checkpoint", "tracker_vendor", "tracker_checkpoint"):
        if getattr(args, key):
            config[key] = str(getattr(args, key).resolve())
    if config.get("profile") != "geometry-only":
        raise ValueError("only geometry-only profile implemented; H4 motion is a separate Probe")
    output = args.output.resolve()
    if output == Path("/tmp") or Path("/tmp") in output.parents:
        raise ValueError("outputs cannot be written to /tmp")
    if args.gpu < 0:
        raise ValueError("physical GPU index must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        # Raw discovery is independent of a yet-unselected visual config. Keep
        # its cursor apart so starting a real consumer can import the same pairs.
        raw_ledger = JobLedger(output/"raw_index.sqlite")
        try:
            raw_ledger.ingest(args.ready, "raw-index")
            print(json.dumps(dict(profile_id=None, scope="raw-index", jobs=raw_ledger.counts())))
        finally:
            raw_ledger.close()
        return
    receipt = output/"run_config.json"
    identity = dict(config=config, ready=str(args.ready.resolve()),
                    annotations=str(args.annotations.resolve()) if args.annotations else None)
    if args.annotations:
        annotation_stat = args.annotations.stat()
        identity["annotation_revision"] = [annotation_stat.st_size, annotation_stat.st_mtime_ns]
    if receipt.exists():
        if json.loads(receipt.read_text()) != identity:
            raise ValueError("resume changes source/config/annotations; use a fresh output")
    else:
        atomic_json(receipt, identity, exclusive=True)
    profile_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    ledger = JobLedger(output/"jobs.sqlite")
    try:
        ledger.ingest(args.ready, profile_id)
        for family in ("depth", "tracker"):
            if not config.get(family+"_backend") and any(not config.get(family+suffix) for suffix in ("_vendor", "_checkpoint")):
                raise ValueError("configure --%s-vendor and --%s-checkpoint, or %s_backend module:factory" % (family, family, family))
        # Fail on unreadable split metadata before allocating persistent GPU models.
        registry = json.loads(Path(config["split_registry"]).read_text()) if config.get("split_registry") else None
        state = gpu_state(args.gpu)
        if state["pids"]:
            raise RuntimeError("assigned GPU occupied; choose an idle GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import cv2
        import torch
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        scratch = ROOT/"tmp"/("egodex-"+profile_id)
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ.update(TMPDIR=str(scratch), TORCH_EXTENSIONS_DIR=str(scratch/"extensions"))
        stop = [False]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        processing = ProcessingConfig(**config.get("processing", {}))
        # Assets are loaded once, in this owned consumer. No implicit downloads.
        mask = (load_backend(config["mask_backend"], config) if config.get("mask_backend") else
                AnnotatedMasks(args.annotations) if args.annotations else None)
        if mask is None:
            raise ValueError("--annotations or an explicit mask backend required")
        depth = (load_backend(config["depth_backend"], config) if config.get("depth_backend") else
                 OmegaDepthBackend(config["depth_vendor"], config["depth_checkpoint"], scratch))
        tracker = (load_backend(config["tracker_backend"], config) if config.get("tracker_backend") else
                   CoTrackerBackend(config["tracker_vendor"], config["tracker_checkpoint"]))
        recover_publications(output, profile_id)
        # A single startup scan rebuilds the indexed native-window resume ledger.
        for path in (output/"shards").glob("*.json"):
            info = json.loads(path.read_text())
            if info["profile_id"] != profile_id:
                raise ValueError("output contains a different preprocessing profile")
            ledger.register_shard(info)
        started, accepted, attempted, last_progress = time.monotonic(), 0, 0, 0.
        admitted, rejection_counts = 0, {}
        runtime = float(config.get("max_seconds", 3600))
        max_windows = int(config.get("max_windows", 1000))
        output_cap = int(config.get("max_output_bytes", 2*2**30))
        scratch_cap = int(config.get("max_scratch_bytes", 5*2**30))
        if min(runtime, max_windows, output_cap, scratch_cap, int(config.get("stride", 8))) <= 0:
            raise ValueError("finite positive time/storage/window/stride budgets required")
        def guard():
            if time.monotonic()-started >= runtime or attempted >= max_windows or stop[0]:
                raise InterruptedError("STOPPED: bounded deadline/window limit or user signal")
            if directory_bytes(output) >= output_cap or directory_bytes(scratch) >= scratch_cap:
                raise InterruptedError("STOPPED: owned artifact budget reached")
            if set(gpu_state(args.gpu)["pids"])-{os.getpid()}:
                raise InterruptedError("STOPPED: foreign GPU process")
        def progress(current=None):
            nonlocal last_progress
            if time.monotonic()-last_progress < 30:
                return
            elapsed = time.monotonic()-started
            row = dict(status="RUNNING", current=current, profile_id=profile_id, accepted=accepted,
                       numeric_candidates=accepted, training_admitted=admitted, rejected=rejection_counts,
                       attempted=attempted, elapsed_s=elapsed, accepted_per_s=accepted/max(elapsed, 1),
                       eta_to_window_limit_s=(max_windows-attempted)*elapsed/max(attempted, 1),
                       jobs=ledger.counts(), gpu=gpu_state(args.gpu), output_bytes=directory_bytes(output))
            atomic_json(output/"progress.json", row)
            print(json.dumps(row), flush=True)
            last_progress = time.monotonic()
        try:
            while not stop[0]:
                guard()
                ledger.ingest(args.ready, profile_id)
                job = ledger.claim(lease_seconds=1800)
                if job is None:
                    progress()
                    if args.once:
                        break
                    time.sleep(2)
                    continue
                writer = None
                report = dict(accepted=0, rejected={}, status="PROCESSING")
                try:
                    raw = Path(job["payload"]["raw_hdf5"])
                    native = read_native(raw)
                    split = stable_split(native.session, registry)
                    key = job["payload"]["key"]
                    old_ticks = ledger.published_ticks(profile_id, key)
                    episode_id = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
                    stem = hashlib.sha256(job["key"].encode()).hexdigest()[:16]
                    def open_writer():
                        return ShardWriter(output/"shards"/(stem+"-"+job["token"]+"-"+str(report["accepted"])+".h5"),
                                           split, native.session, profile_id,
                                           config.get("shard_bytes", 256*2**20))
                    with VideoFrames(raw.with_suffix(".mp4"), native.frames) as video:
                        intrinsic = resize_intrinsic(native.intrinsic, video.original_wh, video.size)
                        episode_started = time.monotonic()
                        for tick in range(3, native.frames-24, config.get("stride", 8)):
                            if tick in old_ticks:
                                continue
                            guard()
                            if time.monotonic()-episode_started > config.get("episode_seconds", 1200):
                                raise TimeoutError("bounded episode deadline")
                            ledger.heartbeat(job)
                            attempted += 1
                            rgb = video.get(range(tick-3, tick+25))
                            try:
                                sample, meta, audit = process_window(native, key, episode_id, tick, rgb,
                                    intrinsic, mask, depth, tracker, processing, config.get("depth_to_native"))
                            except (RejectedWindow, ValueError, KeyError) as error:
                                reason = str(error)
                                report["rejected"][reason] = report["rejected"].get(reason, 0)+1
                                category = reason.split(":", 1)[0]
                                rejection_counts[category] = rejection_counts.get(category, 0)+1
                                with (output/"events.jsonl").open("a") as events:
                                    events.write(json.dumps(dict(key=key, tick=tick, status="REJECTED",
                                        reason=reason, elapsed_s=time.monotonic()-started))+"\n")
                                progress(key)
                                continue
                            if writer is None:
                                writer = open_writer()
                            ledger.heartbeat(job)  # Long inference may have outlived the lease.
                            meta.update(raw_hdf5=str(raw.resolve()),
                                        acquisition_manifest=job["payload"].get("source_manifest"),
                                        preprocessing_profile_id=profile_id)
                            writer.append(sample, meta, audit)
                            accepted += 1
                            admitted += int(meta["training_allowed"])
                            report["accepted"] += 1
                            with (output/"events.jsonl").open("a") as events:
                                events.write(json.dumps(dict(key=key, tick=tick,
                                    status="ADMITTED" if meta["training_allowed"] else "NUMERIC_CANDIDATE",
                                    timings=meta["timings"], elapsed_s=time.monotonic()-started))+"\n")
                            if writer.full:
                                ledger.heartbeat(job)
                                ledger.register_shard(writer.publish())
                                writer = None
                            progress(key)
                    if writer is not None:
                        ledger.heartbeat(job)
                        ledger.register_shard(writer.publish())
                        writer = None
                    report["status"] = "PUBLISHED" if report["accepted"] or old_ticks else "REJECTED"
                    ledger.finish(job, report["status"], report)
                except BaseException as error:
                    try:
                        ledger.heartbeat(job)
                    except RuntimeError:
                        if writer is not None:
                            writer.abort()
                            writer = None
                        report.update(status="LOST_LEASE", error=repr(error))
                        continue
                    if writer is not None and writer.count:
                        ledger.register_shard(writer.publish())  # Keep completed windows on clean stop or retry.
                        writer = None
                    elif writer is not None:
                        writer.abort()
                    report.update(error=repr(error))
                    state = "DEFERRED" if isinstance(error, TimeoutError) else "RETRYABLE"
                    ledger.finish(job, state, report)
                    if isinstance(error, InterruptedError):
                        raise
                    if not isinstance(error, Exception):
                        raise
                    print(json.dumps(dict(job=job["key"], state=state, error=repr(error))), flush=True)
                finally:
                    atomic_json(output/"last_job.json", dict(key=job["key"], **report))
            status = "DRAINED" if args.once else "STOPPED"
        except InterruptedError as error:
            status = "STOPPED"
            print(str(error), flush=True)
        atomic_json(output/"progress.json", dict(status=status, accepted=accepted, attempted=attempted,
                                                  numeric_candidates=accepted, training_admitted=admitted,
                                                  rejected=rejection_counts, gpu=gpu_state(args.gpu),
                                                  elapsed_s=time.monotonic()-started, jobs=ledger.counts()))
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
