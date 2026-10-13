#!/usr/bin/env python3
"""Frozen EgoDex snapshot pretraining/adaptation; optionally mixed with main-three."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/"src"))

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, Dataset
from ptv3_shared_wam.checkpoint import (SCHEMA, atomic_json, directory_bytes, git_commit,
    load_checkpoint, save_checkpoint, prune_recovery_checkpoints)
from ptv3_shared_wam.contracts import GOAL_ENCODING, TASK_CONDITIONING
from ptv3_shared_wam.data import SharedWindows
from ptv3_shared_wam.egodex_data import EgoDexSharedWindows, forecasting_collate
from ptv3_shared_wam.loading import project_ipc_directory
from ptv3_shared_wam.model import SharedWAM
from ptv3_shared_wam.objectives import forecast_loss, forecast_metrics
from ptv3_shared_wam.paths import ROOT
from ptv3_shared_wam.training import setup, gradient_audit, gpu_snapshot, DEFAULT_STATS
from oakink_wm.multisource import mixed_indices


class MixedForecastWindows(Dataset):
    def __init__(self, ego, rigid=None):
        self.ego, self.rigid = ego, rigid

    def __len__(self):
        return len(self.ego)+(len(self.rigid) if self.rigid is not None else 0)

    def __getitem__(self, index):
        return self.ego[index] if index < len(self.ego) else self.rigid[index-len(self.ego)]


class StepSampler:
    def __init__(self, dataset, config, rank, start):
        self.dataset, self.config, self.rank, self.start = dataset, config, rank, start

    def __len__(self):
        return self.config["updates"]-self.start

    def __iter__(self):
        for step in range(self.start+1, self.config["updates"]+1):
            seed = int(np.random.SeedSequence([self.config["seed"], step, self.rank]).generate_state(1)[0])
            rng = np.random.default_rng(seed)
            size = self.config["batch_size"]
            n = round(size*self.config["egodex_fraction"]) if self.dataset.rigid is not None else size
            if self.dataset.rigid is not None and self.config["egodex_fraction"] < 1:
                if size < 2:
                    raise ValueError("mixed-source batches need at least two samples")
                n = min(size-1, max(1, n))
            ego = self.dataset.ego.sample_indices(n, rng)
            other = (mixed_indices(self.dataset.rigid.base, size-n, seed)+len(self.dataset.ego)
                     if size > n else np.empty(0, np.int64))
            result = np.r_[ego, other]
            rng.shuffle(result)
            yield result.tolist()


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, indices, config, device):
    model.eval()
    sums = {}
    for lo in range(0, len(indices), config["batch_size"]):
        batch = batch_to_device(forecasting_collate([dataset[int(i)] for i in indices[lo:lo+config["batch_size"]]]), device)
        prediction = model(batch)
        for key, value in forecast_metrics(prediction, batch, config).items():
            slot = sums.setdefault(key, dict(sum_mm=0., count=0))
            slot["sum_mm"] += value["sum_mm"]
            slot["count"] += value["count"]
    model.train()
    return {k: dict(mean_mm=v["sum_mm"]/v["count"] if v["count"] else None, count=v["count"]) for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--main3-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=TASK/"configs/egodex.json")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--init-shared", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment-card", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if ROOT/"outputs/ptv3-shared-wam" not in out.parents or (out.exists() and not args.resume):
        raise ValueError("fresh outputs/ptv3-shared-wam/<run_id> required; explicit --resume to reuse")
    config = json.loads(args.config.read_text())
    if (config.get("schema") != SCHEMA or config.get("goal_encoding") != GOAL_ENCODING
            or config.get("task_conditioning") != TASK_CONDITIONING
            or config.get("hidden") != 128
            or tuple(config.get(k) for k in ("fps", "history", "horizon", "object_points")) != (30, 4, 24, 32)):
        raise ValueError("current shared goal architecture required")
    if not 0 < config.get("egodex_fraction", 1.) <= 1:
        raise ValueError("egodex_fraction in (0,1] required")
    if not args.main3_root and config.get("egodex_fraction", 1.) != 1:
        raise ValueError("mixed-source fraction requires --main3-root")
    experiment = None
    if args.smoke:
        config.update(updates=4, batch_size=min(config["batch_size"], 2), validation_samples=2,
                      validation_interval=4, checkpoint_interval=4, max_seconds=600, log_interval=1)
    else:
        if not args.experiment_card or TASK/"docs/experiments" not in args.experiment_card.resolve().parents:
            raise ValueError("Task-local experiment card required for non-smoke training")
        import yaml
        experiment = yaml.safe_load(args.experiment_card.read_text().split("---", 2)[1])
        if experiment.get("task") != TASK.name or not experiment.get("seeds") or not experiment.get("experiment_id"):
            raise ValueError("card must identify ptv3-shared-wam experiment and assigned seed")
        config["seed"] = int(experiment["seeds"][0])
        if not args.init_shared and not args.resume:
            raise ValueError("non-smoke adaptation requires --init-shared or --resume")
    if (min(config["updates"], config["batch_size"], config["max_seconds"], config["validation_interval"],
            config["checkpoint_interval"], config.get("log_interval", 20)) <= 0):
        raise ValueError("positive bounded training settings required")
    if directory_bytes(ROOT/"outputs", ROOT/"tmp") >= 400_000_000_000:
        raise RuntimeError("global storage budget reached")
    # Validate data before reserving CUDA; no changes to active main-three modules.
    ego = EgoDexSharedWindows(args.snapshot, "train", engineering=args.smoke)
    dev = EgoDexSharedWindows(args.snapshot, "val", engineering=args.smoke)
    rigid = SharedWindows(args.main3_root, "train") if args.main3_root else None
    rigid_dev = SharedWindows(args.main3_root, "val") if args.main3_root else None
    stats = json.loads(args.stats.read_text())
    if stats.get("split") != "train":
        raise ValueError("TRAIN-only normalization required")
    if rigid is not None and stats.get("normalization_source_manifest_sha256") != rigid.base.meta.get("normalization_source_manifest_sha256"):
        raise ValueError("main-three normalization identity mismatch")
    provenance = dict(git_commit=git_commit(), stage="pretrain", entry="egodex-snapshot",
        engineering_only=args.smoke, experiment_id=experiment["experiment_id"] if experiment else None,
        data=str(args.snapshot.resolve()), snapshot=ego.meta,
        main3=str(args.main3_root.resolve()) if args.main3_root else None)
    parent = load_checkpoint(args.resume or args.init_shared) if args.resume or args.init_shared else None
    if parent:
        if parent["stage"] != "pretrain" or parent["stats"] != stats:
            raise ValueError("matching pretrain checkpoint and input statistics required")
        if not args.smoke and parent["provenance"].get("engineering_only"):
            raise ValueError("engineering checkpoint cannot initialize scientific adaptation")
        architecture = ("hidden", "horizon", "patch_size", "scene_tokens", "interaction_layers", "voxel_m", "voxel_origin_m", "object_points")
        if any(parent["config"].get(k) != config.get(k) for k in architecture):
            raise ValueError("architecture differs from parent shared checkpoint")
    rank, world, device = setup(args)
    provenance["world_size"] = world
    torch.manual_seed(config["seed"]+rank)
    model = SharedWAM(stats, config).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(config["scene_lr"], config["new_lr"]),
                                  weight_decay=config["weight_decay"])
    start = 0
    if parent:
        model.load_state_dict(parent["model"], strict=True)
    if args.resume:
        if parent["config"] != config or any(parent["provenance"].get(k) != provenance.get(k)
                for k in ("data", "snapshot", "main3", "world_size", "engineering_only")):
            raise ValueError("resume changes frozen data/config/world size")
        optimizer.load_state_dict(parent["optimizer"])
        start = parent["step"]
        if start >= config["updates"]:
            raise ValueError("resume checkpoint has already reached configured updates")
        torch.set_rng_state(parent["rank_rng"][rank]["torch"].cpu())
        torch.cuda.set_rng_state(parent["rank_rng"][rank]["cuda"].cpu())
    dataset = MixedForecastWindows(ego, rigid)
    sampler = StepSampler(dataset, config, rank, start)
    project_ipc_directory()
    workers = config.get("loader_workers", 2)
    if not 0 <= workers <= 8:
        raise ValueError("at most eight loader workers per rank")
    options = dict(num_workers=workers, pin_memory=True,
                   generator=torch.Generator().manual_seed(config["seed"]+rank+9001))
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2, timeout=120)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=forecasting_collate, **options)
    validation = np.random.default_rng(config["seed"]+1).choice(len(dev), min(len(dev), config["validation_samples"]), replace=False)
    rank_pids = [os.getpid()]
    if world > 1:
        rank_pids = [None]*world
        distributed.all_gather_object(rank_pids, os.getpid())
    manifest = dict(schema=SCHEMA, task=TASK.name, status="RUNNING", config=config, provenance=provenance)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
        atomic_json(out/"manifest.json", manifest)
    began = time.monotonic()
    try:
        first = next(iter(StepSampler(dataset, config, rank, start)))
        audit = gradient_audit(model, batch_to_device(forecasting_collate([dataset[first[0]]]), device), config, "pretrain")
        if rank == 0:
            atomic_json(out/"engineering_audit.json", audit)
        # Audit must not change dropout/PTv3 RNG on resumed optimization.
        if args.resume:
            torch.set_rng_state(parent["rank_rng"][rank]["torch"].cpu())
            torch.cuda.set_rng_state(parent["rank_rng"][rank]["cuda"].cpu())
        wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
        checkpoint_files = json.loads((out/"checkpoint_files.json").read_text()) if args.resume else []
        curve = json.loads((out/"curve.json").read_text()) if args.resume and (out/"curve.json").exists() else []
        implementations = list((TASK/"src").rglob("*.py"))+list((TASK.parent/"video-data-preparation/src").rglob("*.py"))+[Path(__file__)]
        signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in implementations}
        torch.cuda.reset_peak_memory_stats()
        for step, cpu_batch in enumerate(loader, start+1):
            step_started = time.monotonic()
            batch = batch_to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.get("amp", True)):
                prediction = wrapped(batch)
                loss, terms = forecast_loss(prediction, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite forecasting loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite forecasting gradients")
            optimizer.step()
            if step == start+1 or step % config["log_interval"] == 0:
                elapsed = time.monotonic()-began
                row = dict(step=step, loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                    elapsed_s=elapsed, eta_s=elapsed/(step-start)*(config["updates"]-step),
                    compute_step_s=time.monotonic()-step_started,
                    samples_per_s=config["batch_size"]*world*(step-start)/max(elapsed, 1), gpu=gpu_snapshot())
                if rank == 0:
                    atomic_json(out/"progress.json", row)
                    print(json.dumps(row), flush=True)
                physical = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[device.index]
                pids = set(map(int, __import__("subprocess").check_output(["nvidia-smi", "-i", physical,
                    "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()))
                if pids-set(rank_pids):
                    raise RuntimeError("foreign GPU process detected")
            if step % config["validation_interval"] == 0 or step == config["updates"]:
                if rank == 0:
                    report = dict(step=step, egodex=evaluate(model, dev, validation, config, device))
                    if rigid_dev is not None:
                        ids = mixed_indices(rigid_dev.base, config["validation_samples"], config["seed"]+1, equal_sources=True)
                        report["main3"] = evaluate(model, rigid_dev, ids, config, device)
                    curve.append(report)
                    atomic_json(out/"curve.json", curve)
                if world > 1:
                    distributed.barrier()
            if step % config["checkpoint_interval"] == 0 or step == config["updates"]:
                state = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
                states = [state]
                if world > 1:
                    states = [None]*world
                    distributed.all_gather_object(states, state)
                if rank == 0:
                    path = out/("step%d.pt" % step)
                    save_checkpoint(path, model, optimizer, config, stats, step, "pretrain", provenance, states)
                    checkpoint_files.append(path.name)
                    checkpoint_files = prune_recovery_checkpoints(out, checkpoint_files, config)
                    atomic_json(out/"checkpoint_files.json", checkpoint_files)
                if world > 1:
                    distributed.barrier()
                if directory_bytes(out) > 4_000_000_000 or directory_bytes(ROOT/"outputs", ROOT/"tmp") >= 400_000_000_000:
                    raise RuntimeError("training artifact budget exceeded")
            if any((p.stat().st_size, p.stat().st_mtime_ns) != value for p, value in signatures.items()):
                raise RuntimeError("implementation changed during run")
            if time.monotonic()-began > config["max_seconds"]:
                raise TimeoutError("bounded training deadline")
        manifest["status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        if rank == 0:
            manifest["elapsed_s"] = time.monotonic()-began
            atomic_json(out/"manifest.json", manifest)
        ego.close()
        dev.close()
        if world > 1:
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
