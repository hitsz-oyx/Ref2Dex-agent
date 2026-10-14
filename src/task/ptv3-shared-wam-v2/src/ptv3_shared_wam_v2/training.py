"""Bounded v2 GPU pretraining; reuse the frozen v1 data/ordered prefetch contract."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from .paths import ROOT, TASK, LEGACY, DEFAULT_DATA, SCHEMA
from .checkpoint import (atomic_json, directory_bytes, git_commit, save_checkpoint, load_checkpoint,
                         checkpoint_alias, prune_recovery_checkpoints)
from .model import SharedWAMV2
from .objectives import forecast_loss, forecast_metrics
from ptv3_shared_wam.data import SharedWindows
from ptv3_shared_wam.loading import pretrain_loader, forecasting_collate, to_device
from oakink_wm.multisource import mixed_indices


def gpu_snapshot():
    return dict(allocated_mib=torch.cuda.memory_allocated()/2**20,
        reserved_mib=torch.cuda.memory_reserved()/2**20,
        nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                           "--format=csv,noheader,nounits"], text=True).strip())


def gradient_audit(model, batch, config):
    model.eval()
    prediction = model(batch)
    loss, terms = forecast_loss(prediction, batch, config, model.normalizer)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite actual-data objective")
    report = dict(parameters=model.parameter_counts())
    for task, term in terms.items():
        for group, params in (("PTv3", model.geometry.backbone.parameters()),
                              ("shared", model.interaction.parameters()),
                              ("hand_encoder", model.hand_encoder.parameters()),
                              ("goal", model.goal.parameters())):
            grads = torch.autograd.grad(term, list(params), retain_graph=True, allow_unused=True)
            norm = sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(task + " does not reach " + group)
            report[task + "_" + group + "_gradient"] = norm
    hand = model.encode_hand(batch)
    observed = {key: batch[key] for key in ("xyz", "features", "point_valid", "start_hand", "start_hand_valid")}
    geometry = model.geometry(observed, *hand)
    route = geometry["hand"].square().mean()
    grads = torch.autograd.grad(route, list(model.hand_encoder.parameters()), allow_unused=True)
    report["hand_encoder_via_PT_v3"] = sum(float(g.abs().sum()) for g in grads if g is not None)
    if config["hand_in_ptv3"] and report["hand_encoder_via_PT_v3"] <= 0:
        raise ValueError("hand encoder features did not enter PTv3")
    branch = torch.autograd.grad(terms["scene"], list(model.hand_future.parameters()), allow_unused=True)
    if any(g is not None for g in branch):
        raise ValueError("scene head depends on hand prediction head")
    for key in ("trajectory", "scene"):
        name = "hand" if key == "trajectory" else "scene"
        physical = model.normalizer.denormalize(name, prediction[key + "_normalized"])
        valid = (batch["start_hand_valid"][:, None].expand(*physical.shape[:-1]) if name == "hand"
                 else batch["object_valid"][:, :, None, None].expand(*physical.shape[:-1]))
        if not torch.allclose(physical[valid], prediction[key][valid], atol=1e-7):
            raise ValueError("metric output does not invert horizon normalization")
    model.train()
    return report


@torch.no_grad()
def evaluate(model, dataset, indices, config, device):
    model.eval()
    totals, objective, count = {}, 0., 0
    # Bound readout memory independently of optimizer batch size.
    for start in range(0, len(indices), min(config["batch_size"], 16)):
        samples = indices[start:start + min(config["batch_size"], 16)]
        batch = to_device(forecasting_collate([dataset[int(i)] for i in samples]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["amp"]):
            pred = model(batch)
            loss, _ = forecast_loss(pred, batch, config, model.normalizer)
        objective += float(loss) * len(samples)
        count += len(samples)
        for sid in batch["source_id"].unique().tolist():
            rows = batch["source_id"] == sid
            metrics = forecast_metrics({k: v[rows] for k, v in pred.items()},
                                       {k: v[rows] for k, v in batch.items()}, config)
            for key, value in metrics.items():
                slot = totals.setdefault("source%d/%s" % (sid, key), dict(sum_mm=0., count=0))
                slot["sum_mm"] += value["sum_mm"]
                slot["count"] += value["count"]
    model.train()
    return dict(objective=objective/max(count, 1), metrics={k:dict(mean_mm=v["sum_mm"]/v["count"]
        if v["count"] else None, count=v["count"]) for k,v in totals.items()})


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK / "configs/pretrain_probe.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--experiment-card", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--loader-prefetch", type=int, default=2)
    parser.add_argument("--sequence-cache", type=int, default=128)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()
    out, config = args.output.resolve(), json.loads(args.config.read_text())
    if ROOT / "outputs" / TASK.name not in out.parents or out.exists():
        raise ValueError("fresh v2 output required, including explicit resume into a new run")
    if config.get("schema") != SCHEMA:
        raise ValueError("v2 config required")
    if directory_bytes(ROOT / "outputs", ROOT / "tmp") >= 400_000_000_000:
        raise RuntimeError("shared400GB budget reached")
    experiment = None
    if args.smoke:
        if not 1 <= args.smoke_batch_size <= 256:
            raise ValueError("smoke batch must be in[1,256]")
        config.update(updates=4, batch_size=args.smoke_batch_size, validation_samples=3,
            validation_interval=4, checkpoint_interval=4, log_interval=1, max_seconds=600, seed=17)
    else:
        import yaml
        if not args.experiment_card or TASK / "docs/experiments" not in args.experiment_card.resolve().parents:
            raise ValueError("Task-local experiment card required")
        experiment = yaml.safe_load(args.experiment_card.read_text().split("---", 2)[1])
        if experiment.get("task") != TASK.name or not experiment.get("seeds"):
            raise ValueError("v2 experiment identity and seeds required")
        if config["seed"] != experiment["seeds"][0]:
            raise ValueError("training seed must match experiment")
    world, rank, local = (int(os.environ.get(k, v)) for k,v in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if not 1 <= world <= 6:
        raise ValueError("max6 simultaneous GPUs")
    if world == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    physical = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[local]
    if subprocess.check_output(["nvidia-smi", "-i", physical, "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("assigned GPU already occupied")
    scratch = ROOT / "tmp" / out.name / ("rank%d" % rank)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch / "triton"),
                      TORCH_EXTENSIONS_DIR=str(ROOT / "tmp/shared-wam-v2-extensions"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(config["seed"] + rank)
    train = SharedWindows(args.data, "train", config["object_points"])
    dev = SharedWindows(args.data, "val", config["object_points"])
    stats = json.loads(args.stats.read_text())
    if (stats.get("data") != str(args.data.resolve()) or stats.get("object_points") != config["object_points"]
            or stats.get("normalization_source_manifest_sha256") != train.base.meta.get("normalization_source_manifest_sha256")
            or stats.get("source_weights") != train.base.source_weights.tolist()):
        raise ValueError("horizon statistics do not match the current TRAIN corpus/sampler")
    provenance = dict(git_commit=git_commit(), engineering_only=args.smoke,
        experiment_id=experiment["experiment_id"] if experiment else None,
        data=str(args.data.resolve()), stats=str(args.stats.resolve()), world_size=world,
        initialization="all v2 weights from scratch; no v1/PW checkpoint transfer",
        execution=dict(loader_workers=args.loader_workers, loader_prefetch=args.loader_prefetch,
                       sequence_cache=args.sequence_cache, pin_memory=args.pin_memory))
    model = SharedWAMV2(stats, config).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(config["geometry_lr"], config["shared_lr"]),
                                  weight_decay=config["weight_decay"], fused=True)
    start_step, previous_elapsed = 0, 0.
    parent = None
    if args.resume:
        parent = load_checkpoint(args.resume)
        if parent["config"] != config or parent["stats"] != stats or parent["provenance"]["world_size"] != world:
            raise ValueError("resume must preserve config/statistics/world size")
        if parent["provenance"]["engineering_only"] != args.smoke:
            raise ValueError("engineering checkpoints cannot initialize scientific training")
        model.load_state_dict(parent["model"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        start_step = parent["step"]
        if not 0 < start_step < config["updates"]:
            raise ValueError("resume requires a partial checkpoint before the target update")
        previous_elapsed = parent["provenance"].get("elapsed_s", 0.)
        provenance.update(resume_parent=str(args.resume.resolve()), parent_step=start_step)
    validation = mixed_indices(dev.base, config["validation_samples"], config["validation_seed"], equal_sources=True)
    workers = [None]*world
    if world > 1:
        dist.all_gather_object(workers, os.getpid())
    else:
        workers = [os.getpid()]
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        np.save(out / "validation.npy", validation)
    if world > 1:
        dist.barrier()
    manifest = dict(schema=SCHEMA, task=TASK.name, status="RUNNING", config=config, provenance=provenance,
                    parameters=model.parameter_counts(), worker_pids=workers)
    if rank == 0:
        atomic_json(out / "manifest.json", manifest)
    implementation = list((TASK / "src").rglob("*.py")) + list((LEGACY / "src").rglob("*.py"))
    code_state = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in implementation}
    started = time.monotonic()
    curve, best, files = [], float("inf"), []
    try:
        seed = int(np.random.SeedSequence([config["seed"], start_step + 1, rank]).generate_state(1)[0])
        audit_batch = to_device(forecasting_collate([train[int(mixed_indices(train.base, 1, seed)[0])]]), device)
        audit = gradient_audit(model, audit_batch, config)
        del audit_batch
        if rank == 0:
            atomic_json(out / "engineering_audit.json", audit)
        if parent:
            torch.set_rng_state(parent["rank_rng"][rank]["torch"])
            torch.cuda.set_rng_state(parent["rank_rng"][rank]["cuda"])
        # Resume RNG restoration follows audits so audit execution cannot perturb it.
        wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[local], gradient_as_bucket_view=True) if world > 1 else model
        loader = pretrain_loader(train, config["batch_size"], config["seed"], rank, start_step+1,
            config["updates"], args.loader_workers, args.loader_prefetch, args.sequence_cache, args.pin_memory)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            initial = evaluate(model, dev, validation, config, device)
            curve.append(dict(step=start_step, **initial))
            atomic_json(out / "curve.json", curve)
        if world > 1:
            dist.barrier()
        logged, logged_step = time.monotonic(), start_step
        for step, cpu_batch in enumerate(loader, start_step+1):
            batch = to_device(cpu_batch, device, non_blocking=args.pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config["amp"]):
                pred = wrapped(batch)
                loss, terms = forecast_loss(pred, batch, config, model.normalizer)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step()
            if step == start_step+1 or step % config["log_interval"] == 0:
                torch.cuda.synchronize()
                seconds = (time.monotonic()-logged)/(step-logged_step)
                row = dict(step=step, loss=float(loss), terms={k:float(v) for k,v in terms.items()},
                    elapsed_s=time.monotonic()-started, step_seconds=seconds,
                    eta_s=seconds*(config["updates"]-step), global_batch=config["batch_size"]*world,
                    samples_per_second=config["batch_size"]*world/seconds,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, gpu=gpu_snapshot())
                logged, logged_step = time.monotonic(), step
                if rank == 0:
                    atomic_json(out / "progress.json", row)
                    with (out / "train.jsonl").open("a") as stream:
                        stream.write(json.dumps(row)+"\n")
                    print(json.dumps(row), flush=True)
                pids = set(map(int, subprocess.check_output(["nvidia-smi", "-i", physical,
                    "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()))
                if pids-set(workers):
                    raise RuntimeError("foreign GPU process appeared")
            new_best = False
            if step % config["validation_interval"] == 0 or step == config["updates"]:
                if rank == 0:
                    report = evaluate(model, dev, validation, config, device)
                    if not np.isfinite(report["objective"]):
                        raise ValueError("nonfinite dev objective")
                    curve.append(dict(step=step, **report))
                    atomic_json(out / "curve.json", curve)
                    new_best = report["objective"] < best
                    if new_best:
                        best = report["objective"]
                selection = [new_best]
                if world > 1:
                    dist.broadcast_object_list(selection, src=0)
                new_best = selection[0]
            if new_best or step % config["checkpoint_interval"] == 0 or step == config["updates"]:
                rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
                states = [None]*world
                if world > 1:
                    dist.all_gather_object(states, rng)
                else:
                    states = [rng]
                if rank == 0:
                    path = out / ("step%d.pt" % step)
                    provenance["elapsed_s"] = previous_elapsed + time.monotonic()-started
                    save_checkpoint(path, model, optimizer, config, stats, step, provenance, states)
                    files.append(path.name)
                    checkpoint_alias(out, "latest.pt", path.name)
                    if new_best:
                        checkpoint_alias(out, "best.pt", path.name)
                    files = prune_recovery_checkpoints(out, files, config)
                    atomic_json(out / "checkpoint_files.json", files)
                    if directory_bytes(out, ROOT / "tmp" / out.name) > config["artifact_budget_bytes"]:
                        raise RuntimeError("owned artifact budget exceeded")
                    if directory_bytes(ROOT / "outputs", ROOT / "tmp") >= 400_000_000_000:
                        raise RuntimeError("shared400GB budget reached")
                if world > 1:
                    dist.barrier()
            if time.monotonic()-started+previous_elapsed > config["max_seconds"]:
                raise TimeoutError("bounded total training deadline")
            if any((p.stat().st_size,p.stat().st_mtime_ns) != signature for p,signature in code_state.items()):
                raise RuntimeError("implementation changed during training")
        if rank == 0:
            manifest["status"] = "COMPLETED"
            atomic_json(out / "manifest.json", manifest)
            atomic_json(out / "result.json", dict(status="COMPLETED", elapsed_s=time.monotonic()-started,
                final_step=config["updates"], scope="engineering_only" if args.smoke else "offline Probe; grasp/Cm benefit untested"))
    except BaseException as error:
        if rank == 0:
            manifest.update(status="FAILED", error=repr(error))
            atomic_json(out / "manifest.json", manifest)
        raise
    finally:
        if world > 1:
            dist.destroy_process_group()
