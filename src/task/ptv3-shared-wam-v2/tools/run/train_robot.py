"""V2 WAM robot fine-tuning with an explicit q_ref/q_pd/q_real contract."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
RUNTIME = TASK / "src"
SCHEMA = "ref2dex.ptv3-shared-wam.v2"

import sys
sys.path.insert(0, str(RUNTIME))
from ptv3_shared_wam_v2.checkpoint import (atomic_json, directory_bytes, git_commit,
                                           checkpoint_alias, prune_recovery_checkpoints,
                                           restore_optimizer)
from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows, robot_collate
from ptv3_shared_wam_v2.robot_model import SharedWAMV2Robot, robot_loss
from ptv3_shared_wam_v2.training import gpu_snapshot
from ptv3_shared_wam_v2.paths import SCHEMA as V2_SCHEMA

sys.path.insert(0, str(TASK.parent / "ptv3-shared-wam" / "src"))
from ptv3_shared_wam.action_targets import NATIVE, validate_representation
from ptv3_shared_wam.loading import pretrain_loader, to_device
from ptv3_shared_wam.robot import initialize_retargeter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--init-shared", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="optional robot checkpoint; writes resumed state into a fresh output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-card", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--loader-prefetch", type=int, default=2)
    parser.add_argument("--sequence-cache", type=int, default=16)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def setup(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not 1 <= world <= 6:
        raise ValueError("shared max6 GPU budget")
    if world == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        local = 0
    if not torch.cuda.is_available():
        raise RuntimeError("robot fine-tuning requires CUDA")
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    physical = visible[local]
    if subprocess.check_output(["nvidia-smi", "-i", physical, "--query-compute-apps=pid",
                                "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("assigned GPU occupied before fine-tuning")
    scratch = ROOT / "tmp" / args.output.name / ("rank%d" % rank)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch / "triton"),
                      TORCH_EXTENSIONS_DIR=str(ROOT / "tmp/shared-wam-v2-extensions"))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl")
    return rank, world, local, physical, torch.device("cuda", local)


def check_parent(parent, stats, config):
    if parent.get("schema") != V2_SCHEMA or parent.get("stage") != "pretrain":
        raise ValueError("remote v2 pretraining checkpoint required; v1 is incompatible")
    if parent.get("stats") != stats:
        raise ValueError("parent checkpoint statistics differ from the robot run")
    architectural = ("schema", "goal_encoding", "task_conditioning", "point_width",
                     "shared_width", "shared_heads", "shared_layers", "head_width",
                     "head_heads", "patch_size", "scene_tokens", "ptv3", "fps", "history",
                     "object_points", "horizon", "voxel_m", "voxel_origin_m", "position_scale_m",
                     "velocity_scale_mps", "acceleration_scale_mps2", "hand_in_ptv3", "explicit_hand",
                     "horizon_normalization", "checkpoint_interaction")
    for key in architectural:
        if parent["config"].get(key) != config.get(key):
            raise ValueError("parent/config architecture differs: " + key)
    if parent["provenance"].get("engineering_only"):
        raise ValueError("engineering pretraining checkpoint cannot initialize scientific fine-tune")


def save_robot_checkpoint(path, model, optimizer, config, stats, step, provenance, rank_rng):
    packet = dict(schema=V2_SCHEMA, stage="robot", config=config, stats=stats, step=step,
                  model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                  optimizer=optimizer.state_dict(), provenance=provenance, rank_rng=rank_rng)
    temporary = path.with_suffix(".pt.part")
    torch.save(packet, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, dataset, indices, config, device):
    model.eval()
    total = action = anchor = 0.
    count = 0
    for start in range(0, len(indices), min(config["batch_size"], 16)):
        rows = indices[start:start + min(config["batch_size"], 16)]
        batch = to_device(robot_collate([dataset[int(i)] for i in rows]), device)
        prediction = model(batch)
        loss, terms = robot_loss(prediction, batch, config, model)
        n = len(rows)
        total += float(loss) * n
        action += float(terms["action"]) * n
        anchor += float(terms["anchor"]) * n
        count += n
    model.train()
    return dict(objective=total / max(count, 1), action=action / max(count, 1),
                anchor=anchor / max(count, 1), count=count)


def main():
    args = parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / "outputs/ptv3-shared-wam-v2" not in out.parents:
        raise ValueError("fresh v2 Task output required")
    config = json.loads(args.config.read_text())
    stats = json.loads(args.stats.read_text())
    if config.get("schema") != V2_SCHEMA:
        raise ValueError("v2 config required")
    representation = validate_representation(config.get("action_representation", NATIVE))
    if args.smoke:
        if not 1 <= args.smoke_batch_size <= 32:
            raise ValueError("smoke batch must be in[1,32]")
        config.update(updates=4, batch_size=args.smoke_batch_size, validation_interval=4,
                      checkpoint_interval=4, log_interval=1, max_seconds=600)
    else:
        import yaml
        card = yaml.safe_load(args.experiment_card.read_text().split("---", 2)[1])
        if card.get("task") != TASK.name or not card.get("seeds") or not card.get("experiment_id"):
            raise ValueError("Task-local experiment identity required")
        config["seed"] = int(card["seeds"][0])
    if args.steps is not None:
        config["updates"] = int(args.steps)
    rank, world, local, physical, device = setup(args)
    torch.backends.cuda.matmul.allow_tf32 = bool(config.get("tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(config.get("tf32", False))
    torch.manual_seed(config["seed"] + rank)
    train = V2RLRobotWindows(args.source, "train", config["object_points"], representation)
    dev = V2RLRobotWindows(args.source, "val", config["object_points"], representation)
    parent = torch.load(args.init_shared, map_location="cpu", weights_only=False)
    check_parent(parent, stats, config)
    model = SharedWAMV2Robot(stats, config, representation)
    model.forecaster.load_state_dict(parent["model"], strict=True)
    initialize_retargeter(model.retargeter, train)
    model.to(device)
    groups = model.parameter_groups(config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", .01),
                                  fused=bool(config.get("fused_optimizer", False)))
    resume = None
    start_step = 0
    previous_elapsed = 0.0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("schema") != V2_SCHEMA or resume.get("stage") != "robot":
            raise ValueError("robot resume checkpoint required")
        if resume.get("config") != config or resume.get("stats") != stats:
            raise ValueError("resume must preserve config and statistics")
        if resume.get("provenance", {}).get("world_size") != world:
            raise ValueError("resume must preserve world size")
        resume_provenance = resume.get("provenance", {})
        if resume_provenance.get("source") != str(args.source.resolve()):
            raise ValueError("resume must preserve the RL660 source")
        if resume_provenance.get("action_representation") != representation:
            raise ValueError("resume must preserve action representation")
        if resume_provenance.get("parent_step") != parent["step"]:
            raise ValueError("resume must preserve parent checkpoint step")
        start_step = int(resume.get("step", 0))
        if not 0 < start_step < config["updates"]:
            raise ValueError("resume requires a checkpoint before the target update")
        model.load_state_dict(resume["model"], strict=True)
        restore_optimizer(optimizer, resume["optimizer"])
        previous_elapsed = float(resume.get("provenance", {}).get("elapsed_s", 0.0))
    validation_rng = np.random.default_rng(config.get("validation_seed", config["seed"] + 1))
    validation = validation_rng.choice(len(dev), min(len(dev), config.get("validation_samples", 128)), replace=False)
    provenance = dict(git_commit=git_commit(), task=TASK.name, stage="robot",
                      engineering_only=args.smoke, experiment_id=None if args.smoke else card["experiment_id"],
                      source=str(args.source.resolve()), stats=str(args.stats.resolve()),
                      parent_checkpoint=str(args.init_shared.resolve()), parent_step=parent["step"],
                      world_size=world, action_representation=representation,
                      q_contract=dict(q_ref="geometric interaction_hand_inspire.pt raw cols373:391",
                                      q_pd="RL660 pd_targets; executable loss label",
                                      q_real="RL660 measured q; state/diagnostic, not q_ref"))
    if resume:
        provenance.update(resume_parent=str(args.resume.resolve()), resume_step=start_step)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        np.save(out / "validation.npy", validation)
        atomic_json(out / "manifest.json", dict(schema=V2_SCHEMA, stage="robot", status="RUNNING",
            config=config, provenance=provenance, parent=dict(schema=parent["schema"], stage=parent["stage"],
            step=parent["step"]), parameters=sum(p.numel() for p in model.parameters())))
    if world > 1:
        dist.barrier()
    # One actual-batch wiring/gradient audit before the loop.
    rng = np.random.default_rng(config["seed"] + rank)
    audit_batch = to_device(robot_collate([train[int(i)] for i in train.sample_indices(rng, 1)]), device)
    model.train()
    prediction = model(audit_batch)
    audit_loss, audit_terms = robot_loss(prediction, audit_batch, config, model)
    if not torch.isfinite(audit_loss):
        raise FloatingPointError("nonfinite v2 robot audit loss")
    group_gradients = {}
    for group in groups:
        params = list(group["params"])
        grads = torch.autograd.grad(audit_loss, params, retain_graph=True, allow_unused=True)
        norm = sum(float(g.detach().float().square().sum()) for g in grads if g is not None) ** .5
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("v2 robot objective does not reach optimizer group: " + group["name"])
        group_gradients[group["name"]] = norm
    audit_loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_grad", 1.)))
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError("v2 robot gradient audit failed")
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        atomic_json(out / "engineering_audit.json", dict(loss=float(audit_loss),
            terms={k: float(v) for k, v in audit_terms.items()}, grad_norm=grad_norm,
            group_gradients=group_gradients,
            q_contract=provenance["q_contract"], gpu=gpu_snapshot()))
    del audit_batch, prediction
    # The audit intentionally runs before restoring RNG so resumed batches and
    # stochastic layers match the checkpoint's next-update state.
    if resume:
        states = resume.get("rank_rng") or []
        if len(states) != world:
            raise ValueError("resume checkpoint is missing per-rank RNG state")
        torch.set_rng_state(states[rank]["torch"])
        torch.cuda.set_rng_state(states[rank]["cuda"])
    torch.cuda.empty_cache()
    wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[local], gradient_as_bucket_view=True) if world > 1 else model
    loader = pretrain_loader(train, config["batch_size"], config["seed"], rank, start_step + 1, config["updates"],
                             args.loader_workers, args.loader_prefetch, args.sequence_cache,
                             args.pin_memory, collate_fn=robot_collate)
    batches = iter(loader)
    started = time.monotonic()
    last_resource_check = 0.
    implementation = list((TASK / "src").rglob("*.py")) + list(
        (TASK.parent / "ptv3-shared-wam" / "src").rglob("*.py"))
    code_state = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in implementation}
    curve, files, best = [], [], float("inf")
    for step in range(start_step + 1, config["updates"] + 1):
        batch = to_device(next(batches), device, non_blocking=args.pin_memory)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.get("amp", True)):
            prediction = wrapped(batch)
            loss, terms = robot_loss(prediction, batch, config, model)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite v2 robot loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_grad", 1.))
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite v2 robot gradients")
        optimizer.step()
        now = time.monotonic()
        if now - last_resource_check >= 120 or last_resource_check == 0:
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                raise RuntimeError("less than20GiB free disk")
            pids = set(map(int, subprocess.check_output(
                ["nvidia-smi", "-i", physical, "--query-compute-apps=pid",
                 "--format=csv,noheader"], text=True).split()))
            if pids - {os.getpid()}:
                raise RuntimeError("foreign GPU process appeared")
            if rank == 0 and directory_bytes(out, ROOT / "tmp" / out.name) > config.get("artifact_budget_bytes", 8_000_000_000):
                raise RuntimeError("v2 robot artifact budget exceeded")
            last_resource_check = now
        if step == 1 or step % config.get("log_interval", 20) == 0:
            torch.cuda.synchronize()
            seconds = (time.monotonic() - started) / step
            row = dict(step=step, loss=float(loss), terms={k: float(v) for k, v in terms.items()},
                       elapsed_s=time.monotonic() - started, step_seconds=seconds,
                       eta_s=seconds * (config["updates"] - step), global_batch=config["batch_size"] * world,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, gpu=gpu_snapshot())
            if rank == 0:
                print(json.dumps(row), flush=True)
                atomic_json(out / "progress.json", row)
                with (out / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
        new_best = False
        if step % config.get("validation_interval", 500) == 0 or step == config["updates"]:
            report = evaluate(model, dev, validation, config, device) if rank == 0 else None
            if rank == 0:
                curve.append(dict(step=step, **report))
                atomic_json(out / "curve.json", curve)
                new_best = report["objective"] < best
                if new_best:
                    best = report["objective"]
            if world > 1:
                flag = [new_best]
                dist.broadcast_object_list(flag, src=0)
                new_best = flag[0]
        if new_best or step % config.get("checkpoint_interval", 1000) == 0 or step == config["updates"]:
            rank_rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
            states = [rank_rng]
            if world > 1:
                states = [None] * world
                dist.all_gather_object(states, rank_rng)
            if rank == 0:
                path = out / ("step%d.pt" % step)
                provenance["elapsed_s"] = previous_elapsed + time.monotonic() - started
                save_robot_checkpoint(path, model, optimizer, config, stats, step, provenance, states)
                files.append(path.name)
                checkpoint_alias(out, "latest.pt", path.name)
                if new_best:
                    checkpoint_alias(out, "best.pt", path.name)
                files = prune_recovery_checkpoints(out, files, config)
                if directory_bytes(out, ROOT / "tmp" / out.name) > config.get("artifact_budget_bytes", 8_000_000_000):
                    raise RuntimeError("v2 robot artifact budget exceeded")
                atomic_json(out / "checkpoint_files.json", files)
            if world > 1:
                dist.barrier()
        if time.monotonic() - started + previous_elapsed > config.get("max_seconds", 172800):
            raise TimeoutError("v2 robot run exceeded bounded wall time")
        if any((p.stat().st_size, p.stat().st_mtime_ns) != signature
               for p, signature in code_state.items()):
            raise RuntimeError("implementation changed during training")
    if rank == 0:
        atomic_json(out / "manifest.json", dict(schema=V2_SCHEMA, stage="robot", status="COMPLETED",
            config=config, provenance=provenance, parent=dict(schema=parent["schema"], stage=parent["stage"],
            step=parent["step"]), parameters=sum(p.numel() for p in model.parameters())))
        atomic_json(out / "result.json", dict(status="COMPLETED", curve=curve, checkpoint_files=files,
            scope="engineering_only" if args.smoke else "offline robot fine-tune Probe"))
    if world > 1:
        dist.destroy_process_group()


def _record_failure(error):
    """Leave a machine-readable failure marker when a run aborts."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    try:
        args = parse_args()
        out = args.output.resolve()
        if out.exists():
            payload = {"status": "FAILED", "error": repr(error)}
            manifest = out / "manifest.json"
            if manifest.is_file():
                try:
                    payload = json.loads(manifest.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            payload.update(status="FAILED", error=repr(error))
            atomic_json(manifest, payload)
    except BaseException:
        # Preserve the original training exception; failure marking is best effort.
        pass


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        _record_failure(error)
        raise
