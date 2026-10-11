#!/usr/bin/env python3
"""Train the no-video Goal-Point-WAM three-domain Probe on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


TASK = Path(__file__).resolve().parents[2]
CM_TASK = TASK.parent / "cm-pointflow-effect-pretrain"
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(CM_TASK / "src"))

from goal_point_wam.data import GoalWindows, goal_collate  # noqa: E402
from goal_point_wam.pretrained import initialize_pw
from goal_point_wam.model import GoalPointWAM  # noqa: E402
from oakink_wm.model import metrics, object_motion_masks  # noqa: E402
from oakink_wm.multisource import MAIN_SOURCE_NAMES, mixed_indices  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for parameter in model.parameters():
        h.update(parameter.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loader(dataset, indices: np.ndarray, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False,
        collate_fn=goal_collate, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )


def gpu_snapshot(device: torch.device) -> dict:
    if device.type != "cuda":
        return dict(device="cpu")
    index = torch.cuda.current_device()
    result = dict(device=f"cuda:{index}", allocated_bytes=torch.cuda.memory_allocated(index),
                  reserved_bytes=torch.cuda.memory_reserved(index))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical = visible[index].strip() if visible[0] else str(index)
    result["physical_gpu"] = physical
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "-i", str(physical), "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        util, used, total = [int(piece.strip()) for piece in text.split(",")]
        result.update(utilization_percent=util, memory_used_mib=used, memory_total_mib=total)
    except (OSError, subprocess.SubprocessError, ValueError):
        result["query_error"] = True
    return result


def checkpoint(path: Path, model, optimizer, step: int, config: dict, identity: dict, best: float):
    state = dict(checkpoint_kind="goal-point-wam.single.v1", model=model.state_dict(),
                 optimizer=optimizer.state_dict(), step=step, config=config, identity=identity,
                 best=best, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                 numpy_rng=np.random.get_state(), python_rng=random.getstate())
    part = path.with_suffix(".pt.part")
    torch.save(state, part)
    part.replace(path)


@torch.no_grad()
def evaluate(model, dataset: GoalWindows, indices: np.ndarray, batch_size: int,
             workers: int, device: torch.device) -> dict:
    model.eval()
    names = tuple(dataset.source_names)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    trajectory_sum = {name: 0.0 for name in names}
    trajectory_count = {name: 0 for name in names}
    world_sum = {name: 0.0 for name in names}
    world_count = {name: 0 for name in names}
    moving_sum = {name: 0.0 for name in names}
    moving_count = {name: 0 for name in names}
    static_sum = {name: 0.0 for name in names}
    static_count = {name: 0 for name in names}
    for cpu_batch in loader(dataset, indices, batch_size, workers):
        batch = move_batch(cpu_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = model(batch)
        trajectory_error = torch.linalg.vector_norm(
            prediction["trajectory"] - batch["trajectory_target"], dim=-1
        )
        trajectory_mask = batch["trajectory_valid"]
        world_metrics = metrics(prediction, batch)["point_epe"]
        moving, static = object_motion_masks(batch)
        source_ids = batch["source_id"].tolist()
        for row, source_id in enumerate(source_ids):
            name = names[int(source_id)]
            tmask = trajectory_mask[row]
            trajectory_sum[name] += float(trajectory_error[row][tmask].sum().item())
            trajectory_count[name] += int(tmask.sum().item())
            valid_objects = batch["object_valid"][row]
            values = world_metrics[row, :, -1]
            world_sum[name] += float(values[valid_objects].sum().item())
            world_count[name] += int(valid_objects.sum().item())
            for mask, sums, counts_for_mask in ((moving[row], moving_sum, moving_count),
                                                  (static[row], static_sum, static_count)):
                sums[name] += float(values[mask].sum().item())
                counts_for_mask[name] += int(mask.sum().item())
    model.train()
    result = {}
    for name in names:
        result[f"source/{name}/trajectory_epe_m"] = trajectory_sum[name] / trajectory_count[name] if trajectory_count[name] else None
        result[f"source/{name}/world_point_epe_m_h24"] = world_sum[name] / world_count[name] if world_count[name] else None
        result[f"source/{name}/moving_world_point_epe_m_h24"] = moving_sum[name] / moving_count[name] if moving_count[name] else None
        result[f"source/{name}/static_world_point_epe_m_h24"] = static_sum[name] / static_count[name] if static_count[name] else None
        result[f"source/{name}/counts"] = dict(trajectory_points=trajectory_count[name],
            objects=world_count[name], moving_objects=moving_count[name], static_objects=static_count[name])
    valid_world = [result[f"source/{name}/world_point_epe_m_h24"] for name in names if world_count[name]]
    result["macro/world_point_epe_m_h24"] = float(np.mean(valid_world))
    result["macro/trajectory_epe_m"] = float(np.mean([result[f"source/{name}/trajectory_epe_m"] for name in names if trajectory_count[name]]))
    return result


def goal_isolation(model, batch: dict[str, torch.Tensor]) -> float:
    """Return the maximum world-output change after replacing only the goal."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        original = model(batch)
    changed = {key: value.clone() for key, value in batch.items()}
    if model.goal_type == "object_points":
        changed["goal_object_points"] = changed["goal_object_points"].flip(1) + .07
    else:
        changed["goal_hand"] = torch.flip(changed["goal_hand"], dims=(1, 2)) + .07
        changed["goal_hand_valid"] = batch["goal_hand_valid"].clone()
    with torch.no_grad():
        altered = model(changed)
    difference = max(
        (original["translation"] - altered["translation"]).abs().max().item(),
        (original["rotation"] - altered["rotation"]).abs().max().item(),
    )
    if was_training:
        model.train()
    return float(difference)


def engineering_audit(model, batch, saved):
    """Real-batch goal isolation, branch gradients and exact checkpoint replay."""
    model.eval()
    prediction = model(batch)
    shared = model.scene.scene_proj[0].weight
    trajectory = model.trajectory_loss(prediction["trajectory"], batch)
    world, _ = model.world_loss(prediction, batch)
    gradients = {}
    for name, loss in (("trajectory", trajectory), ("world", world)):
        grad, = torch.autograd.grad(loss, shared, retain_graph=True)
        norm = float(grad.norm().item())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(name + " loss does not reach shared scene projection")
        gradients[name + "_scene_gradient_norm"] = norm
    reference = {key: value.detach().clone() for key, value in prediction.items()}
    del prediction, trajectory, world
    state = torch.load(saved, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    with torch.no_grad():
        replay = model(batch)
    replay_max = max(float((reference[key] - replay[key]).abs().max()) for key in reference)
    isolation = goal_isolation(model, batch)
    if replay_max > 1e-5 or isolation > 1e-5:
        raise RuntimeError("checkpoint replay or goal isolation failed")
    return dict(**gradients, checkpoint_replay_max=replay_max, goal_world_isolation_max=isolation)


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    if config.get("schema") != "goal-point-wam.three-domain-no-video.v1":
        raise ValueError("unexpected Goal-Point-WAM config")
    if args.steps is not None:
        config["updates"] = int(args.steps)
    if args.smoke:
        config.update(microbatch=1, accumulation=1, workers=0, validation_samples=12,
                      validation_interval=2, checkpoint_interval=2, updates=min(config["updates"], 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("GPU required by this run")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "input_manifest.json").exists() and not args.resume:
        raise FileExistsError("existing output requires --resume or a new run directory")
    atomic_json(output / "config.json", config)
    train_base = __import__("oakink_wm.multisource", fromlist=["MixedWindows"]).MixedWindows(args.data, "train")
    val_base = __import__("oakink_wm.multisource", fromlist=["MixedWindows"]).MixedWindows(args.data, "val")
    names = tuple(d["name"] for d in train_base.meta["sources"])
    if names != MAIN_SOURCE_NAMES or tuple(d["name"] for d in val_base.meta["sources"]) != names:
        raise ValueError(f"expected main three source manifest, got {names}")
    if train_base.meta.get("supervision") != "measured_dynamic_hand_main":
        raise ValueError("video/auxiliary source is not allowed in this run")
    goal_args = dict(goal_type=config.get("goal_type", "hand_endpoint"),
                     goal_points=int(config.get("goal_points", 32)))
    train, val = GoalWindows(train_base, **goal_args), GoalWindows(val_base, **goal_args)
    stats = json.loads(args.stats.read_text())
    if stats.get("sources") != list(names) or stats.get("split") != "train":
        raise ValueError("normalization stats do not describe OakInk2/GRAB/ARCTIC train data")
    if stats.get("normalization_source_manifest_sha256") != train_base.meta.get("normalization_source_manifest_sha256"):
        raise ValueError("normalization/source identity mismatch")
    data_hash = digest(args.data / "processed/manifest.json")
    identity = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    data_manifest_sha256=data_hash, stats_sha256=digest(args.stats), sources=names,
                    config_sha256=digest(args.config), script_sha256=digest(Path(__file__)),
                    model_parameters=None, device=str(device), no_video=True)
    model = GoalPointWAM(stats, config)
    if args.init_pw:
        identity["pretrained"] = initialize_pw(model, args.init_pw, config, identity["stats_sha256"])
    elif not args.resume and not args.random_init:
        raise ValueError("fresh runs require --init-pw; --random-init is an explicit control")
    identity["model_sha256"] = digest(TASK / "src/goal_point_wam/model.py")
    identity["pretrained_loader_sha256"] = digest(TASK / "src/goal_point_wam/pretrained.py")
    identity["initial_parameter_sha256"] = parameter_hash(model)
    model = model.to(device)
    identity["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    step, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if (state["identity"].get("data_manifest_sha256") != data_hash or state["config"] != config
                or state["identity"].get("model_sha256") != identity["model_sha256"]
                or state["identity"].get("stats_sha256") != identity["stats_sha256"]):
            raise ValueError("resume identity/config mismatch")
        identity = state["identity"]
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step, best = int(state["step"]), float(state["best"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda": torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
    atomic_json(output / "input_manifest.json", identity)
    train_indices = mixed_indices(train_base, config["updates"] * config["microbatch"] * config["accumulation"],
                                   int(config["seed"]) + 1)
    validation = mixed_indices(val_base, config["validation_samples"], int(config["validation_seed"]),
                               equal_sources=True)
    np.save(output / "validation_balanced.npy", validation)
    batches = iter(loader(train, train_indices[step * config["microbatch"] * config["accumulation"]:],
                          config["microbatch"], int(config["workers"])))
    started = time.time()
    deadline = args.deadline or started + float(config["group_seconds"])
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop_requested.__setitem__(0, True))
    log = (output / "train.jsonl").open("a")
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < config["updates"] and time.time() < deadline and not stop_requested[0]:
            update_start = time.monotonic()
            if step < config["warmup_updates"]:
                scale = (step + 1) / config["warmup_updates"]
            else:
                scale = .1 + .9 * .5 * (1 + math.cos(math.pi * (step - config["warmup_updates"]) /
                                                       max(1, config["updates"] - config["warmup_updates"])))
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * scale
            loss_value = 0.0
            terms = {}
            for _ in range(config["accumulation"]):
                batch = move_batch(next(batches), device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and config["amp"]):
                    prediction = model(batch)
                loss, terms = model.loss(prediction, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Goal-Point-WAM loss")
                (loss / config["accumulation"]).backward()
                loss_value += float(loss.detach().cpu()) / config["accumulation"]
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            row = dict(step=step, loss=loss_value, gradient_norm=float(grad),
                       forward_training=model.training,
                       seconds=time.monotonic() - update_start, elapsed_seconds=time.time() - started,
                       learning_rate=optimizer.param_groups[0]["lr"], gpu=gpu_snapshot(device))
            row.update({key: float(value) for key, value in terms.items()})
            log.write(json.dumps(row) + "\n"); log.flush()
            if step == 1 or step % 10 == 0 or args.smoke:
                print(json.dumps(row), flush=True)
            atomic_json(output / "progress.json", dict(status="RUNNING", **row))
            if step % int(config["validation_interval"]) == 0 or step == config["updates"]:
                measured = evaluate(model, val, validation, int(config["microbatch"]),
                                    int(config["workers"]), device)
                atomic_json(output / "validation_latest.json", dict(step=step, metrics=measured))
                score = measured["macro/world_point_epe_m_h24"] + measured["macro/trajectory_epe_m"]
                if score < best:
                    best = score
                    checkpoint(output / "best.pt", model, optimizer, step, config, identity, best)
            if step % int(config["checkpoint_interval"]) == 0:
                checkpoint(output / "latest.pt", model, optimizer, step, config, identity, best)
        checkpoint(output / "latest.pt", model, optimizer, step, config, identity, best)
        checkpoint(output / "final.pt", model, optimizer, step, config, identity, best)
        final = dict(status="COMPLETED" if step == config["updates"] else "BUDGET_STOP", step=step,
                     elapsed_seconds=time.time() - started, best_score=best if math.isfinite(best) else None,
                     goal_world_isolation_max=goal_isolation(model, batch),
                     gpu=gpu_snapshot(device), engineering_only=args.smoke)
        if step:
            final["validation"] = json.loads((output / "validation_latest.json").read_text()).get("metrics", {})
        if args.smoke:
            final["engineering_audit"] = engineering_audit(model, batch, output / "final.pt")
        atomic_json(output / "result.json", final)
        atomic_json(output / "progress.json", final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        atomic_json(output / "progress.json", dict(status="FAILED", step=step, error=repr(error),
                                                     gpu=gpu_snapshot(device)))
        raise
    finally:
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=TASK / "configs/three_domain_no_video_probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--deadline", type=float)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--init-pw", type=Path)
    initialization.add_argument("--random-init", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
