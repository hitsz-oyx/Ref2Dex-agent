"""Bounded single-GPU / torchrun training with fixed train/dev contracts."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as distributed

from .checkpoint import SCHEMA, atomic_json, directory_bytes, git_commit, load_checkpoint, save_checkpoint
from .contracts import GOAL_ENCODING, TASK_CONDITIONING
from .data import SharedWindows, shared_collate
from .model import SharedWAM
from .loading import pretrain_loader, to_device
from .objectives import forecast_loss, forecast_metrics
from .paths import ROOT, TASK, add_dependencies
from .robot import RobotPair, RobotWindows, human_replay, initialize_retargeter, robot_loss

add_dependencies()
from oakink_wm.multisource import mixed_indices

DEFAULT_DATA = ROOT / "outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree"
DEFAULT_STATS = DEFAULT_DATA.parent / "norm_stats.json"
DEFAULT_PW = DEFAULT_DATA.parent / "training-r1/train-action/final.pt"
OWNER = ROOT / "outputs/trajectory-policy"


def gpu_snapshot():
    if not torch.cuda.is_available():
        return dict(device="cpu")
    return dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                reserved_mib=torch.cuda.memory_reserved() / 2**20,
                nvidia_smi=subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"], text=True).strip())


def parse_args(stage):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK / "configs" / (stage + ".json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--init-pw", type=Path, default=DEFAULT_PW)
    parser.add_argument("--init-shared", type=Path)
    parser.add_argument("--init-r", type=Path)
    parser.add_argument("--source", type=Path, default=OWNER / "competent-pd-teacher-20261011-r2")
    parser.add_argument("--reference", type=Path, default=OWNER / "competent-reference-20261011-r1/reference.npz")
    parser.add_argument("--canonical", type=Path, default=ROOT / "outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=1,
                        help="engineering load check, at most128 samples per rank")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--loader-prefetch", type=int, default=2)
    parser.add_argument("--sequence-cache", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--experiment-card", type=Path)
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
    elif not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("torchrun requires explicit idle CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        raise RuntimeError("real PTv3 training/engineering smoke requires CUDA")
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    physical = visible[local]
    if subprocess.check_output(["nvidia-smi", "-i", physical,
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("assigned GPU occupied before training")
    scratch = ROOT / "tmp" / args.output.name / ("rank%d" % rank)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), TRITON_CACHE_DIR=str(scratch / "triton"),
                      TORCH_EXTENSIONS_DIR=str(ROOT / "tmp/shared-wam-extensions"))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    if world > 1:
        distributed.init_process_group("nccl")
    return rank, world, torch.device("cuda", local)


def tensor_batch(samples, device):
    return to_device(shared_collate(samples), device)


def gradient_audit(model, batch, config, stage):
    """Actual PTv3 gradients, branch isolation and inverse-action anchor semantics."""
    model.eval()
    prediction = model(batch)
    _, losses = forecast_loss(prediction, batch, config)
    forecaster = model if stage == "pretrain" else model.forecaster
    backbone = list(forecaster.scene.backbone.parameters())
    interaction = list(forecaster.interaction.parameters())
    report = {}
    for name, loss in losses.items():
        for group, parameters in (("PTv3", backbone), ("interaction", interaction),
                                  ("goal", list(forecaster.goal.parameters()))):
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            norm = sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(name + " does not reach " + group)
            report[name + "_" + group + "_gradient"] = norm
    # U depends on shared goal context, but never on the hand prediction head.
    branch_grads = torch.autograd.grad(losses["scene"], list(forecaster.hand_future.parameters()), retain_graph=True, allow_unused=True)
    if any(g is not None for g in branch_grads):
        raise ValueError("scene loss has a hand-head dependency")
    report["scene_to_hand_head"] = False
    if stage == "robot":
        _, terms = robot_loss(prediction, batch, config, model.retargeter)
        for name in ("action", "anchor"):
            fg = torch.autograd.grad(terms[name], list(forecaster.parameters()),
                                     retain_graph=True, allow_unused=True)
            rg = torch.autograd.grad(terms[name], list(model.retargeter.parameters()),
                                     retain_graph=True, allow_unused=True)
            fs = sum(float(g.abs().sum()) for g in fg if g is not None)
            rs = sum(float(g.abs().sum()) for g in rg if g is not None)
            if (name == "anchor" and fs != 0) or rs <= 0 or (name == "action" and fs <= 0):
                raise ValueError("joint action/GT anchor gradient contract failed")
            report[name + "_forecaster_gradient"] = fs
            report[name + "_R_gradient"] = rs
    model.train()
    return report


@torch.no_grad()
def evaluate(model, dataset, indices, config, stage, device):
    model.eval()
    totals, objective, batches = {}, 0., 0
    for start in range(0, len(indices), config["batch_size"]):
        batch = tensor_batch([dataset[int(i)] for i in indices[start:start + config["batch_size"]]], device)
        prediction = model(batch)
        loss, _ = forecast_loss(prediction, batch, config) if stage == "pretrain" else robot_loss(
            prediction, batch, config, model.retargeter)
        objective += float(loss) * len(batch["xyz"])
        batches += len(batch["xyz"])
        source = batch["source_id"]
        for sid in source.unique().tolist():
            rows = source == sid
            p = {k: v[rows] for k, v in prediction.items()}
            b = {k: v[rows] for k, v in batch.items()}
            values = forecast_metrics(p, b, config)
            for key, value in values.items():
                slot = totals.setdefault("source%d/%s" % (sid, key), dict(sum_mm=0., count=0))
                slot["sum_mm"] += value["sum_mm"]
                slot["count"] += value["count"]
        if "action" in prediction:
            scale = torch.tensor(config["native_PD_scale"], device=device)
            for name, key in (("pred", "action"), ("GT", "action_anchor")):
                delta = (prediction[key][:, 0].clamp(-1, 1) - batch["native_action"][:, 0]) * scale
                values = dict(PD_xyz_mm=delta[:, :3].norm(dim=-1) * 1000,
                              PD_rotation_rad=delta[:, 3:6].abs().amax(-1),
                              PD_finger_rad=delta[:, 6:].abs().amax(-1),
                              clipping=(prediction[key][:, 0].abs() > 1).any(-1).float())
                for key, value in values.items():
                    totals.setdefault(name + "/" + key, []).extend(value.cpu().tolist())
    result = dict(objective=objective / max(batches, 1), metrics={})
    for key, value in totals.items():
        if isinstance(value, list):
            result["metrics"][key] = dict(mean=float(np.mean(value)), p95=float(np.percentile(value, 95)))
        else:
            result["metrics"][key] = dict(mean_mm=value["sum_mm"] / value["count"] if value["count"] else None,
                                           count=value["count"])
    model.train()
    return result


def run(stage):
    args = parse_args(stage)
    out = args.output.resolve()
    if ROOT / "outputs/ptv3-shared-wam" not in out.parents:
        raise ValueError("output must be inside outputs/ptv3-shared-wam/<run_id>")
    if out.exists() and not args.resume:
        raise FileExistsError("fresh output required; use --resume explicitly")
    if stage == "robot" and not args.init_shared and not args.resume:
        raise ValueError("robot fine-tuning requires --init-shared")
    if directory_bytes(ROOT / "outputs", ROOT / "tmp") >= 400_000_000_000:
        raise RuntimeError("shared400GB storage budget reached")
    config = json.loads(args.config.read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("shared-WAM config required")
    if config.get("goal_encoding") != GOAL_ENCODING:
        raise ValueError("config must specify the current-point/horizon-delta goal contract")
    if config.get("task_conditioning") != TASK_CONDITIONING:
        raise ValueError("config must specify shared scene/hand/goal conditioning")
    if args.steps is not None:
        config["updates"] = args.steps
    experiment = None
    if not args.smoke:
        if args.experiment_card is None:
            raise ValueError("non-smoke training requires a Task-local experiment card")
        import yaml
        card = args.experiment_card.resolve()
        if TASK / "docs/experiments" not in card.parents:
            raise ValueError("Task-local card required")
        experiment = yaml.safe_load(card.read_text().split("---", 2)[1])
        if experiment.get("task") != TASK.name or not experiment.get("seeds") or not experiment.get("experiment_id"):
            raise ValueError("card task and assigned seeds required")
        config["seed"] = int(experiment["seeds"][0])
    else:
        if not 1 <= args.smoke_batch_size <= 128:
            raise ValueError("engineering batch size must be in[1,128]")
        config.update(updates=min(config["updates"], 4), batch_size=args.smoke_batch_size, validation_samples=3,
                      validation_interval=min(config["validation_interval"], 4), max_seconds=600,
                      log_interval=1)
    if config["updates"] < 1 or config["batch_size"] < 1 or not 0 <= config.get("replay_fraction", 0.) < 1:
        raise ValueError("positive update/batch count and replay fraction in[0,1) required")
    rank, world, device = setup(args)
    torch.manual_seed(config["seed"] + rank)
    stats = json.loads(args.stats.read_text())
    provenance = dict(git_commit=git_commit(), stage=stage, engineering_only=args.smoke,
                      experiment_id=experiment["experiment_id"] if experiment else None,
                      stats=str(args.stats.resolve()), data=str(args.data.resolve()), world_size=world,
                      execution=dict(loader_workers=args.loader_workers, loader_prefetch=args.loader_prefetch,
                                     sequence_cache=args.sequence_cache, pin_memory=args.pin_memory))
    if stage == "pretrain":
        train = SharedWindows(args.data, "train", config["object_points"])
        dev = SharedWindows(args.data, "val", config["object_points"])
        if (stats.get("split") != "train" or stats.get("sources") != ["oakink2", "grab", "arctic"]
                or stats.get("normalization_source_manifest_sha256") != train.base.meta.get("normalization_source_manifest_sha256")):
            raise ValueError("train-only matching three-domain statistics required")
        model = SharedWAM(stats, config)
        if not args.resume:
            from goal_point_wam.pretrained import initialize_pw, file_digest
            provenance["PW_transfer"] = initialize_pw(model, args.init_pw, config, file_digest(args.stats))
        groups = model.parameter_groups(config["scene_lr"], config["new_lr"])
        validation = mixed_indices(dev.base, config["validation_samples"],
            config.get("validation_seed", config["seed"] + 1), equal_sources=True)
        human = None
    else:
        train = RobotWindows(args.source, args.reference, args.canonical, "train", config["object_points"])
        dev = RobotWindows(args.source, args.reference, args.canonical, "val", config["object_points"])
        parent = load_checkpoint(args.resume or args.init_shared)
        if not args.smoke and parent["provenance"].get("engineering_only"):
            raise ValueError("engineering checkpoints cannot silently initialize a scientific robot run")
        if parent["stats"] != stats:
            raise ValueError("robot input normalization differs from pretraining")
        architecture = ("hidden", "horizon", "patch_size", "scene_tokens", "interaction_layers", "voxel_m", "voxel_origin_m", "object_points", "goal_encoding", "task_conditioning")
        if any(parent["config"].get(k) != config.get(k) for k in architecture):
            raise ValueError("robot stage changes the pretraining architecture")
        model = RobotPair(SharedWAM(stats, config))
        if not args.resume:
            if parent["stage"] != "pretrain":
                raise ValueError("fresh robot stage starts from shared pretraining")
            model.forecaster.load_state_dict(parent["model"], strict=True)
            if args.init_r:
                from trajectory_policy.chunk_retargeter import load_chunk
                rpacket = torch.load(args.init_r, map_location="cpu", weights_only=False)
                if rpacket["native_controller"] != train.controller:
                    raise ValueError("R initialization and robot controller differ")
                model.retargeter = load_chunk(rpacket, "cpu")
            else:
                initialize_retargeter(model.retargeter, train)
        config["native_PD_scale"] = np.asarray(train.controller["scale"])[list(__import__(
            "trajectory_policy.retargeter_data", fromlist=["ACTIVE"]).ACTIVE)].tolist()
        config["native_PD_scale"][6:] = [v / 2 for v in config["native_PD_scale"][6:]]
        config["native_controller"] = train.controller
        provenance.update(source=str(args.source.resolve()), reference=str(args.reference.resolve()),
                          canonical=str(args.canonical.resolve()), init_shared=str(args.init_shared.resolve()) if args.init_shared else None)
        groups = model.parameter_groups(config)
        rng = np.random.default_rng(config["seed"] + 1)
        validation = rng.choice(len(dev), min(len(dev), config["validation_samples"]), replace=False)
        human = SharedWindows(args.data, "train", config["object_points"]) if config.get("replay_fraction", 0.) else None
    model.to(device)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", .01))
    start_step = 0
    if args.resume:
        state = load_checkpoint(args.resume)
        if state["stage"] != stage or state["config"] != config or state["stats"] != stats:
            raise ValueError("resume stage/config/statistics differ")
        for key in ("data", "source", "reference", "canonical", "world_size"):
            if state["provenance"].get(key) != provenance.get(key):
                raise ValueError("resume data/resource contract differs: " + key)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        if "PW_transfer" in state["provenance"]:
            provenance["PW_transfer"] = state["provenance"]["PW_transfer"]
        provenance["resume_parent"] = dict(checkpoint=str(args.resume.resolve()),
            git_commit=state["provenance"].get("git_commit"), step=start_step)
        rng = state["rank_rng"][rank]
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state(rng["cuda"].cpu())
    if rank == 0:
        out.mkdir(parents=True, exist_ok=args.resume is not None)
        np.save(out / "validation.npy", validation)
    if world > 1:
        distributed.barrier()
    workers = [os.getpid()]
    if world > 1:
        workers = [None] * world
        distributed.all_gather_object(workers, os.getpid())
    manifest = dict(schema=SCHEMA, status="RUNNING", task=TASK.name, stage=stage, config=config,
                    provenance=provenance, parameters=sum(p.numel() for p in model.parameters()),
                    worker_pids=workers, training_allowed=not args.smoke)
    if rank == 0:
        atomic_json(out / "manifest.json", manifest)
    started = time.monotonic()
    best, best_step, curve = float("inf"), None, []
    if args.resume and (out / "curve.json").exists():
        curve = json.loads((out / "curve.json").read_text())
        previous = min(curve, key=lambda row: row["objective"])
        best, best_step = previous["objective"], previous["step"]
    implementation = list((TASK / "src").rglob("*.py")) + [Path(__file__)]
    code_state = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in implementation}
    batch_size = config["batch_size"]
    def draw(step):
        seed = int(np.random.SeedSequence([config["seed"], step, rank]).generate_state(1)[0])
        if stage == "pretrain":
            return [train[int(i)] for i in mixed_indices(train.base, batch_size, seed)]
        rng = np.random.default_rng(seed)
        ticks = np.asarray([key[0] for key in train.keys])
        startup, later = np.flatnonzero(ticks < 16), np.flatnonzero(ticks >= 16)
        indices = np.r_[rng.choice(startup, batch_size // 2), rng.choice(later, batch_size - batch_size // 2)]
        return [train[int(i)] for i in indices]
    # Audit the same wiring on a small actual batch; full FP32 retained graphs
    # need not consume the memory budget of a large bf16 optimizer batch.
    audit_batch = tensor_batch(draw(start_step + 1)[:1], device)
    audit = gradient_audit(model, audit_batch, config, stage)
    del audit_batch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if rank == 0:
        atomic_json(out / "engineering_audit.json", audit)
    model.train()
    wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
        find_unused_parameters=stage == "robot") if world > 1 else model
    checkpoint_interval = int(config.get("checkpoint_interval", config["validation_interval"]))
    if checkpoint_interval < 1:
        raise ValueError("positive checkpoint interval required")
    checkpoint_files = []
    if args.resume and (out / "checkpoint_files.json").exists():
        checkpoint_files = json.loads((out / "checkpoint_files.json").read_text())
    loader = None
    if args.loader_workers:
        if stage != "pretrain":
            raise ValueError("CPU prefetch currently supports pretraining only")
        loader = pretrain_loader(train, batch_size, config["seed"], rank, start_step + 1,
            config["updates"], args.loader_workers, args.loader_prefetch, args.sequence_cache,
            args.pin_memory)
    batches = iter(loader) if loader is not None else None
    logged_at, logged_step = time.monotonic(), start_step
    data_wait_total = 0.
    try:
        for step in range(start_step + 1, config["updates"] + 1):
            data_started = time.perf_counter()
            cpu_batch = next(batches) if batches is not None else shared_collate(draw(step))
            data_wait_total += time.perf_counter() - data_started
            batch = to_device(cpu_batch, device, non_blocking=args.pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.get("amp", True)):
                prediction = wrapped(batch)
                loss, terms = forecast_loss(prediction, batch, config) if stage == "pretrain" else robot_loss(
                    prediction, batch, config, model.retargeter)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            fraction = config.get("replay_fraction", 0.) if human is not None else 0.
            ((1. - fraction) * loss).backward()
            if fraction:
                ids = mixed_indices(human.base, batch_size, config["seed"] + step * world + rank)
                hb = tensor_batch([human_replay(human[int(i)]) for i in ids], device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.get("amp", True)):
                    hl, _ = forecast_loss(wrapped(hb), hb, config)
                (fraction * hl).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_grad", 1.))
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step()
            if step == start_step + 1 or step % config.get("log_interval", 20) == 0:
                elapsed = time.monotonic() - started
                interval = time.monotonic() - logged_at
                step_seconds = interval / (step - logged_step)
                data_wait_seconds = data_wait_total / (step - logged_step)
                data_wait_total = 0.
                logged_at, logged_step = time.monotonic(), step
                row = dict(step=step, loss=float(loss.detach()), terms={k: float(v.detach()) for k, v in terms.items()},
                           elapsed_s=elapsed, eta_s=step_seconds * (config["updates"] - step),
                           step_seconds=step_seconds, data_wait_seconds=data_wait_seconds, global_batch=batch_size * world,
                           samples_per_second=batch_size * world / max(step_seconds, 1e-8),
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           gpu=gpu_snapshot())
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    atomic_json(out / "progress.json", row)
                    with (out / "train.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                physical = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[device.index]
                pids = set(map(int, subprocess.check_output(["nvidia-smi", "-i", physical,
                    "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()))
                if pids - set(workers):
                    raise RuntimeError("foreign process on assigned GPU")
            if step % config["validation_interval"] == 0 or step == config["updates"]:
                report = evaluate(model, dev, validation, config, stage, device) if rank == 0 else None
                if rank == 0:
                    curve.append(dict(step=step, **report))
                    atomic_json(out / "curve.json", curve)
                    if report["objective"] < best:
                        best, best_step = report["objective"], step
                if world > 1:
                    distributed.barrier()
            save_due = (step % checkpoint_interval == 0 or step == config["updates"]
                        or (step == 1 and config.get("save_first_step", False)))
            if save_due:
                rank_rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
                states = [rank_rng]
                if world > 1:
                    states = [None] * world
                    distributed.all_gather_object(states, rank_rng)
                if rank == 0:
                    path = out / ("step%d.pt" % step)
                    save_checkpoint(path, model, optimizer, config, stats, step, stage, provenance, states)
                    checkpoint_files.append(path.name)
                    from .checkpoint import prune_recovery_checkpoints
                    checkpoint_files = prune_recovery_checkpoints(out, checkpoint_files, config)
                    atomic_json(out / "checkpoint_files.json", checkpoint_files)
                if world > 1:
                    distributed.barrier()
                if directory_bytes(out, ROOT / "tmp" / out.name) > 4_000_000_000:
                    raise RuntimeError("run artifact budget4GB reached")
                if directory_bytes(ROOT / "outputs", ROOT / "tmp") >= 400_000_000_000:
                    raise RuntimeError("shared400GB storage budget reached")
            if any((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != signature
                   for path, signature in code_state.items()):
                raise RuntimeError("implementation changed during this run")
            if time.monotonic() - started > config["max_seconds"]:
                raise TimeoutError("bounded training deadline")
        if rank == 0:
            manifest["status"] = "COMPLETED"
            atomic_json(out / "manifest.json", manifest)
            atomic_json(out / "result.json", dict(status="COMPLETED", best_step=best_step,
                final_checkpoint=str(out / ("step%d.pt" % config["updates"])), curve=curve,
                engineering_audit=audit, elapsed_s=time.monotonic() - started,
                scope="engineering_only" if args.smoke else "offline Probe; autonomous grasp untested"))
    except BaseException as error:
        if rank == 0:
            manifest.update(status="FAILED", error=repr(error))
            atomic_json(out / "manifest.json", manifest)
        raise
    finally:
        if world > 1:
            distributed.destroy_process_group()
