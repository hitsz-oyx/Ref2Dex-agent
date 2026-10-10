"""Fit the ref2 Stage 1 goal-conditioned low-level policy by BC.

This is an offline distillation Probe only.  It accepts native action labels
from the self-owned tracker execution wave, while the old engineering-only
packet contributes only hand/object future goals.  Its actions/q are refused as
labels and enough provenance is recorded to make a later physical run auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OLD = ROOT / "src/task/consequence-evaluator"
sys.path[:0] = [str(TASK / "src"), str(OLD / "src"), str(ROOT)]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_state(gpu: int):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    util, used, total = (int(value.strip()) for value in row.strip().split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return dict(utilization=util, used_mib=used, total_mib=total,
                processes=processes)


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--goal-reference", type=Path, required=True,
                        help="engineering-only packet used for hand/object goals only")
    parser.add_argument("--goal-env", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    args.output = args.output.resolve()
    for path in (args.trajectory, args.trajectory_manifest, args.goal_reference):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")
    if not 1 <= args.steps <= 3000 or args.seconds <= 0 or args.batch_size <= 0:
        raise ValueError("Probe budget is outside the declared bound")

    before = gpu_state(args.gpu)
    if before["utilization"] > 10 or before["used_mib"] > 512:
        raise ValueError("GPU is not idle: " + repr(before))
    initial_pids = {
        int(line.split(",")[0]) for line in before["processes"].splitlines() if line.strip()
    }
    # Keep all temporary compiler/cache material in the project-owned tree.
    scratch = ROOT / "tmp/trajectory-policy"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        TMPDIR=str(scratch),
        TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"),
        TRITON_CACHE_DIR=str(scratch / "triton"),
        OMP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )

    import torch
    from trajectory_policy.goal_policy import ACTIVE_ACTION, INPUT_DIM, GoalConditionedPolicy, Standardizer
    from trajectory_policy.goal_policy_data import build_goal_policy_dataset, split_by_environment

    if not torch.cuda.is_available():
        raise RuntimeError("the BC Probe requires a CUDA-capable Graspenv runtime")
    if torch.__version__ != "2.4.1+cu121":
        raise ValueError("pinned Graspenv Torch 2.4.1+cu121 required")
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    trajectory_manifest = json.loads(args.trajectory_manifest.read_text())
    if trajectory_manifest.get("status") != "COMPLETED":
        raise ValueError("completed tracker execution manifest required")
    if trajectory_manifest.get("schema") != "ref2dex.reference-tracker.v1":
        raise ValueError("unexpected trajectory source schema")
    if trajectory_manifest.get("training_contract") != (
            "on-policy PPO only; no imitation/action-label supervision; fixed final checkpoint"):
        raise ValueError("trajectory source training contract changed")
    roles = trajectory_manifest.get("roles")
    if not isinstance(roles, list) or roles.count("tracker") < 2:
        raise ValueError("tracker role coverage is insufficient")

    source_hashes = {
        str(args.trajectory.resolve()): sha(args.trajectory),
        str(args.trajectory_manifest.resolve()): sha(args.trajectory_manifest),
        str(args.goal_reference.resolve()): sha(args.goal_reference),
        str((TASK / "src/trajectory_policy/goal_policy.py").resolve()):
            sha(TASK / "src/trajectory_policy/goal_policy.py"),
        str((TASK / "src/trajectory_policy/goal_policy_data.py").resolve()):
            sha(TASK / "src/trajectory_policy/goal_policy_data.py"),
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
    }
    declared_goal_hash = next(
        (digest for path, digest in trajectory_manifest.get("input_sha256", {}).items()
         if Path(path).name == args.goal_reference.name),
        None,
    )
    if declared_goal_hash is not None and declared_goal_hash != source_hashes[str(args.goal_reference.resolve())]:
        raise ValueError("goal packet drifted from the tracker manifest")
    with args.goal_reference.open("rb") as stream:
        goal_packet = pickle.load(stream)
    if (goal_packet.get("engineering_only") is not True
            or goal_packet.get("training_allowed") is not False):
        raise ValueError("goal packet must remain explicitly engineering-only and non-training")
    if args.goal_env < 0 or args.goal_env >= int(goal_packet["hand_keypoints"].shape[1]):
        raise ValueError("goal-env outside goal packet")
    reference = dict(
        hand_keypoints=np.asarray(goal_packet["hand_keypoints"][:, args.goal_env], np.float32),
        object_pose=np.asarray(goal_packet["object_pose"][:, args.goal_env], np.float32),
    )
    if reference["hand_keypoints"].shape[1:] != (11, 3) or reference["object_pose"].shape[1:] != (4, 4):
        raise ValueError("goal packet hand/object shape mismatch")
    if not np.isfinite(reference["hand_keypoints"]).all() or not np.isfinite(reference["object_pose"]).all():
        raise ValueError("nonfinite goal packet geometry")
    with np.load(args.trajectory, allow_pickle=False) as packet:
        trajectory = {key: packet[key] for key in packet.files}
    dataset = build_goal_policy_dataset(
        trajectory, reference["hand_keypoints"], reference["object_pose"], roles, role="tracker"
    )
    train_mask, validation_mask = split_by_environment(dataset)
    standardizer = Standardizer.fit(dataset.features[train_mask])
    train_x = torch.from_numpy(standardizer.encode(dataset.features[train_mask])).to(device)
    train_y = torch.from_numpy(dataset.labels[train_mask]).to(device)
    validation_x = torch.from_numpy(standardizer.encode(dataset.features[validation_mask])).to(device)
    validation_y = torch.from_numpy(dataset.labels[validation_mask]).to(device)
    active = torch.as_tensor(ACTIVE_ACTION, device=device)

    policy = GoalConditionedPolicy(width=args.width).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True)
    monitor_path = args.output / "monitor.jsonl"
    started = time.monotonic()
    best = dict(val_active_l1=float("inf"), step=0, state=None)
    log_rows = []

    def log(step, train_loss, val_l1, elapsed):
        current = gpu_state(args.gpu)
        row = dict(step=int(step), train_loss=float(train_loss),
                   val_active_l1=float(val_l1), elapsed_s=float(elapsed),
                   eta_s=float(max(0., (elapsed / max(step, 1)) * (args.steps - step))),
                   gpu=current)
        with monitor_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        log_rows.append(row)
        return current

    policy.train()
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds:
            break
        indices = torch.randint(len(train_x), (min(args.batch_size, len(train_x)),),
                                generator=generator, device=device)
        prediction = policy(train_x[indices])
        loss = torch.nn.functional.smooth_l1_loss(prediction[:, active], train_y[indices][:, active])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.inference_mode():
                value = policy(validation_x)
                val_l1 = (value[:, active] - validation_y[:, active]).abs().mean().item()
            if not np.isfinite(val_l1) or not np.isfinite(loss.item()):
                raise FloatingPointError("nonfinite BC loss")
            if val_l1 < best["val_active_l1"]:
                best = dict(val_active_l1=val_l1, step=step,
                            state={key: value.detach().cpu().clone()
                                   for key, value in policy.state_dict().items()})
            log(step, loss.item(), val_l1, time.monotonic() - started)

    if best["state"] is None:
        raise RuntimeError("BC budget ended before one evaluation")
    policy.load_state_dict(best["state"], strict=True)
    checkpoint = dict(
        schema=GoalConditionedPolicy.schema,
        input_dim=INPUT_DIM,
        width=args.width,
        active_action=list(ACTIVE_ACTION),
        state_dict=policy.state_dict(),
        standardizer=standardizer.as_dict(),
        teacher_source="owned reference-tracker execution; tracker role native action labels",
        goal_source="engineering-only packet hand/object geometry only; packet actions/q discarded",
        goal_packet_authorization="engineering_only=true, training_allowed=false; goal-only use",
        inference_contract="live q/dq/hand/object/velocity + previous native command + 8 future hand/object goals; no future q",
    )
    checkpoint_path = args.output / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    elapsed = time.monotonic() - started
    after = gpu_state(args.gpu)
    foreign = set()
    for line in after["processes"].splitlines():
        if line.strip():
            foreign.add(int(line.split(",")[0]))
    own_pid = os.getpid()
    if (foreign - initial_pids) - {own_pid}:
        raise RuntimeError("unexpected GPU process appeared: " + repr((foreign - initial_pids) - {own_pid}))

    result_status = "PROMISING" if best["val_active_l1"] <= 0.05 else "UNPROMISING"
    result = dict(
        status=result_status,
        offline_only=True,
        train_rows=int(train_mask.sum()), validation_rows=int(validation_mask.sum()),
        train_environments=sorted(set(dataset.env_ids[train_mask].tolist())),
        validation_environments=sorted(set(dataset.env_ids[validation_mask].tolist())),
        updates=int(best["step"]), elapsed_s=float(elapsed),
        best_validation_active_l1=float(best["val_active_l1"]),
        gate=dict(validation_active_l1_threshold=0.05,
                  physical_execution_required=True),
        checkpoint_sha256=sha(checkpoint_path),
        gpu_before=before, gpu_after=after, log_rows=len(log_rows),
    )
    manifest = dict(
        schema="ref2dex.goal-conditioned-bc.v1", status="COMPLETED",
        task="trajectory-policy", run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        seed=args.seed, physical_gpu=args.gpu, input_dim=INPUT_DIM,
        goal_horizon=8, action_contract="native 18-D action; twelve active coordinates",
        teacher_role="tracker", teacher_source=checkpoint["teacher_source"],
        goal_source=checkpoint["goal_source"],
        inference_contract=checkpoint["inference_contract"],
        input_sha256=source_hashes, gpu_before=before, gpu_after=after,
        budget_s=args.seconds, requested_steps=args.steps,
        result_status=result_status,
    )
    write(args.output / "manifest.json", manifest)
    write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
