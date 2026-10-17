"""Fit a direct measured-state + GT-τ executor by behavior cloning.

This is the first probe in the direct-executor route.  Labels are the command
that was actually applied by DExplore, restricted to the twelve active Inspire
coordinates.  The old PD retargeter is not imported or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_state(gpu: int):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    util, used, total = (int(value.strip()) for value in row.strip().split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return dict(utilization=util, used_mib=used, total_mib=total,
                processes=processes)


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--trajectory-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unseen-trajectory", type=Path,
                   help="optional corrected new-sequence teacher trajectory for offline audit")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--idle-context-pid", type=int, action="append", default=[])
    p.add_argument("--seed", type=int, default=417)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--seconds", type=float, default=600.)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--oracle-c", action="store_true",
                   help="append current GT interaction context C for the oracle comparison")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def goal_features(goal):
    goal = np.asarray(goal, dtype=np.float32)
    if goal.ndim == 3 and goal.shape[-2:] == (4, 4):
        goal = goal[:, :3, :].reshape(len(goal), 12)
    if goal.ndim != 2 or goal.shape[1] != 12 or not np.isfinite(goal).all():
        raise ValueError("goal must be finite [env, 4, 4] or [env, 12]")
    return goal


def contact_features(packet, ticks, env):
    pair = packet["surface_gap"][ticks, env] <= .01
    return np.concatenate((pair[:, None].astype(np.float32),
        packet["surface_gap"][ticks, env, None].astype(np.float32),
        packet["support_gap"][ticks, env, None].astype(np.float32),
        packet["velocity"][ticks, env].astype(np.float32),
        packet["table_footprint"][ticks, env, None].astype(np.float32)), axis=1)


def dataset(packet, env_ids, horizon, oracle_c=False):
    from trajectory_policy.retargeter_data import ACTIVE, features
    q, dq, hand = packet["q"], packet["dq"], packet["hand"]
    obj, velocity = packet["obj"], packet["velocity"]
    action, applied = packet["action"], packet["applied"]
    if not np.array_equal(action, applied):
        raise ValueError("BC labels must equal actually applied native commands")
    controls, envs = action.shape[:2]
    if q.shape[0] != controls + 1 or len(env_ids) < 1:
        raise ValueError("invalid state/action time alignment")
    if max(env_ids) >= envs or controls <= horizon:
        raise ValueError("invalid environment or insufficient control horizon")
    ticks = np.arange(controls - horizon, dtype=np.int64)
    rows = []
    for env in env_ids:
        future = hand[ticks[:, None] + 1 + np.arange(horizon)[None, :], env]
        state, tau = features(q[ticks, env], dq[ticks, env], hand[ticks, env],
                              obj[ticks, env], velocity[ticks, env], future)
        previous = np.zeros((len(ticks), len(ACTIVE)), dtype=np.float32)
        if len(ticks) > 1:
            previous[1:] = applied[ticks[:-1], env][:, list(ACTIVE)]
        row = (state, tau, np.broadcast_to(goal_features(packet["goal"])[env],
                                                  (len(ticks), 12)).copy(), previous,
                     applied[ticks, env][:, list(ACTIVE)].astype(np.float32),
                     np.full(len(ticks), env, dtype=np.int64), ticks.copy())
        if oracle_c:
            row = row + (contact_features(packet, ticks, env),)
        rows.append(row)
    base_rows = [row[:7] for row in rows]
    state, tau, goal, previous, label, env, tick = [np.concatenate(v, axis=0)
        for v in zip(*base_rows)]
    result = dict(state=state, tau=tau, goal=goal, previous=previous,
                  label=label, env=env, tick=tick)
    if oracle_c:
        result["contact"] = np.concatenate([r[-1] for r in rows], axis=0)
    return result


def standardize(train):
    # State and τ have different physical units; per-feature statistics keep
    # the small hand displacements numerically visible to the MLP.
    values = (train["state"], train["tau"].reshape(len(train["tau"]), -1),
              train["goal"], train["previous"])
    if "contact" in train:
        values = values + (train["contact"],)
    x = np.concatenate(values, axis=1)
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-4] = 1.
    return mean, scale


def encode(data, mean, scale):
    state_dim = data["state"].shape[1]
    tau_dim = data["tau"].shape[1] * data["tau"].shape[2]
    goal_dim = data["goal"].shape[1]
    values = (data["state"], data["tau"].reshape(len(data["tau"]), -1),
              data["goal"], data["previous"])
    if "contact" in data:
        values = values + (data["contact"],)
    x = np.concatenate(values, axis=1)
    offset = 0
    state = (x[:, offset:offset + state_dim] - mean[offset:offset + state_dim]) / scale[offset:offset + state_dim]
    offset += state_dim
    tau = (x[:, offset:offset + tau_dim] - mean[offset:offset + tau_dim]) / scale[offset:offset + tau_dim]
    offset += tau_dim
    goal = (x[:, offset:offset + goal_dim] - mean[offset:offset + goal_dim]) / scale[offset:offset + goal_dim]
    offset += goal_dim
    previous = (x[:, offset:offset + 12] - mean[offset:offset + 12]) / scale[offset:offset + 12]
    offset += 12
    result = [state.astype(np.float32), tau.reshape(len(tau), -1, 33).astype(np.float32),
              goal.astype(np.float32), previous.astype(np.float32)]
    if "contact" in data:
        result.append(((x[:, offset:] - mean[offset:]) / scale[offset:]).astype(np.float32))
    return tuple(result)


def evaluate(policy, data, mean, scale, device, torch):
    encoded = encode(data, mean, scale)
    state, tau, goal, previous = encoded[:4]
    with torch.inference_mode():
        inputs = [torch.from_numpy(x).to(device) for x in (state, tau, goal, previous)]
        if "contact" in data:
            inputs.append(torch.from_numpy(encoded[4]).to(device))
        pred = policy(*inputs).cpu().numpy()
    error = pred - data["label"]
    return dict(active_l1=float(np.abs(error).mean()), active_rmse=float(np.sqrt(np.mean(error ** 2))),
                max_abs=float(np.abs(error).max()), mean_action_abs=float(np.abs(pred).mean()))


def main():
    args = parse()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")
    if not args.trajectory.exists() or not args.trajectory_manifest.exists():
        raise FileNotFoundError("trajectory and completed manifest are required")
    before = gpu_state(args.gpu)
    allowed = set(args.idle_context_pid)
    if before["utilization"] > 15 or before["used_mib"] > 1024:
        raise RuntimeError("GPU is occupied: " + repr(before))
    scratch = ROOT / "tmp/trajectory-policy/direct-tau-bc"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"),
                      OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    import torch
    from trajectory_policy.tau_executor import DirectTauExecutorOracleCPolicy, DirectTauExecutorPolicy
    if not torch.cuda.is_available() or torch.__version__ != "2.0.1+cu118":
        raise RuntimeError("direct BC requires the pinned DExplore Torch 2.0.1+cu118 runtime")
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    manifest = json.loads(args.trajectory_manifest.read_text())
    if manifest.get("status") != "COMPLETED" or manifest.get("training_allowed") is not True:
        raise ValueError("completed training-eligible teacher rollout required")
    with np.load(args.trajectory, allow_pickle=False) as f:
        packet = {k: f[k].copy() for k in f.files}
    envs = packet["action"].shape[1]
    train = dataset(packet, list(range(0, min(16, envs))), args.horizon, args.oracle_c)
    validation = dataset(packet, list(range(min(16, envs), min(24, envs))), args.horizon, args.oracle_c)
    test = dataset(packet, list(range(min(24, envs), envs)), args.horizon, args.oracle_c)
    mean, scale = standardize(train)
    train_encoded = encode(train, mean, scale)
    val_encoded = encode(validation, mean, scale)
    train_t = [torch.from_numpy(x).to(device) for x in train_encoded]
    val_t = [torch.from_numpy(x).to(device) for x in val_encoded]
    train_y = torch.from_numpy(train["label"]).to(device)
    val_y = torch.from_numpy(validation["label"]).to(device)
    Policy = DirectTauExecutorOracleCPolicy if args.oracle_c else DirectTauExecutorPolicy
    policy = Policy(horizon=args.horizon, width=args.width).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True)
    monitor_path = args.output / "monitor.jsonl"
    started = time.monotonic()
    best = None
    rows = []
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds:
            break
        ids = torch.randint(len(train_y), (min(args.batch_size, len(train_y)),),
                            generator=generator, device=device)
        pred = policy(*(x[ids] for x in train_t))
        loss = torch.nn.functional.smooth_l1_loss(pred, train_y[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.inference_mode():
                val_pred = policy(*val_t)
                val_l1 = (val_pred - val_y).abs().mean().item()
            current = gpu_state(args.gpu)
            elapsed = time.monotonic() - started
            row = dict(step=step, train_loss=float(loss.item()), val_active_l1=float(val_l1),
                       elapsed_s=elapsed, eta_s=max(0., elapsed / step * (args.steps - step)),
                       gpu=current)
            rows.append(row)
            with monitor_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if best is None or val_l1 < best["val_active_l1"]:
                best = dict(val_active_l1=float(val_l1), step=step,
                            state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
    if best is None:
        raise RuntimeError("BC budget ended before validation")
    policy.load_state_dict(best["state"])
    unseen_result = None
    if args.unseen_trajectory:
        with np.load(args.unseen_trajectory, allow_pickle=False) as f:
            unseen_packet = {k: f[k].copy() for k in f.files}
        unseen = dataset(unseen_packet, list(range(unseen_packet["action"].shape[1])), args.horizon, args.oracle_c)
        unseen_result = evaluate(policy, unseen, mean, scale, device, torch)
    checkpoint = dict(schema=Policy.schema, horizon=args.horizon,
                      width=args.width, state_dict=policy.state_dict(), mean=mean, scale=scale,
                      active_action=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15],
                      inference_contract=("live measured state87 + GT tau future hand8 + task goal pose12 + previous applied active action"
                          + (" + GT interaction context C10" if args.oracle_c else "")
                          + " -> direct active native action12"),
                      label_contract="applied native action active coordinates; no PD target or R supervision")
    torch.save(checkpoint, args.output / "best.pt")
    best_summary = dict(val_active_l1=best["val_active_l1"], step=best["step"], state=None)
    result = dict(status="PROMISING" if best["val_active_l1"] < .20 else "UNPROMISING",
                  offline_only=True, train= evaluate(policy, train, mean, scale, device, torch),
                  validation=evaluate(policy, validation, mean, scale, device, torch),
                  test=evaluate(policy, test, mean, scale, device, torch),
                  unseen=unseen_result, best=best_summary,
                  checkpoint_sha256=sha(args.output / "best.pt"),
                  source_limitation="32 environments duplicate one s3 motion; unseen result is sequence-disjoint only if supplied")
    write(args.output / "result.json", result)
    source_hashes = {str(args.trajectory.resolve()): sha(args.trajectory),
                     str(args.trajectory_manifest.resolve()): sha(args.trajectory_manifest),
                     str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                     str((TASK / "src/trajectory_policy/tau_executor.py").resolve()): sha(TASK / "src/trajectory_policy/tau_executor.py")}
    if args.unseen_trajectory:
        source_hashes[str(args.unseen_trajectory.resolve())] = sha(args.unseen_trajectory)
    write(args.output / "manifest.json", dict(schema="ref2dex.direct-tau-bc.v1", status="COMPLETED",
        task="trajectory-policy", experiment_id=("P-20261017-direct-tau-executor-oracle-c" if args.oracle_c else "P-20261017-direct-tau-executor"), run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, seed=args.seed, horizon=args.horizon, width=args.width,
        source=str(args.trajectory.resolve()), unseen_source=str(args.unseen_trajectory.resolve()) if args.unseen_trajectory else None,
        input_sha256=source_hashes, gpu_before=before, gpu_after=gpu_state(args.gpu),
        elapsed_s=time.monotonic() - started, checkpoint_sha256=sha(args.output / "best.pt"),
        training_rows=len(train["label"]), validation_rows=len(validation["label"]), test_rows=len(test["label"])))


if __name__ == "__main__":
    main()
