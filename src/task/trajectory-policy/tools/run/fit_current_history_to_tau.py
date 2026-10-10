"""Fit a pure measured-history H -> hand-trajectory forecaster on current data.

This is a forecast-only Stage 2 Probe.  It deliberately stops before URDF
projection, the native executor, physics, PPO and Cm.  The input is the clean
300-D four-state measured history contract from ``proposal_history``; labels
are measured future hand keypoints expressed in the current object frame.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OLD = ROOT / "src/task/consequence-evaluator"
sys.path[:0] = [str(TASK / "src"), str(OLD / "src"), str(ROOT)]

from consequence_evaluator.data import sha
from consequence_evaluator.hand_execution import transform_future
from consequence_evaluator.proposal_history import (
    HORIZON, INPUT_DIM, MeasuredHistoryToTau, condition,
)


SCHEMA = "ref2dex.current-measured-history-tau.v1"


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(value, floor=1e-3):
    value = np.asarray(value, dtype=np.float32)
    return value.mean(0, dtype=np.float64).astype("float32"), np.maximum(
        value.std(0, dtype=np.float64), floor).astype("float32")


def encode(value, stat):
    return ((np.asarray(value, dtype=np.float32) - stat[0]) / stat[1]).astype("float32")


def metrics(prediction, target):
    delta = np.asarray(prediction) - np.asarray(target)
    return dict(
        point_rmse_m=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=-1)))),
        h1_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, 0] ** 2, axis=-1)))),
        h8_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, 7] ** 2, axis=-1)))),
        h24_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, -1] ** 2, axis=-1)))),
        wrist_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, :, 0] ** 2, axis=-1)))),
    )


def gpu_state(gpu: int):
    row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    util, used, total = (int(v.strip()) for v in row.strip().split(","))
    processes = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return dict(utilization=util, used_mib=used, total_mib=total,
                processes=processes)


def collect(trajectory_path: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "COMPLETED"
            or manifest.get("schema") != "ref2dex.tau-tracker.v1"
            or manifest.get("num_envs") != 64
            or manifest.get("tau_only") is not True):
        raise ValueError("completed retained tau tracker source required")
    roles = np.asarray(manifest.get("roles", []))
    selected = np.flatnonzero(roles == "tracker")
    if len(selected) != 16:
        raise ValueError("current source must contain exactly 16 tracker environments")
    with np.load(trajectory_path, allow_pickle=False) as source:
        key_map = {
            "q": "dof_position", "dq": "dof_velocity",
            "hand": "hand_keypoints", "obj": "object_pose",
            "velocity": "object_velocity",
        }
        if not set(key_map.values()).issubset(source.files):
            raise ValueError("retained trajectory lacks measured H fields")
        data = {logical: source[physical].copy() for logical, physical in key_map.items()}
    if (data["q"].shape != (543, 64, 18)
            or data["dq"].shape != data["q"].shape
            or data["hand"].shape != (543, 64, 11, 3)
            or data["obj"].shape != (543, 64, 4, 4)
            or data["velocity"].shape != (543, 64, 6)):
        raise ValueError("current source measured-state shapes mismatch")
    arrays = {"history": [], "current": [], "target": [], "env": [], "tick": []}
    controls = len(data["q"]) - 1
    for env in selected.tolist():
        for tick in range(0, controls - HORIZON + 1):
            past = np.maximum(tick + np.arange(-3, 1), 0)
            history, current = condition(
                data["obj"][past, env], data["hand"][past, env],
                data["q"][past, env], data["dq"][past, env],
                data["velocity"][past, env])
            target = transform_future(
                data["obj"][tick, env], data["hand"][tick + 1:tick + HORIZON + 1, env])
            arrays["history"].append(history)
            arrays["current"].append(current)
            arrays["target"].append(target)
            arrays["env"].append(env)
            arrays["tick"].append(tick)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if not all(np.isfinite(arrays[key]).all() for key in ("history", "current", "target")):
        raise ValueError("nonfinite current-source forecast rows")
    return arrays, manifest, selected


def split(arrays, selected):
    # Fixed environment-held-out split, selected before fitting or model choice.
    train_envs, val_envs, test_envs = selected[:10], selected[10:13], selected[13:]
    result = {}
    for name, envs in (("train", train_envs), ("val", val_envs), ("test", test_envs)):
        result[name] = np.isin(arrays["env"], envs)
    return result, dict(train=train_envs.tolist(), val=val_envs.tolist(), test=test_envs.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")
    if not 1 <= args.steps <= 1200 or not 1 <= args.seconds <= 600:
        raise ValueError("bounded current-source fit required")
    before = gpu_state(args.gpu)
    if before["utilization"] > 10 or before["used_mib"] > 512:
        raise RuntimeError("GPU not idle: " + repr(before))
    initial_pids = {int(row.split(",")[0]) for row in before["processes"].splitlines() if row.strip()}
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    arrays, source_manifest, selected = collect(args.trajectory, args.trajectory_manifest)
    masks, env_split = split(arrays, selected)
    source_hashes = {
        str(args.trajectory.resolve()): sha(args.trajectory),
        str(args.trajectory_manifest.resolve()): sha(args.trajectory_manifest),
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str((OLD / "src/consequence_evaluator/proposal_history.py").resolve()):
            sha(OLD / "src/consequence_evaluator/proposal_history.py"),
        str((OLD / "src/consequence_evaluator/hand_execution.py").resolve()):
            sha(OLD / "src/consequence_evaluator/hand_execution.py"),
    }
    subsets = {
        name: {key: value[masks[name]] for key, value in arrays.items()}
        for name in ("train", "val", "test")
    }
    train = subsets["train"]
    history_stat = stats(train["history"])
    current_stat = stats(train["current"])
    target_stats = {
        "absolute": stats(train["target"]),
        "displacement": stats(train["target"] - train["current"][:, None]),
    }
    tensors = {
        name: {
            "history": torch.as_tensor(encode(subsets[name]["history"], history_stat), device=device),
        }
        for name in ("train", "val", "test")
    }
    tensors["train"]["target"] = {
        arm: torch.as_tensor(encode(
            train["target"] if arm == "absolute" else train["target"] - train["current"][:, None],
            target_stats[arm]), device=device)
        for arm in target_stats
    }
    for name in ("val", "test"):
        tensors[name]["target"] = {
            arm: torch.as_tensor(encode(
                subsets[name]["target"] if arm == "absolute" else
                subsets[name]["target"] - subsets[name]["current"][:, None],
                target_stats[arm]), device=device)
            for arm in target_stats
        }
    initial = MeasuredHistoryToTau(width=256)
    models = {arm: copy.deepcopy(initial).to(device) for arm in target_stats}
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                  for arm, model in models.items()}
    train_groups = [np.flatnonzero(train["env"] == env) for env in env_split["train"]]
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True)
    manifest = dict(
        schema=SCHEMA, status="RUNNING",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        task="trajectory-policy", run_id=args.output.name, physical_gpu=args.gpu,
        gpu_before=before, seed=args.seed, input_sha256=source_hashes,
        source_schema=source_manifest["schema"], source_tracker_envs=selected.tolist(),
        environment_split=env_split, windows={name: int(mask.sum()) for name, mask in masks.items()},
        architecture=dict(history_dim=INPUT_DIM, horizon=HORIZON, width=256),
        input_contract="four measured states t-3:t: hand/object geometry, finger q/dq and object velocity in current object frame; no future q/object, action, phase, clock, contact or Cm",
        target_contract="measured hand t+1:t+24 in current object frame; matched absolute and current-hand displacement arms",
        startup_contract="t=0..2 history uses repeated frame0 measurements; startup metrics are reported separately",
        test_used_for_selection=False, physics=False, projection=False,
    )
    write(args.output / "manifest.json", manifest)
    started = time.monotonic(); best = {arm: (float("inf"), None, None) for arm in models}; history = []

    def check_gpu():
        current = gpu_state(args.gpu)
        for row in current["processes"].splitlines():
            pid, memory = (int(value) for value in row.split(","))
            if pid not in initial_pids | {os.getpid()} and memory > 512:
                raise RuntimeError("foreign GPU work: " + repr(current))
        if time.monotonic() - started > args.seconds:
            raise TimeoutError("current-source fit deadline")
        return current

    def predict(arm, name):
        model = models[arm]; model.eval(); values = []
        with torch.inference_mode():
            for begin in range(0, len(tensors[name]["history"]), 512):
                raw = model(tensors[name]["history"][begin:begin + 512])
                value = raw.cpu().numpy() * target_stats[arm][1] + target_stats[arm][0]
                if arm == "displacement":
                    value = value + subsets[name]["current"][begin:begin + 512, None]
                values.append(value.astype("float32"))
        return np.concatenate(values)

    # The clean proposal model consumes only the normalized measured-history H.
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds:
            break
        batch = np.asarray([rng.choice(group) for group in
                            (train_groups[rng.integers(len(train_groups))] for _ in range(args.batch_size))])
        for arm, model in models.items():
            model.train(); optimizers[arm].zero_grad(set_to_none=True)
            prediction = model(tensors["train"]["history"][batch])
            loss = (prediction - tensors["train"]["target"][arm][batch]).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite current-source fit")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizers[arm].step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            check_gpu(); row = dict(step=step, elapsed_s=time.monotonic() - started, val_l1={})
            for arm in models:
                pred = predict(arm, "val")
                raw = subsets["val"]["target"] if arm == "absolute" else subsets["val"]["target"] - subsets["val"]["current"][:, None]
                val_l1 = float(np.abs((pred if arm == "absolute" else pred - subsets["val"]["current"][:, None]) - raw).mean())
                row["val_l1"][arm] = val_l1
                if val_l1 < best[arm][0]:
                    best[arm] = (val_l1, step, {k: v.detach().cpu().clone() for k, v in models[arm].state_dict().items()})
            row["gpu"] = gpu_state(args.gpu); row["eta_s"] = (time.monotonic() - started) / step * (args.steps - step)
            history.append(row)
            with (args.output / "monitor.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)

    if any(value[2] is None for value in best.values()):
        raise RuntimeError("fit ended before a validation checkpoint")
    predictions = {}; results = {}
    for arm, model in models.items():
        model.load_state_dict(best[arm][2]); model.eval()
    for name in ("train", "val", "test"):
        results[name] = {}
        current = subsets[name]["current"]
        target = subsets[name]["target"]
        persistence = np.repeat(current[:, None], HORIZON, axis=1)
        for arm in models:
            prediction = predict(arm, name)
            predictions[(name, arm)] = prediction
            results[name][arm] = metrics(prediction, target)
        results[name]["persistence"] = metrics(persistence, target)
        # Separate startup rows from the rest without changing model selection.
        startup = subsets[name]["tick"] < 8
        results[name]["startup_rows"] = int(startup.sum())
        if startup.any():
            results[name]["startup"] = {
                arm: metrics(predictions[(name, arm)][startup], target[startup]) for arm in models
            }
            results[name]["startup"]["persistence"] = metrics(persistence[startup], target[startup])
    displacement = results["test"]["displacement"]
    persistence = results["test"]["persistence"]
    screen = bool(displacement["point_rmse_m"] <= .9 * persistence["point_rmse_m"]
                  and displacement["h24_rmse_m"] < persistence["h24_rmse_m"])
    status = "PROMISING" if screen else "UNPROMISING"
    np.savez_compressed(args.output / "test-predictions.npz",
                        displacement=predictions[("test", "displacement")],
                        absolute=predictions[("test", "absolute")],
                        target=subsets["test"]["target"],
                        current=subsets["test"]["current"],
                        env=subsets["test"]["env"], tick=subsets["test"]["tick"])
    checkpoints = {}
    for arm, model in models.items():
        path = args.output / (arm + "-best.pt")
        torch.save(dict(schema=SCHEMA, arm=arm, model=model.state_dict(),
                        statistics=dict(history=history_stat, current=current_stat, target=target_stats[arm]),
                        architecture=manifest["architecture"], selected_step=best[arm][1]), path)
        checkpoints[arm] = sha(path)
    result = dict(status=status, offline_only=True, screen=screen, selected_steps={arm: best[arm][1] for arm in best},
                  best_val_l1={arm: best[arm][0] for arm in best}, metrics=results,
                  windows=manifest["windows"], elapsed_s=time.monotonic() - started,
                  checkpoint_sha256=checkpoints,
                  scope="current-source pure measured-history H-to-tau forecast only; no projection, physics, PPO or Cm claim",
                  input_sha256=source_hashes, history=history)
    write(args.output / "result.json", result)
    manifest.update(status="COMPLETED", selected_steps={arm: best[arm][1] for arm in best},
                    checkpoint_sha256=checkpoints, elapsed_s=result["elapsed_s"], result_status=status)
    write(args.output / "manifest.json", manifest)
    print(json.dumps(dict(status=status, screen=screen, test=results["test"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
