"""Fit a bounded offline H -> hand-trajectory policy on history-preserving rollouts."""

import argparse
import json
from pathlib import Path
import os
import subprocess
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.data import sha
from consequence_evaluator.hand_execution import transform_future
from consequence_evaluator.history_to_tau import (
    CURRENT_HAND_DIM, HISTORY_DIM, HORIZON, HistoryToTau, SCHEMA)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(value, floor=1e-4):
    value = np.asarray(value)
    return value.mean(0, dtype=np.float64).astype("float32"), np.maximum(
        value.std(0, dtype=np.float64), floor).astype("float32")


def encode(value, stat):
    return ((np.asarray(value) - stat[0]) / stat[1]).astype("float32")


def collect(root, stride):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("status") != "COMPLETED" or not manifest.get("actor_observation_saved")
            or manifest.get("mode") != "retarget"):
        raise ValueError("completed history-preserving retarget rollout required")
    with np.load(root / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"actor_observation", "object_pose", "hand_keypoints", "length"}
    if not required.issubset(data):
        raise ValueError("history-preserving rollout fields missing")
    histories = []; currents = []; targets = []; episodes = []; ticks = []
    for env, length in enumerate(data["length"].tolist()):
        for tick in range(8, int(length) - HORIZON + 1, stride):
            pose = data["object_pose"][tick, env]
            current = transform_future(pose, data["hand_keypoints"][tick, env][None])[0]
            future = transform_future(pose, data["hand_keypoints"][tick + 1:tick + HORIZON + 1, env])
            histories.append(data["actor_observation"][tick, env])
            currents.append(current)
            targets.append(future)
            episodes.append(f"{manifest['seed']}:{env}")
            ticks.append(tick)
    arrays = dict(history=np.asarray(histories, dtype="float32"),
                  current_hand=np.asarray(currents, dtype="float32"),
                  target=np.asarray(targets, dtype="float32"),
                  episode=np.asarray(episodes), tick=np.asarray(ticks, dtype=np.int32))
    if arrays["history"].shape[1] != HISTORY_DIM or arrays["current_hand"].shape[1:] != (11, 3):
        raise ValueError("history-to-tau feature shape mismatch")
    if arrays["target"].shape[1:] != (HORIZON, 11, 3) or not all(
            np.isfinite(arrays[key]).all() for key in ("history", "current_hand", "target")):
        raise ValueError("nonfinite history-to-tau data")
    return arrays, manifest, {str(root / name): sha(root / name) for name in ("manifest.json", "trajectory.npz")}


def metrics(prediction, target):
    delta = np.asarray(prediction) - np.asarray(target)
    return dict(point_rmse_m=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=-1)))),
                h8_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, 7] ** 2, axis=-1)))),
                h24_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, -1] ** 2, axis=-1)))),
                wrist_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, :, 0] ** 2, axis=-1)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "val", "test"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=416)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator")
            or not 1 <= args.steps <= 2000 or not 1 <= args.seconds <= 600
            or not 1 <= args.stride <= 24 or not 32 <= args.batch <= 1024):
        raise ValueError("bounded H-to-tau fit required")
    if subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    data = {}; manifests = {}; frozen = {str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str(TASK / "src/consequence_evaluator/history_to_tau.py"):
            sha(TASK / "src/consequence_evaluator/history_to_tau.py")}
    for name in ("train", "val", "test"):
        data[name], manifests[name], hashes = collect(getattr(args, name), args.stride)
        frozen.update(hashes)
    actors = {manifest.get("actor_sha256") for manifest in manifests.values()}
    if len(actors) != 1 or None in actors:
        raise ValueError("H-to-tau splits must use one actor")
    train = data["train"]
    h_stat = stats(train["history"]); current_stat = stats(train["current_hand"])
    target_stat = stats(train["target"])
    tensors = {name: {
        "history": torch.as_tensor(encode(value["history"], h_stat), device=device),
        "current": torch.as_tensor(encode(value["current_hand"], current_stat), device=device),
        "target": torch.as_tensor(encode(value["target"], target_stat), device=device)}
        for name, value in data.items()}
    groups = [np.flatnonzero(train["episode"] == episode)
              for episode in np.unique(train["episode"])]
    model = HistoryToTau().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status="RUNNING", git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), input_sha256=frozen,
        physical_gpu=args.gpu, seed=args.seed, steps_cap=args.steps, seconds_cap=args.seconds,
        stride=args.stride, batch=args.batch, actor_sha256=next(iter(actors)),
        architecture=dict(history_dim=HISTORY_DIM, current_hand_dim=CURRENT_HAND_DIM,
                          horizon=HORIZON, width=model.width),
        input_contract="actor obs H[t] + current hand in current object frame",
        target_contract="measured hand t+1:t+24 in current object frame",
        episode_split=True, test_used_for_selection=False)
    write(output / "manifest.json", manifest)
    best = float("inf"); selected = None; history = []; started = time.monotonic()

    def predict(name):
        model.eval(); values = []
        with torch.inference_mode():
            for begin in range(0, len(data[name]["history"]), 512):
                end = begin + 512
                normalized = model(tensors[name]["history"][begin:end], tensors[name]["current"][begin:end])
                values.append((normalized.cpu().numpy() * target_stat[1] + target_stat[0]).astype("float32"))
        return np.concatenate(values)

    for step in range(1, args.steps + 1):
        if time.monotonic() - started > args.seconds:
            break
        batch = np.asarray([rng.choice(groups[rng.integers(len(groups))]) for _ in range(args.batch)])
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(tensors["train"]["history"][batch], tensors["train"]["current"][batch])
        loss = (prediction - tensors["train"]["target"][batch]).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite H-to-tau loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            val_prediction = predict("val")
            val_loss = float(np.abs(val_prediction - data["val"]["target"]).mean())
            row = dict(step=step, train_l1=float(loss), val_l1=val_loss,
                       elapsed_s=time.monotonic() - started)
            history.append(row); print(json.dumps(row), flush=True)
            if val_loss < best:
                best = val_loss; selected = step
                torch.save(dict(schema=SCHEMA, model=model.state_dict(), statistics=dict(
                    history=h_stat, current_hand=current_stat, target=target_stat),
                    architecture=manifest["architecture"], step=step, actor_sha256=next(iter(actors))),
                           output / "best.pt")
    if selected is None:
        raise RuntimeError("fit ended before a validation checkpoint")
    packet = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(packet["model"]); model.eval()
    metrics_by_split = {}; predictions = {}
    for name in ("train", "val", "test"):
        predictions[name] = predict(name)
        current = data[name]["current_hand"][:, None]
        persistence = np.repeat(current, HORIZON, axis=1)
        metrics_by_split[name] = dict(predicted=metrics(predictions[name], data[name]["target"]),
                                      persistence=metrics(persistence, data[name]["target"]),
                                      mean_train=metrics(np.broadcast_to(
                                          train["target"].mean(0), data[name]["target"].shape),
                                          data[name]["target"]))
    test_gate = bool(metrics_by_split["test"]["predicted"]["point_rmse_m"]
                     <= .9 * metrics_by_split["test"]["persistence"]["point_rmse_m"]
                     and metrics_by_split["test"]["predicted"]["h24_rmse_m"]
                     < metrics_by_split["test"]["persistence"]["h24_rmse_m"])
    result = dict(status="PROMISING" if test_gate else "UNCLEAR", test_gate=test_gate,
                  selected_step=selected, best_val_l1=best, metrics=metrics_by_split,
                  windows={name: int(len(value["history"])) for name, value in data.items()},
                  elapsed_s=time.monotonic() - started, checkpoint_sha256=sha(output / "best.pt"),
                  scope="offline H-to-tau prediction only; no R/PW/selector/control claim",
                  input_sha256=frozen, history=history)
    np.savez_compressed(output / "test-predictions.npz", prediction=predictions["test"],
                        target=data["test"]["target"], current=data["test"]["current_hand"],
                        episode=data["test"]["episode"], tick=data["test"]["tick"])
    write(output / "result.json", result)
    manifest.update(status="COMPLETED", selected_step=selected, best_val_l1=best,
                    checkpoint_sha256=result["checkpoint_sha256"], elapsed_s=result["elapsed_s"])
    write(output / "manifest.json", manifest)
    print(json.dumps({key: result[key] for key in ("status", "test_gate", "metrics", "selected_step")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
