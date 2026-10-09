"""Fit a matched C0/C1 trajectory evaluator on approximate H panels.

The input panels are constructed from history-preserving structured rollouts.
They match H/state by a recorded tolerance rather than an exact fork, so this
fit is a Probe about data sufficiency, not a deployable ranking result.
"""

import argparse
import copy
import json
import os
from pathlib import Path
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
from consequence_evaluator.old_utility import panel_metrics
from consequence_evaluator.trajectory_utility import TrajectoryUtility


SCHEMA = "ref2dex.history-candidate-utility-fit.v1"
HORIZON = 24
TRAJECTORY_DIM = 33
EFFECT_DIM = 12


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_stats(value, floor=1e-4):
    value = np.asarray(value)
    axes = tuple(range(value.ndim - 1))
    return (value.mean(axes, dtype=np.float64).astype("float32"),
            np.maximum(value.std(axes, dtype=np.float64), floor).astype("float32"))


def encode(value, stats):
    mean, scale = stats
    return ((np.asarray(value) - mean) / scale).astype("float32")


def regression(target, prediction, episode):
    target = np.asarray(target); prediction = np.asarray(prediction); episode = np.asarray(episode)
    mse = [float(np.mean((prediction[episode == value] - target[episode == value]) ** 2))
           for value in np.unique(episode)]
    return dict(mae=float(np.abs(target - prediction).mean()),
                rmse=float(np.sqrt(np.mean((target - prediction) ** 2))),
                episode_mse=float(np.mean(mse)), episodes=len(mse), windows=len(target))


def load_data(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "ref2dex.history-candidate-panel.v1" or manifest.get("status") != "COMPLETED":
        raise ValueError("completed history candidate panel required")
    with np.load(root / "windows.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"history", "pw_hand_future", "effect_gt", "label", "episode", "split",
                "split_group", "panel", "candidate", "h_to_anchor", "hand_to_anchor_mm", "q_to_anchor"}
    if not required.issubset(data):
        raise ValueError("candidate panel fields missing")
    n = len(data["label"])
    if (data["history"].shape != (n, 1442)
            or data["pw_hand_future"].shape != (n, HORIZON, 11, 3)
            or data["effect_gt"].shape != (n, HORIZON, EFFECT_DIM)):
        raise ValueError("candidate panel shape mismatch")
    if not all(np.isfinite(data[key]).all() for key in (
            "history", "pw_hand_future", "effect_gt", "label", "h_to_anchor",
            "hand_to_anchor_mm", "q_to_anchor")):
        raise ValueError("nonfinite candidate panel")
    panels = np.unique(data["panel"])
    rows = np.asarray([np.flatnonzero(data["panel"] == panel) for panel in panels])
    if rows.shape[1] != 7 or not np.all(data["candidate"][rows] == np.arange(7)):
        raise ValueError("complete 7-candidate panels required")
    if np.max(data["h_to_anchor"]) > .03001 or np.max(data["hand_to_anchor_mm"]) > 3.001:
        raise ValueError("panel tolerance contract drift")
    split_by_panel = data["split"][rows[:, 0]]
    if any(np.unique(data["split"][row]).size != 1 for row in rows):
        raise ValueError("panel crosses split")
    return root, manifest, data, rows, split_by_panel


def panel_batches(rows, train_panel_ids, steps, batch, seed):
    if batch < 7:
        raise ValueError("batch must contain at least one full panel")
    rng = np.random.default_rng(seed)
    per_step = max(1, batch // 7)
    batches = []
    for _ in range(steps):
        selected = rng.choice(train_panel_ids, size=per_step, replace=True)
        batches.append(rows[selected].reshape(-1))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=415)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator")
            or not 1 <= args.steps <= 1200 or not 1 <= args.seconds <= 600
            or not 7 <= args.batch <= 512):
        raise ValueError("bounded history candidate utility fit required")
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied: " + occupied)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    root, manifest, data, rows, split_by_panel = load_data(args.data)
    panel_ids = np.arange(len(rows))
    split_ids = {name: panel_ids[split_by_panel == name]
                 for name in ("train", "val", "test")}
    if any(not len(value) for value in split_ids.values()):
        raise ValueError("train/val/test panel splits must all be nonempty")
    flat_train = rows[split_by_panel == "train"].reshape(-1)
    stats = {
        "history": fit_stats(data["history"][flat_train]),
        "trajectory": fit_stats(data["pw_hand_future"][flat_train].reshape(-1, HORIZON, TRAJECTORY_DIM)),
        "effect": fit_stats(data["effect_gt"][flat_train]),
    }
    trajectory = data["pw_hand_future"].reshape(-1, HORIZON, TRAJECTORY_DIM)
    tensors = {
        "history": torch.as_tensor(encode(data["history"], stats["history"]), device=device),
        "trajectory": torch.as_tensor(encode(trajectory, stats["trajectory"]), device=device),
        "effect": torch.as_tensor(encode(data["effect_gt"], stats["effect"]), device=device),
        "label": torch.as_tensor(data["label"], device=device),
    }
    initial = TrajectoryUtility().to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    models = {arm: copy.deepcopy(initial) for arm in ("C0", "C1")}
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                  for arm, model in models.items()}
    batches = panel_batches(rows, split_ids["train"], args.steps, args.batch, args.seed)
    frozen = {str(root / name): sha(root / name) for name in ("manifest.json", "windows.npz")}
    frozen.update({str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                   str(TASK / "src/consequence_evaluator/trajectory_utility.py"):
                       sha(TASK / "src/consequence_evaluator/trajectory_utility.py"),
                   str(TASK / "src/consequence_evaluator/old_utility.py"):
                       sha(TASK / "src/consequence_evaluator/old_utility.py")})
    output.mkdir(parents=True)
    torch.save(initial_state, output / "initial.pt")
    fit_manifest = dict(schema=SCHEMA, status="RUNNING",
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        input_sha256=frozen, initial_sha256=sha(output / "initial.pt"),
                        physical_gpu=args.gpu, seed=args.seed, steps_cap=args.steps,
                        seconds_cap=args.seconds, batch=args.batch,
                        architecture=dict(history_dim=1442, trajectory_dim=TRAJECTORY_DIM,
                                          object_effect_dim=EFFECT_DIM, width=128, layers=2),
                        arms={"C0": "H+tau", "C1": "H+tau+E_GT"},
                        panel_contract="7 candidates with H RMS<=.03, hand RMS<=3mm, q RMS<=.03 to anchor",
                        approximate_h_match=True, train_panels=int(len(split_ids["train"])),
                        val_panels=int(len(split_ids["val"])), test_panels=int(len(split_ids["test"])),
                        shared_initialization=True)
    write(output / "manifest.json", fit_manifest)

    def predict(model, chosen, tau_rows=None, arm="C0"):
        model.eval(); values = []
        tau_rows = chosen if tau_rows is None else tau_rows
        for begin in range(0, len(chosen), 256):
            index = chosen[begin:begin + 256]
            tau_index = tau_rows[begin:begin + 256]
            with torch.inference_mode():
                values.append(model(tensors["history"][index], tensors["trajectory"][tau_index],
                                    tensors["effect"][index], arm == "C1").cpu().numpy())
        return np.concatenate(values)

    best = {arm: float("inf") for arm in models}; selected = {}; history = []
    start = time.monotonic(); completed = 0
    for step, batch in enumerate(batches, 1):
        if time.monotonic() - start > args.seconds:
            break
        losses = {}
        for arm, model in models.items():
            model.train(); optimizers[arm].zero_grad(set_to_none=True)
            score = model(tensors["history"][batch], tensors["trajectory"][batch],
                          tensors["effect"][batch], arm == "C1")
            loss = ((score - tensors["label"][batch]) ** 2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite history candidate utility loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizers[arm].step()
            losses[arm] = float(loss)
        completed = step
        if step == 1 or step % 100 == 0 or step == args.steps:
            validation = {}
            val_rows = rows[split_by_panel == "val"].reshape(-1)
            for arm, model in models.items():
                pred = predict(model, val_rows, arm=arm)
                validation[arm] = regression(data["label"][val_rows], pred, data["episode"][val_rows])
                if validation[arm]["episode_mse"] < best[arm]:
                    best[arm] = validation[arm]["episode_mse"]; selected[arm] = step
                    torch.save(dict(schema=SCHEMA, model=model.state_dict(), statistics=stats,
                                    architecture=fit_manifest["architecture"], arm=arm, step=step,
                                    data_sha256=sha(root / "windows.npz")), output / (arm + ".pt"))
            elapsed = time.monotonic() - start
            gpu = subprocess.check_output([
                "nvidia-smi", "-i", str(args.gpu), "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader"], text=True).strip()
            row = dict(step=step, elapsed_s=elapsed, losses=losses, val=validation, gpu=gpu)
            history.append(row)
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(dict(step=step, elapsed_s=round(elapsed, 1),
                                  eta_s=round(elapsed / step * (args.steps - step), 1),
                                  gpu=gpu, val_rmse={arm: value["rmse"] for arm, value in validation.items()})),
                  flush=True)

    test_rows = rows[split_by_panel == "test"]
    test_flat = test_rows.reshape(-1)
    test_panels = len(test_rows)
    rng = np.random.default_rng(args.seed + 1)
    donor = np.stack([np.roll(row, int(rng.integers(1, 7))) for row in test_rows]).reshape(-1)
    target = data["label"][test_flat].reshape(test_panels, 7)
    metrics = {}; shuffled = {}; predictions = {}
    for arm, model in models.items():
        checkpoint = output / (arm + ".pt")
        if not checkpoint.exists():
            raise RuntimeError("fit ended before validation checkpoint: " + arm)
        packet = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(packet["model"])
        predictions[arm] = predict(model, test_flat, arm=arm).reshape(test_panels, 7)
        shuffled[arm] = predict(model, test_flat, donor, arm=arm).reshape(test_panels, 7)
        metrics[arm] = panel_metrics(target, predictions[arm])
    shuffle_metrics = {arm: panel_metrics(target, shuffled[arm]) for arm in models}
    gain = metrics["C1"]["pairwise_accuracy"] - metrics["C0"]["pairwise_accuracy"]
    drop = metrics["C1"]["pairwise_accuracy"] - shuffle_metrics["C1"]["pairwise_accuracy"]
    screen_gate = bool(metrics["C1"]["pairwise_accuracy"] >= .70 and gain >= .03
                       and drop >= .03 and metrics["C1"]["mean_regret"] <= metrics["C0"]["mean_regret"])
    result = dict(status="PROMISING" if screen_gate else "UNCLEAR",
                  screen_gate=screen_gate, metrics=metrics, shuffle_metrics=shuffle_metrics,
                  C1_gain_vs_C0=float(gain), C1_tau_shuffle_drop=float(drop),
                  selected_steps=selected, completed_steps=completed,
                  evidence_scope="episode-split approximate-H panels; offline trajectory utility only",
                  deployable_planner=False, native_execution_claim=False,
                  checkpoint_sha256={arm: sha(output / (arm + ".pt")) for arm in models},
                  input_sha256=frozen, panel_rows=test_rows.tolist(), donor=donor.tolist())
    np.savez_compressed(output / "panel-predictions.npz", target=target,
                        panel_rows=test_rows, donor=donor, **predictions,
                        C0_tau_shuffle=shuffled["C0"], C1_tau_shuffle=shuffled["C1"])
    write(output / "result.json", result)
    fit_manifest.update(status="COMPLETED", completed_steps=completed,
                        selected_steps=selected, elapsed_s=time.monotonic() - start,
                        checkpoint_sha256=result["checkpoint_sha256"])
    write(output / "manifest.json", fit_manifest)
    print(json.dumps({key: result[key] for key in (
        "status", "screen_gate", "metrics", "shuffle_metrics",
        "C1_gain_vs_C0", "C1_tau_shuffle_drop")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
