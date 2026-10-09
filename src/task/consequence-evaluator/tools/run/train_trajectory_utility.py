"""Fit matched C0/C1/C2 evaluators with a real hand-trajectory interface.

C0 receives H and a candidate hand trajectory tau.  C1 additionally receives
the measured GT object effect and C2 receives the frozen PointWorld effect.
This is an offline oracle Probe: the trajectory and object effect are both
observed from the prepared windows and no execution claim is made here.
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
from scipy.stats import spearmanr

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.data import sha
from consequence_evaluator.old_utility import panel_metrics
from consequence_evaluator.trajectory_utility import (
    HORIZON, OBJECT_EFFECT_DIM, SCHEMA, TRAJECTORY_DIM, TrajectoryUtility)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def regression(y, pred, episode):
    y = np.asarray(y); pred = np.asarray(pred); episode = np.asarray(episode)
    mse = [float(np.mean((pred[episode == e] - y[episode == e]) ** 2))
           for e in np.unique(episode)]
    corr = (float(spearmanr(y, pred).statistic)
            if np.std(y) > 0 and np.std(pred) > 0 else None)
    return dict(mae=float(np.abs(y - pred).mean()),
                rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
                episode_mse=float(np.mean(mse)), spearman=corr,
                episodes=len(mse), windows=len(y))


def fit_stats(value, floor=1e-4):
    value = np.asarray(value)
    axes = tuple(range(value.ndim - 1))
    mean = value.mean(axes, dtype=np.float64)
    scale = np.maximum(value.std(axes, dtype=np.float64), floor)
    return mean.astype("float32"), scale.astype("float32")


def encode(value, stats):
    mean, scale = stats
    return ((np.asarray(value) - mean) / scale).astype("float32")


def load_data(data_root, pw_root):
    data_root = Path(data_root).resolve(); pw_root = Path(pw_root).resolve()
    manifest = json.loads((data_root / "manifest.json").read_text())
    pw_manifest = json.loads((pw_root / "manifest.json").read_text())
    if (manifest.get("schema") != "ref2dex.consequence-old-utility.v1"
            or manifest.get("status") != "COMPLETED"
            or pw_manifest.get("status") != "COMPLETED"
            or pw_manifest.get("oracle_observed_hand") is not True
            or manifest.get("windows_sha256") != sha(data_root / "windows.npz")
            or pw_manifest.get("windows_sha256") != manifest.get("windows_sha256")
            or pw_manifest.get("future_sha256") != sha(pw_root / "future.npz")):
        raise ValueError("fixed same-row GT/PW data required")
    with np.load(data_root / "windows.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(pw_root / "future.npz", allow_pickle=False) as source:
        pw_future = np.asarray(source["future"], dtype="float32")
    required = {"history", "future", "pw_hand_future", "label", "episode", "split",
                "split_group", "panel", "candidate"}
    if not required.issubset(data):
        raise ValueError("trajectory windows missing required arrays")
    n = len(data["label"])
    if (data["history"].shape != (n, 1442)
            or data["future"].shape != (n, HORIZON, 45)
            or data["pw_hand_future"].shape != (n, HORIZON, 11, 3)
            or pw_future.shape != data["future"].shape):
        raise ValueError("trajectory/effect window shape mismatch")
    if not all(np.isfinite(data[key]).all() for key in ("history", "future", "pw_hand_future", "label")):
        raise ValueError("nonfinite trajectory evaluator data")
    for group in np.unique(data["split_group"]):
        if len(np.unique(data["split"][data["split_group"] == group])) != 1:
            raise ValueError("split leakage")
    return data, pw_future, manifest, pw_manifest


def episode_batches(rows, episodes, steps, batch, seed):
    groups = [rows[episodes[rows] == episode] for episode in np.unique(episodes[rows])]
    if not groups or any(not len(group) for group in groups):
        raise ValueError("nonempty episode groups required")
    rng = np.random.default_rng(seed)
    return [np.asarray([rng.choice(groups[rng.integers(len(groups))]) for _ in range(batch)],
                       dtype=np.int64) for _ in range(steps)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=408)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator")
            or not 1 <= args.steps <= 1200 or not 1 <= args.seconds <= 600
            or not 1 <= args.batch <= 512):
        raise ValueError("fresh bounded trajectory-evaluator fit required")
    if subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    data, pw_future, manifest, pw_manifest = load_data(args.data, args.pw)
    ids = {split: np.flatnonzero(data["split"] == split)
           for split in ("train", "val", "test", "panel")}
    train_rows = ids["train"]
    tau = data["pw_hand_future"].reshape(-1, HORIZON, TRAJECTORY_DIM)
    effect_gt = data["future"][:, :, :OBJECT_EFFECT_DIM]
    effect_pw = pw_future[:, :, :OBJECT_EFFECT_DIM]
    stats = {
        "history": fit_stats(data["history"][train_rows]),
        "trajectory": fit_stats(tau[train_rows]),
        "effect": fit_stats(effect_gt[train_rows]),
    }
    tensors = {
        "history": torch.as_tensor(encode(data["history"], stats["history"]), device=device),
        "trajectory": torch.as_tensor(encode(tau, stats["trajectory"]), device=device),
        "effect_gt": torch.as_tensor(encode(effect_gt, stats["effect"]), device=device),
        "effect_pw": torch.as_tensor(encode(effect_pw, stats["effect"]), device=device),
        "label": torch.as_tensor(data["label"], device=device),
    }
    # Keep the three input arms matched at initialization.  A prior Probe
    # constructed three independent models and saved a fourth, unused model
    # as ``initial.pt``; that made small arm deltas uninterpretable.
    initial_model = TrajectoryUtility().to(device)
    initial_state = copy.deepcopy(initial_model.state_dict())
    models = {arm: copy.deepcopy(initial_model) for arm in ("C0", "C1", "C2")}
    opts = {arm: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            for arm, model in models.items()}
    batches = episode_batches(train_rows, data["episode"], args.steps, args.batch, args.seed)

    frozen = {str(path): sha(path) for path in (
        args.data.resolve() / "manifest.json", args.data.resolve() / "windows.npz",
        args.pw.resolve() / "manifest.json", args.pw.resolve() / "future.npz",
        Path(__file__).resolve(), TASK / "src/consequence_evaluator/trajectory_utility.py",
        TASK / "src/consequence_evaluator/old_utility.py")}
    output.mkdir(parents=True)
    torch.save(initial_state, output / "initial.pt")
    fit_manifest = dict(
        schema=SCHEMA, status="RUNNING", git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        task="consequence-evaluator", route="mainline-trajectory-evaluator",
        input_sha256=frozen, initial_sha256=sha(output / "initial.pt"),
        physical_gpu=args.gpu, seed=args.seed, steps_cap=args.steps,
        seconds_cap=args.seconds, batch=args.batch,
        architecture=dict(history_dim=1442, trajectory_dim=TRAJECTORY_DIM,
                          object_effect_dim=OBJECT_EFFECT_DIM, width=128, layers=2),
        input_contract="H + tau[24,11,3] in current object frame + optional object effect[24,12]",
        arms={"C0": "H+tau", "C1": "H+tau+E_GT", "C2": "H+tau+E_PW"},
        tau_source="pw_hand_future; observed hand trajectory in current query object frame",
        object_effect_source="future[:,:,:12] or frozen PointWorld future[:,:,:12]",
        label=manifest["label"], test_used_for_selection=False,
        ordinary_rows={split: int(len(rows)) for split, rows in ids.items()},
        panel_contract="25 same-H anchors x 7 candidates; candidate trajectories/effects retained",
        sampler="episode-uniform then window-uniform",
        shared_initialization=True)
    write(output / "manifest.json", fit_manifest)

    best = {arm: float("inf") for arm in models}; selected = {}; history = []
    start = time.monotonic(); completed = 0

    def effect_key(arm):
        return "effect_gt" if arm == "C1" else "effect_pw"

    @torch.inference_mode()
    def predict(model, rows, arm, tau_rows=None):
        model.eval(); values = []
        tau_rows = rows if tau_rows is None else tau_rows
        for begin in range(0, len(rows), 256):
            chosen = rows[begin:begin + 256]; candidate_tau = tau_rows[begin:begin + 256]
            effect = tensors[effect_key(arm)][chosen]
            values.append(model(tensors["history"][chosen], tensors["trajectory"][candidate_tau],
                                effect, arm != "C0").cpu().numpy())
        return np.concatenate(values)

    for step, batch in enumerate(batches, 1):
        if time.monotonic() - start > args.seconds:
            break
        losses = {}
        for arm, model in models.items():
            model.train(); opts[arm].zero_grad(set_to_none=True)
            score = model(tensors["history"][batch], tensors["trajectory"][batch],
                          tensors[effect_key(arm)][batch], arm != "C0")
            loss = ((score - tensors["label"][batch]) ** 2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite trajectory evaluator loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opts[arm].step()
            losses[arm] = float(loss)
        completed = step
        if step == 1 or step % 100 == 0 or step == args.steps:
            val = {}
            for arm, model in models.items():
                rows = ids["val"]; pred = predict(model, rows, arm)
                val[arm] = regression(data["label"][rows], pred, data["episode"][rows])
                if val[arm]["episode_mse"] < best[arm]:
                    best[arm] = val[arm]["episode_mse"]; selected[arm] = step
                    torch.save(dict(schema=SCHEMA, model=model.state_dict(), statistics=stats,
                                    architecture=fit_manifest["architecture"], arm=arm, step=step,
                                    data_sha256=manifest["windows_sha256"],
                                    pw_sha256=pw_manifest["future_sha256"]),
                               output / (arm + ".pt"))
            elapsed = time.monotonic() - start
            gpu = subprocess.check_output([
                "nvidia-smi", "-i", str(args.gpu), "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader"], text=True).strip()
            row = dict(step=step, elapsed_s=elapsed, losses=losses, val=val, gpu=gpu)
            history.append(row)
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(dict(step=step, elapsed_s=round(elapsed, 1),
                                  eta_s=round(elapsed / step * (args.steps - step), 1),
                                  gpu=gpu,
                                  val_rmse={arm: value["rmse"] for arm, value in val.items()})),
                  flush=True)

    panels = np.unique(data["panel"][ids["panel"]])
    panel_rows = np.asarray([
        sorted(ids["panel"][data["panel"][ids["panel"]] == panel],
               key=lambda row: data["candidate"][row]) for panel in panels])
    if panel_rows.shape != (25, 7):
        raise ValueError("complete frozen25x7 panel required")
    for rows in panel_rows:
        if (not np.array_equal(data["candidate"][rows], np.arange(7))
                or not np.all(data["history"][rows] == data["history"][rows[0]])):
            raise ValueError("held panel identity failure")
    flat = panel_rows.ravel(); rng = np.random.default_rng(args.seed + 1)
    donor = np.stack([np.roll(rows, int(rng.integers(1, 7))) for rows in panel_rows]).ravel()
    target = data["label"][flat].reshape(25, 7)
    metrics = {}; ordinary = {}; predictions = {}; shuffled = {}
    for arm, model in models.items():
        saved = torch.load(output / (arm + ".pt"), map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        predictions[arm] = predict(model, flat, arm).reshape(25, 7)
        metrics[arm] = panel_metrics(target, predictions[arm])
        rows = ids["test"]
        ordinary[arm] = regression(data["label"][rows], predict(model, rows, arm), data["episode"][rows])
        shuffled[arm] = predict(model, flat, arm, donor).reshape(25, 7)
    shuffle_metrics = {arm: panel_metrics(target, value) for arm, value in shuffled.items()}
    gain = metrics["C1"]["pairwise_accuracy"] - metrics["C0"]["pairwise_accuracy"]
    drop = metrics["C1"]["pairwise_accuracy"] - shuffle_metrics["C1"]["pairwise_accuracy"]
    gt_gate = bool(metrics["C1"]["pairwise_accuracy"] >= .70 and gain >= .03
                   and drop >= .03 and metrics["C1"]["mean_regret"] <= metrics["C0"]["mean_regret"])
    pw_gap = metrics["C1"]["pairwise_accuracy"] - metrics["C2"]["pairwise_accuracy"]
    pw_gate = bool(gt_gate and metrics["C2"]["pairwise_accuracy"] >= .70 and pw_gap <= .05
                   and metrics["C2"]["mean_regret"] <= metrics["C1"]["mean_regret"] + .05)
    result = dict(status="PROMISING" if gt_gate else "UNCLEAR", gt_information_gate=gt_gate,
                  PW_retention_gate=pw_gate, metrics=metrics, shuffle_metrics=shuffle_metrics,
                  ordinary_test=ordinary, teacher=panel_metrics(target, target),
                  C1_gain_vs_C0=gain, C1_tau_shuffle_drop=drop, C1_minus_C2=pw_gap,
                  selected_steps=selected, completed_steps=completed,
                  evidence_scope="single recovered cohort, 25 same-H anchors; offline tau oracle",
                  C2_oracle_observed_hand=True, deployable_planner=False,
                  checkpoint_sha256={arm: sha(output / (arm + ".pt")) for arm in models},
                  input_sha256=frozen)
    if any(sha(path) != digest for path, digest in frozen.items()):
        raise ValueError("fit inputs drift")
    np.savez_compressed(output / "panel-predictions.npz", target=target, panel_rows=panel_rows,
                        donor=donor, **predictions, C0_tau_shuffle=shuffled["C0"],
                        C1_tau_shuffle=shuffled["C1"], C2_tau_shuffle=shuffled["C2"])
    write(output / "result.json", result)
    fit_manifest.update(status="COMPLETED", completed_steps=completed, selected_steps=selected,
                        elapsed_s=time.monotonic() - start, checkpoint_sha256=result["checkpoint_sha256"])
    write(output / "manifest.json", fit_manifest)
    print(json.dumps({key: result[key] for key in (
        "status", "gt_information_gate", "PW_retention_gate", "metrics", "shuffle_metrics",
        "C1_gain_vs_C0", "C1_tau_shuffle_drop", "C1_minus_C2")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
