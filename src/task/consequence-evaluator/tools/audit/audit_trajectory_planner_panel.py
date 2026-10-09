"""Audit the frozen H -> tau -> PointWorld -> trajectory-C2 panel chain.

This is a bounded offline wiring/OOD diagnostic.  The seven proposals still
come from the existing action-conditioned bridge, so this does not establish
that a direct trajectory policy or an online controller works.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.data import sha
from consequence_evaluator.hand_execution import metrics
from consequence_evaluator.old_utility import panel_metrics
from consequence_evaluator.trajectory_planner import TrajectoryPlanner


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def panel_rows(data):
    panels = np.unique(data["panel"][data["panel"] >= 0])
    rows = np.asarray([
        sorted(np.flatnonzero(data["panel"] == panel),
               key=lambda row: data["candidate"][row])
        for panel in panels])
    if rows.shape != (25, 7) or not np.all(data["history"][rows] == data["history"][rows[:, :1]]):
        raise ValueError("expected 25 same-H x 7 candidate panel")
    if not all(np.array_equal(data["candidate"][row], np.arange(7)) for row in rows):
        raise ValueError("candidate order is not 0..6")
    return rows


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--pointworld", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_parent = (ROOT / "outputs/consequence-evaluator").resolve()
    if output.exists() or output.parent.resolve() != expected_parent:
        raise ValueError("fresh task-owned planner audit output required")
    if subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied")

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False

    data_root = args.data.resolve()
    with np.load(data_root / "windows.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    rows = panel_rows(data)
    flat = rows.ravel()
    required = {"history", "pw_object_history", "pw_hand_history", "pw_hand_future", "label"}
    if not required.issubset(data):
        raise ValueError("trajectory panel is missing planner inputs")
    if data["pw_object_history"].shape != (len(data["label"]), 4, 4, 4):
        raise ValueError("unexpected PointWorld object-history shape")
    if data["pw_hand_history"].shape != (len(data["label"]), 4, 11, 3):
        raise ValueError("unexpected PointWorld hand-history shape")

    planner = TrajectoryPlanner(
        args.bridge, args.evaluator, args.pointworld, args.canonical, device="cuda:0")
    planner.verify()
    # Pass one measured state per anchor; choose() expands each state to its
    # seven frozen proposals.  Passing all 175 observed rows would create 175
    # separate candidate panels rather than one 25-anchor panel.
    anchors = rows[:, 0]
    choice, values, trajectories, effects = planner.choose(
        data["history"][anchors], data["pw_object_history"][anchors],
        data["pw_hand_history"][anchors])
    values = np.asarray(values, dtype="float32")
    trajectories = np.asarray(trajectories, dtype="float32")
    effects = np.asarray(effects, dtype="float32")
    if values.shape != (25, 7) or trajectories.shape != (25, 7, 24, 11, 3):
        raise ValueError("planner candidate output shape mismatch")
    if effects.shape != (25, 7, 24, 45) or not np.isfinite(values).all():
        raise ValueError("planner produced invalid candidate output")

    labels = data["label"][flat].reshape(25, 7)
    panel_values = values.reshape(25, 7)
    selected = panel_values.argmax(1)
    # The planner returns model-generated tau, while the labels belong to the
    # observed candidate rows.  This comparison is intentionally reported as
    # OOD diagnostic evidence rather than a ranking claim for a deployable policy.
    bank = trajectories
    bank_spread = {
        "mean_rms": float(np.mean([
            rms(bank[row] - bank[row, :1]) for row in range(len(bank))])),
        "max_rms": float(np.max([
            rms(bank[row] - bank[row, :1]) for row in range(len(bank))])),
    }
    observed = data["pw_hand_future"][flat].reshape(25, 7, 24, 11, 3)
    planner_vs_observed = {
        "point_rmse_m": float(np.sqrt(np.mean((bank - observed) ** 2))),
        "h24_rmse_m": float(np.sqrt(np.mean((bank[:, :, -1] - observed[:, :, -1]) ** 2))),
    }
    # The H/HA bridge has a 24-step point trajectory, so use the same metrics
    # helper after flattening the panel while retaining its coordinate contract.
    bridge_metrics = metrics(
        trajectories.reshape(-1, 24, 11, 3),
        observed.reshape(-1, 24, 11, 3))
    selection = panel_metrics(labels, panel_values)
    output.mkdir(parents=True)
    hashes = {str(path.resolve()): sha(path.resolve()) for path in (
        data_root / "manifest.json", data_root / "windows.npz", args.bridge.resolve(),
        args.evaluator.resolve(), args.evaluator.resolve().parent / "manifest.json",
        args.pointworld.resolve(), args.canonical.resolve(), Path(__file__).resolve(),
        TASK / "src/consequence_evaluator/trajectory_planner.py")}
    result = dict(
        status="COMPLETED",
        schema="ref2dex.trajectory-planner-panel-audit.v1",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu,
        input_sha256=hashes,
        panel_rows=rows.tolist(),
        planner_selection=selection,
        selected_candidates=selected.tolist(),
        candidate_bank_spread=bank_spread,
        planner_vs_observed_tau=planner_vs_observed,
        bridge_metrics=bridge_metrics,
        pointworld_repeatability=bool(planner.first_repeat_checked),
        limitation=("HA residual candidates are only an adapter for this audit; generated tau "
                     "is OOD relative to the observed-tau evaluator fit, so no online or task "
                     "control claim is made."))
    np.savez_compressed(
        output / "predictions.npz", rows=rows, labels=labels,
        values=panel_values, selected=selected, trajectories=bank, effects=effects,
        observed_tau=observed)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({key: result[key] for key in (
        "planner_selection", "candidate_bank_spread", "planner_vs_observed_tau",
        "bridge_metrics", "pointworld_repeatability")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
