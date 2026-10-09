"""Audit whether a frozen hand bridge responds to candidate-plan scale.

This is a CPU-only attribution audit.  It keeps the observed same-H panel,
the frozen HA bridge, and all model parameters fixed while multiplying the
known residual plan at inference.  It does not claim that an out-of-range
plan is a valid control intervention.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.data import sha
from consequence_evaluator.hand_execution import HandExecution, compose_motion, metrics


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def panel_rows(data):
    panels = np.unique(data["panel"][data["panel"] >= 0])
    rows = np.asarray([
        sorted(np.flatnonzero(data["panel"] == panel), key=lambda row: data["candidate"][row])
        for panel in panels
    ])
    if rows.shape != (25, 7):
        raise ValueError("expected 25 same-H x 7 candidate panel")
    if not np.all(data["history"][rows] == data["history"][rows[:, :1]]):
        raise ValueError("panel history identity drift")
    if not all(np.array_equal(data["candidate"][row], np.arange(7)) for row in rows):
        raise ValueError("candidate order is not 0..6")
    return rows


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.parent.resolve() != (ROOT / "outputs/consequence-evaluator").resolve():
        raise ValueError("fresh task-owned plan-scale audit output required")
    scales = np.asarray(args.scales, dtype="float32")
    if scales.ndim != 1 or len(scales) < 2 or not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("scales must be finite nonnegative values")
    with np.load(args.data / "windows.npz", allow_pickle=False) as packed:
        data = {key: packed[key] for key in packed.files}
    rows = panel_rows(data)
    flat = rows.ravel()
    bridge = torch.load(args.bridge, map_location="cpu", weights_only=False)
    if bridge.get("schema") != "ref2dex.hand-execution.v1" or bridge.get("arm") != "HA":
        raise ValueError("HA bridge checkpoint required")
    required = {"history", "action", "pw_hand_history", "pw_hand_future"}
    if not required.issubset(data):
        raise ValueError("trajectory windows missing bridge fields")
    if data["action"].shape[1:] != (24, 18) or data["pw_hand_history"].shape[1:] != (4, 11, 3):
        raise ValueError("unexpected bridge input shape")
    state = np.concatenate((data["history"], data["pw_hand_history"].reshape(len(data["history"]), -1)), -1)
    current = data["pw_hand_history"][:, -1]
    target = data["pw_hand_future"]
    model = HandExecution(bridge["architecture"]["width"]).eval()
    model.load_state_dict(bridge["model"], strict=True)
    mean, scale = (np.asarray(value, dtype="float32") for value in bridge["statistics"])
    plan_unit = np.asarray(bridge["plan_unit"], dtype="float32")
    state_norm = (state - mean) / scale
    state_tensor = torch.as_tensor(state_norm, dtype=torch.float32)
    current_tensor = torch.as_tensor(current, dtype=torch.float32)
    target_panel = target[flat]
    scale_metrics = {}
    predictions = {}
    with torch.inference_mode():
        for factor in scales:
            raw_plan = data["action"] * factor
            plan_tensor = torch.as_tensor(raw_plan / plan_unit, dtype=torch.float32)
            motion = model(state_tensor, plan_tensor) * float(bridge["output_unit_m"])
            prediction = compose_motion(current_tensor, motion).numpy()
            panel_prediction = prediction[flat]
            panel_prediction = panel_prediction.reshape(25, 7, 24, 11, 3)
            baseline = panel_prediction[:, :1]
            candidate_spread = panel_prediction - baseline
            scale_metrics[str(float(factor))] = dict(
                bridge_metrics=metrics(panel_prediction.reshape(-1, 24, 11, 3), target_panel),
                candidate_mean_rms=rms(candidate_spread[:, 1:]),
                candidate_max_rms=float(np.max(np.sqrt(np.mean(np.square(candidate_spread[:, 1:]), axis=(1, 2, 3))))),
                response_to_zero_plan_rms=rms(panel_prediction - predictions["0.0"])
                if "0.0" in predictions else None,
                action_scale=float(factor),
            )
            predictions[str(float(factor))] = panel_prediction
    output.mkdir(parents=True)
    hashes = {str(path.resolve()): sha(path.resolve()) for path in (
        args.data / "manifest.json", args.data / "windows.npz", args.bridge,
        Path(__file__).resolve())}
    result = dict(
        status="COMPLETED",
        schema="ref2dex.hand-bridge-plan-scale-audit.v1",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_sha256=hashes,
        panel_rows=rows.tolist(),
        scales=[float(value) for value in scales],
        metrics=scale_metrics,
        limitation=("Inference-only plan scaling outside the trained residual distribution; "
                    "candidate spread and fit error are not control or ranking evidence."),
    )
    np.savez_compressed(output / "predictions.npz", rows=rows, target=target_panel,
                        **{f"scale_{key.replace('.', '_')}": value for key, value in predictions.items()})
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
