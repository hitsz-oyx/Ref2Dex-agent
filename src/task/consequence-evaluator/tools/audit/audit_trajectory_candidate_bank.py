"""Audit frozen H->tau bridge proposals on the held same-H candidate panel."""

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
sys.path.insert(0, str(TASK / "tools" / "run"))

from consequence_evaluator.data import sha
from consequence_evaluator.hand_execution import HandExecution, compose_motion, metrics
from consequence_evaluator.old_utility import panel_metrics
from consequence_evaluator.trajectory_utility import TrajectoryUtility


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def panel_rows(data):
    panels = np.unique(data["panel"][data["panel"] >= 0])
    rows = np.asarray([
        sorted(np.flatnonzero(data["panel"] == panel), key=lambda row: data["candidate"][row])
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
    parser.add_argument("--bridge-h", type=Path, required=True)
    parser.add_argument("--bridge-ha", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.parent.resolve() != (ROOT / "outputs/consequence-evaluator").resolve():
        raise ValueError("fresh task-owned audit output required")
    if subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied")
    os_env = dict(CUDA_VISIBLE_DEVICES=str(args.gpu))
    # The parent process is still clean; setting this before CUDA initialization
    # keeps the audit pinned to the requested physical device.
    import os
    os.environ.update(os_env)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    with np.load(args.data / "windows.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    rows = panel_rows(data); flat = rows.ravel()
    bridges = {"H": torch.load(args.bridge_h, map_location="cpu", weights_only=False),
               "HA": torch.load(args.bridge_ha, map_location="cpu", weights_only=False)}
    if any(bridge.get("schema") != "ref2dex.hand-execution.v1" for bridge in bridges.values()):
        raise ValueError("hand execution bridge schema mismatch")
    state = np.concatenate((data["history"], data["pw_hand_history"].reshape(len(data["label"]), -1)), axis=1)
    current = data["pw_hand_history"][:, -1]
    target = data["pw_hand_future"]
    plan_unit = np.asarray(bridges["HA"]["plan_unit"], dtype="float32")
    predictions = {}
    with torch.inference_mode():
        for arm in ("H", "HA"):
            bridge = bridges[arm]
            stats_mean, stats_scale = bridge["statistics"]
            model = HandExecution(bridge["architecture"]["width"]).to(device).eval()
            model.load_state_dict(bridge["model"], strict=True)
            state_norm = (state - np.asarray(stats_mean)) / np.asarray(stats_scale)
            plan = np.zeros((len(data["label"]), 24, 18), dtype="float32") if arm == "H" else data["action"]
            motion = model(torch.as_tensor(state_norm, device=device).float(),
                           torch.as_tensor(plan / plan_unit, device=device).float()) * float(bridge["output_unit_m"])
            predictions[arm] = compose_motion(torch.as_tensor(current, device=device).float(), motion).cpu().numpy()
    bridge_metrics = {arm: metrics(predictions[arm][flat], target[flat]) for arm in predictions}
    candidate_diversity = {
        arm: dict(mean_rms=float(np.mean([rms(predictions[arm][row] - predictions[arm][row[0]])
                                           for row in rows])),
                  max_rms=float(np.max([rms(predictions[arm][row] - predictions[arm][row[0]])
                                        for row in rows])))
        for arm in predictions
    }

    evaluator_state = {}
    for arm in ("C0", "C1", "C2"):
        packet = torch.load(args.evaluator / (arm + ".pt"), map_location="cpu", weights_only=False)
        model_q = TrajectoryUtility(**{key: packet["architecture"][key]
                                       for key in ("history_dim", "width", "layers")}).to(device).eval()
        model_q.load_state_dict(packet["model"], strict=True)
        stat = packet["statistics"]
        hmean, hscale = stat["history"]; tmean, tscale = stat["trajectory"]; emean, escale = stat["effect"]
        h = (data["history"][flat] - hmean) / hscale
        tau = predictions["HA"][flat].reshape(len(flat), 24, 33)
        tau = (tau - tmean) / tscale
        effect_raw = (data["future"] if arm == "C1" else
                      np.load(ROOT / "outputs/consequence-evaluator/old-utility-pw-20261009-r2/future.npz")["future"])
        effect = (effect_raw[flat, :, :12] - emean) / escale
        with torch.inference_mode():
            score = model_q(torch.as_tensor(h, device=device).float(),
                            torch.as_tensor(tau, device=device).float(),
                            torch.as_tensor(effect, device=device).float(), arm != "C0").cpu().numpy()
        evaluator_state[arm] = score.reshape(25, 7)
    labels = data["label"][flat].reshape(25, 7)
    selection = {arm: panel_metrics(labels, score) for arm, score in evaluator_state.items()}
    output.mkdir(parents=True)
    hashes = {str(path.resolve()): sha(path.resolve()) for path in (
        args.data / "manifest.json", args.data / "windows.npz", args.bridge_h, args.bridge_ha,
        args.evaluator / "manifest.json", Path(__file__).resolve())}
    result = dict(status="COMPLETED", schema="ref2dex.trajectory-candidate-bank-audit.v1",
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  physical_gpu=args.gpu, input_sha256=hashes, panel_rows=rows.tolist(),
                  bridge_metrics=bridge_metrics, candidate_diversity=candidate_diversity,
                  evaluator_selection=selection,
                  label_informative_anchors=int(np.sum(np.any(np.abs(labels - labels[:, :1]) > .02, axis=1))),
                  limitation="HA proposals are frozen ordinary-data bridge outputs; evaluator was fit on observed tau")
    np.savez_compressed(output / "predictions.npz", rows=rows, labels=labels,
                        target_tau=target[flat], H_tau=predictions["H"][flat],
                        HA_tau=predictions["HA"][flat], **evaluator_state)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({key: result[key] for key in (
        "bridge_metrics", "candidate_diversity", "evaluator_selection",
        "label_informative_anchors")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
