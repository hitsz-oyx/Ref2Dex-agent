"""Compare Current421 and Scheduled686 BC checkpoints on paired DEV rows."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np


TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()


def load_split(manifest: Path, split: str):
    import sys

    sys.path.insert(0, str(TASK / "src"))
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus

    corpus = TeacherRolloutCorpus(manifest, split)
    measured, task, action, object_id, points = [], [], [], [], []
    sequences, active_masks = [], []
    for index in range(len(corpus)):
        episode = corpus.episode(index)
        keep = np.asarray(episode["active"], dtype=bool)
        if not keep.any():
            continue
        object_index = len(points)
        points.append(episode["object_points"])
        measured.append(episode["measured"][keep])
        task.append(episode["task"][keep])
        action.append(episode["action"][keep])
        object_id.append(np.full(int(keep.sum()), object_index, dtype=np.int64))
        sequences.append(episode["sequence"])
        active_masks.append(keep)
    return {
        "measured": np.concatenate(measured),
        "task": np.concatenate(task),
        "action": np.concatenate(action),
        "object_id": np.concatenate(object_id),
        "points": np.stack(points),
        "sequences": sequences,
        "active_masks": active_masks,
    }


def align_tau(cache: Path, split: str, data: dict):
    layout = json.loads((cache / "layout.json").read_text())
    rows = [row for row in layout["records"] if row["split"] == split]
    if [row["sequence"] for row in rows] != data["sequences"]:
        raise ValueError(f"{split} sequence order changed")
    tau = np.load(cache / f"{split}-tau.npy", mmap_mode="r")
    valid = np.load(cache / f"{split}-valid.npy", mmap_mode="r")
    compressed_tau, compressed_valid = [], []
    for row, mask in zip(rows, data["active_masks"]):
        if len(mask) != int(row["commands"]):
            raise ValueError(f"{split} active mask length mismatch for {row['sequence']}")
        compressed_tau.append(np.asarray(tau[row["lo"] : row["hi"]])[mask])
        compressed_valid.append(np.asarray(valid[row["lo"] : row["hi"]])[mask])
    return np.concatenate(compressed_tau), np.concatenate(compressed_valid)


def batch_predict(model, data, tau, valid, device, batch_size):
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for lo in range(0, len(data["action"]), batch_size):
            hi = min(lo + batch_size, len(data["action"]))
            measured = torch.as_tensor(data["measured"][lo:hi], device=device)
            task = torch.as_tensor(data["task"][lo:hi], device=device)
            points = torch.as_tensor(data["points"][data["object_id"][lo:hi]], device=device)
            if model.with_tau:
                batch_tau = torch.as_tensor(
                    np.asarray(tau[lo:hi]), device=device
                ).reshape(-1, 8, 11, 3)
                batch_valid = torch.as_tensor(valid[lo:hi], device=device)
                prediction = model(measured, task, points, tau=batch_tau, tau_valid=batch_valid)
            else:
                prediction = model(measured, task, points)
            outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs)


def metrics(prediction, action):
    error = np.abs(prediction - action)
    return {
        "command_mae": float(error.mean()),
        "root_mae": float(error[:, :6].mean()),
        "finger_mae": float(error[:, 6:].mean()),
        "max_abs": float(error.max()),
        "per_action_mae": error.mean(0).tolist(),
        "per_action_signed_bias": (prediction - action).mean(0).tolist(),
        "direction_cosine": {name: float(np.mean(np.sum(prediction[:, sl] * action[:, sl], axis=1) /
            np.maximum(np.linalg.norm(prediction[:, sl], axis=1) * np.linalg.norm(action[:, sl], axis=1), 1e-8)))
            for name, sl in (("wrist6", slice(0,6)), ("fingers6", slice(6,12)))},
    }


def phase_metrics(prediction, action, task):
    phases = np.argmax(task[:, 7:10], axis=1)
    return {
        str(phase): metrics(prediction[phases == phase], action[phases == phase])
        for phase in range(3)
        if np.any(phases == phase)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tau-cache", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--scheduled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import sys

    sys.path.insert(0, str(TASK / "src"))
    import torch

    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy

    data = load_split(args.manifest.resolve(), "dev")
    tau, valid = align_tau(args.tau_cache.resolve(), "dev", data)
    valid_ids = np.arange(len(valid))
    tau = np.where(valid[:, None], tau, 0.)
    if not len(valid_ids):
        raise ValueError("no valid paired tau rows")
    paired = {
        "measured": data["measured"][valid_ids],
        "task": data["task"][valid_ids],
        "action": data["action"][valid_ids],
        "object_id": data["object_id"][valid_ids],
        "points": data["points"],
    }
    paired_tau, paired_valid = tau[valid_ids], valid[valid_ids]

    device = torch.device("cuda")
    current_packet = torch.load(args.current.resolve(), map_location="cpu", weights_only=False)
    scheduled_packet = torch.load(args.scheduled.resolve(), map_location="cpu", weights_only=False)
    current = MeasuredInspirePolicy(with_tau=False).to(device)
    from wam_rl_policy_v3.measured_execution import TAU_LAYOUT_SCHEMA, GOAL_CONTRACT
    layout = json.loads((args.tau_cache / "layout.json").read_text())
    if (layout.get("schema") != TAU_LAYOUT_SCHEMA or layout.get("goal_contract") != GOAL_CONTRACT
            or scheduled_packet.get("schema") != "ref2dex.inspire-measured-scheduled-bc.v2"
            or scheduled_packet.get("config", {}).get("goal_contract") != GOAL_CONTRACT):
        raise ValueError("paired comparison requires corrected shared-goal cache and Scheduled686 checkpoint")
    scheduled = MeasuredInspirePolicy(with_tau=True).to(device)
    current.load_state_dict(current_packet["model"], strict=True)
    scheduled.load_state_dict(scheduled_packet["model"], strict=True)
    current_prediction = batch_predict(current, paired, paired_tau, paired_valid, device, args.batch_size)
    scheduled_prediction = batch_predict(scheduled, paired, paired_tau, paired_valid, device, args.batch_size)

    result = {
        "schema": "ref2dex.inspire-measured-bc-paired-dev.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "manifest": str(args.manifest.resolve()),
        "tau_cache": str(args.tau_cache.resolve()),
        "current_checkpoint": str(args.current.resolve()),
        "scheduled_checkpoint": str(args.scheduled.resolve()),
        "split": "dev",
        "paired_rows": int(len(valid_ids)),
        "paired_valid_rows": int(valid.sum()), "paired_invalid_rows": int((~valid).sum()),
        "paired_subset": "all task-active rows, including real startup with zero invalid tau",
        "current421": metrics(current_prediction, paired["action"]),
        "scheduled686": metrics(scheduled_prediction, paired["action"]),
        "delta_scheduled_minus_current": {},
        "phase_current421": phase_metrics(current_prediction, paired["action"], paired["task"]),
        "phase_scheduled686": phase_metrics(scheduled_prediction, paired["action"], paired["task"]),
        "prediction_delta_mae": float(np.abs(scheduled_prediction - current_prediction).mean()),
    }
    result["by_tau_validity"] = {}
    for name, mask in (("valid", valid), ("invalid", ~valid)):
        result["by_tau_validity"][name] = None if not mask.any() else dict(
            rows=int(mask.sum()), current421=metrics(current_prediction[mask], paired["action"][mask]),
            scheduled686=metrics(scheduled_prediction[mask], paired["action"][mask]))
    current_error = np.abs(current_prediction - paired["action"])
    scheduled_error = np.abs(scheduled_prediction - paired["action"])
    result["delta_scheduled_minus_current"]["command_mae"] = float(
        scheduled_error.mean() - current_error.mean()
    )
    result["delta_scheduled_minus_current"]["root_mae"] = float(
        scheduled_error[:, :6].mean() - current_error[:, :6].mean()
    )
    result["delta_scheduled_minus_current"]["finger_mae"] = float(
        scheduled_error[:, 6:].mean() - current_error[:, 6:].mean()
    )
    result["delta_scheduled_minus_current"]["max_abs"] = float(
        scheduled_error.max() - current_error.max()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
