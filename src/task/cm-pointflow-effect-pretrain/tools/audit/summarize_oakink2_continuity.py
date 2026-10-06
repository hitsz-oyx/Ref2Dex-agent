#!/usr/bin/env python3
"""Check adjacent-frame continuity and filtering overlap from completed audit caches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.concatenate(values) if values else np.empty(0)
    values = values[np.isfinite(values)]
    return {"count": int(values.size), "quantiles_min_p50_p90_p99_max":
            np.quantile(values, [0, .5, .9, .99, 1]).tolist() if values.size else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    result = json.loads((root / "audit/result.json").read_text())
    assert result["status"] == "DATA_AUDIT_COMPLETE" and result["sequences"] == 100
    # Validate frozen identities before the protocol is updated with results.
    inputs = json.loads((root / "input_manifest.json").read_text())
    for name, digest in inputs["sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    hand_steps, object_steps, rotation_steps = [], [], []
    totals = dict(frame_gaps=0, missing_right_frames=0, missing_left_frames=0,
                  both_hand_frames=0, frames=0, object_frames=0, invalid_object_frames=0,
                  hand_point_steps_over_5cm=0, object_steps_over_5cm=0,
                  rotation_steps_over_02rad=0)
    arms = {k: {"windows": 0, "moving": 0} for k in
            ("all", "program", "near", "program_and_near", "program_or_near")}
    sequences, anomalies = [], []
    for trial in result["trials"]:
        seq = trial["sequence"]
        with np.load(root / "audit/sequences" / (seq + ".npz")) as data:
            ids, hand, valid = data["frame_ids"], data["hand_xyz"], data["hand_valid"]
        consecutive = np.diff(ids) == 1
        active = consecutive[:, None] & valid[1:] & valid[:-1]
        steps = np.linalg.norm(hand[1:] - hand[:-1], axis=-1)[active].reshape(-1)
        hand_steps.append(steps)
        totals["frame_gaps"] += int((~consecutive).sum())
        totals["frames"] += len(ids)
        totals["missing_right_frames"] += int((~valid[:, 0]).sum())
        totals["missing_left_frames"] += int((~valid[:, 1]).sum())
        totals["both_hand_frames"] += int(valid.all(-1).sum())
        totals["hand_point_steps_over_5cm"] += int((steps > .05).sum())
        seq_moving = 0
        for obj in trial["objects"]:
            with np.load(root / "audit/sequences" / (seq + "__" + obj["object"] + ".npz")) as data:
                T, rigid, good = data["object_transform"], data["object_valid"], data["window_valid"]
                program = np.lib.stride_tricks.sliding_window_view(data["program_frame"], 9).all(-1) & good
                near = data["near_window"] & good
            totals["object_frames"] += len(T)
            totals["invalid_object_frames"] += int((~rigid).sum())
            adjacent = consecutive & rigid[1:] & rigid[:-1]
            shift = np.linalg.norm(T[1:, :3, 3] - T[:-1, :3, 3], axis=-1)[adjacent]
            R = T[1:, :3, :3][adjacent] @ T[:-1, :3, :3][adjacent].transpose(0, 2, 1)
            angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
            object_steps.append(shift)
            rotation_steps.append(angle)
            totals["object_steps_over_5cm"] += int((shift > .05).sum())
            totals["rotation_steps_over_02rad"] += int((angle > .2).sum())
            jump_indices = np.flatnonzero(adjacent)[(shift > .05) | (angle > .2)]
            if len(jump_indices):
                crossing = np.zeros(len(good), dtype=bool)
                for tick in jump_indices:
                    crossing[max(0, tick - 7):min(len(good), tick + 1)] = True
                anomalies.append(dict(sequence=seq, object=obj["object"],
                                      frame_pairs=[[int(ids[t]), int(ids[t+1])] for t in jump_indices],
                                      affected_valid_windows=int((crossing & good).sum()),
                                      affected_program_windows=int((crossing & program).sum())))
            displacement = np.linalg.norm(T[8:, :3, 3] - T[:-8, :3, 3], axis=-1)
            relative = T[8:, :3, :3] @ T[:-8, :3, :3].transpose(0, 2, 1)
            h8angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
            moving = (displacement > .002) | (h8angle > .02)
            for arm, mask in (("all", good), ("program", program), ("near", near),
                              ("program_and_near", program & near), ("program_or_near", program | near)):
                arms[arm]["windows"] += int(mask.sum())
                arms[arm]["moving"] += int((mask & moving).sum())
            seq_moving += int(((program | near) & moving).sum())
        sequences.append({"sequence": seq, "interaction_moving_windows": seq_moving})
    for row in arms.values():
        row["static_fraction"] = 1 - row["moving"] / row["windows"] if row["windows"] else None
    assert arms["all"]["windows"] == result["counts"]["windows"]
    assert arms["program_or_near"]["moving"] == result["counts"]["interaction_moving"]
    report = dict(input_hashes_match=True,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  result_sha256=hashlib.sha256((root / "audit/result.json").read_bytes()).hexdigest(),
                  totals=totals, filters=arms,
                  adjacent_hand_point_displacement_m=distribution(hand_steps),
                  adjacent_object_translation_m=distribution(object_steps),
                  adjacent_object_rotation_rad=distribution(rotation_steps),
                  sequences_with_interaction_motion=sum(s["interaction_moving_windows"] > 0 for s in sequences),
                  sequences=sequences, object_jump_locations=anomalies,
                  limitation="Jump thresholds are diagnostics, not validated rejection rules. Overlapping windows are not independent samples.")
    (root / "audit/continuity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("sequences", "object_jump_locations")}, indent=2))


if __name__ == "__main__":
    main()
