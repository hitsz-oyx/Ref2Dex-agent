#!/usr/bin/env python3
"""Analyze the bounded native relative-task-value controller probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_executable_contact_opportunity import clustered_interval, support


def run(args):
    records = [torch.load(path, map_location="cpu", weights_only=False) for path in args.records]
    if not records or any(r.get("schema") != "ref2dex.native_pd_closed_loop.v1" for r in records):
        raise ValueError("native closed-loop records required")
    names = records[0]["policy_names"]
    if names != ["cm", "state_policy", "shuffled", "always_base", "always_fixed", "cm_calibrated"]:
        raise ValueError(f"unexpected policy panel: {names}")
    if any(r["policy_names"] != names or r["future_done"].any() for r in records):
        raise ValueError("policy panel or complete-label contract drift")
    assignment = torch.cat([r["assignment"] for r in records]).numpy()
    propensity = torch.cat([r["propensity"] for r in records]).numpy()
    motion = torch.cat([r["motion_id"] for r in records]).numpy()
    start = torch.cat([r["start_frame"] for r in records]).numpy()
    episode = np.array(sum((r["episode_id"] for r in records), []))
    groups = np.array([f"{int(m)}/{int(s)}" for m, s in zip(motion, start)])
    outcome = {key: torch.cat([r["outcome"][key] for r in records]).numpy()
               for key in ("score_mm", "lost_clearance", "joint_last3", "retained")}
    changed_base = torch.cat([r["changed_base"] for r in records]).numpy()
    changed_fixed = torch.cat([r["changed_fixed"] for r in records]).numpy()
    recommendations = torch.cat([r["recommendations"] for r in records])
    n = len(assignment)
    if not np.allclose(propensity, 1 / 6):
        raise ValueError("six-arm propensity drift")

    supports = {name: support(index, assignment, episode, groups)
                for index, name in enumerate(names)}
    adequate = all(s["matches"] >= 48 and s["episodes"] >= 12 and s["initial_groups"] >= 8
                   for s in supports.values())

    def contrast(option, reference=4):
        weight = (assignment == option) / propensity - (assignment == reference) / propensity
        score = weight * outcome["score_mm"]
        risk = weight * outcome["lost_clearance"]
        joint = weight * outcome["joint_last3"]
        frame = clustered_interval(score, groups)
        episode_interval = clustered_interval(score, episode)
        effect = float(score.mean())
        gates = dict(
            support=supports[names[option]]["matches"] >= 48 and supports[names[reference]]["matches"] >= 48,
            gain_at_least_2mm=effect >= 2.0,
            initial_group90_positive=frame is not None and frame[0] > 0,
            episode90_positive=episode_interval is not None and episode_interval[0] > 0,
            geometric_loss_point_bound=float(risk.mean()) <= .02,
            joint_force_point_bound=float(joint.mean()) >= -.05,
        )
        gates["passed"] = all(gates.values())
        return dict(option=names[option], reference=names[reference], score_uplift_mm=effect,
                    initial_group90=frame, episode90=episode_interval,
                    lost_clearance_difference=float(risk.mean()),
                    joint_force_difference=float(joint.mean()), gate=gates)

    contrasts = {
        "cm_vs_fixed": contrast(0),
        "calibrated_vs_fixed": contrast(5),
        "calibrated_vs_cm": contrast(5, reference=0),
    }
    policies = {}
    for index, name in enumerate(names):
        mask = assignment == index
        program = torch.cat([r["program"] for r in records])[mask]
        policies[name] = dict(
            support=supports[name],
            observed_score_mm=float(outcome["score_mm"][mask].mean()) if mask.any() else None,
            observed_lost_clearance_rate=float(outcome["lost_clearance"][mask].mean()) if mask.any() else None,
            observed_joint_last3_rate=float(outcome["joint_last3"][mask].mean()) if mask.any() else None,
            observed_retained_rate=float(outcome["retained"][mask].mean()) if mask.any() else None,
            program_frame_counts=torch.bincount(program.flatten(), minlength=8).tolist(),
            nonbase_frames=int((program != 4).sum()),
            physical_changed_base_frames=int(changed_base[mask].sum()),
            physical_changed_fixed_frames=int(changed_fixed[mask].sum()),
        )

    calibrated = assignment == 5
    cm = assignment == 0
    cal_choice = recommendations[..., 5]
    cm_choice = recommendations[..., 0]
    changed_vs_cm = int((cal_choice[calibrated] != cm_choice[calibrated]).sum()) if calibrated.any() else 0
    cal_frames = int(calibrated.sum() * cal_choice.shape[-1])
    primary = contrasts["calibrated_vs_fixed"]
    label = "PROMISING" if adequate and primary["gate"]["passed"] else ("UNPROMISING" if adequate else "UNCLEAR")
    result = dict(
        schema="ref2dex.relative_task_value_native_analysis.v1",
        experiment_id="P-20261003-relative-task-value-native",
        run_status="COMPLETED", label=label, rows=n, policies=policies,
        contrasts=contrasts, support_adequate=adequate,
        calibrated_windows=int(calibrated.sum()), calibrated_decision_frames=cal_frames,
        calibrated_recommendation_changes_vs_cm=changed_vs_cm,
        calibrated_recommendation_change_fraction_vs_cm=(changed_vs_cm / cal_frames if cal_frames else None),
        source_records=[str(path.resolve()) for path in args.records],
        scope="bounded randomized native H10 controller windows; frozen Cm physical guard plus frozen relative task-value head; no PPO or final success claim",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
