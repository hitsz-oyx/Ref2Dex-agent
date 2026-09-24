"""Audit randomized ten-step grip+lift versus lift-only option."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.CmResidual.randomized_action import FINGER_SYNERGY_INDICES
from src.task.CmResidual.tools.analyze_crossobject_action import bootstrap_ci, estimate
from src.task.CmResidual.tools.analyze_finger_primer_lift import SELF_TRAINED_SHA
from src.task.CmResidual.tools.probe_expert_crossobject_cm import SPLIT_SHA
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    manifest = json.loads((args.input.parent / "run_manifest.json").read_text())
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("source_partition") != "train" or
            manifest.get("source_actor_role") != "self_trained" or
            manifest.get("checkpoint_sha256") != SELF_TRAINED_SHA or
            manifest.get("motion_manifest_sha256") != SPLIT_SHA or
            manifest.get("transitions_sha256") != sha256(args.input) or
            manifest.get("source_objects") != ["airplane", "mug", "toothpaste"]):
        raise ValueError("sustained-option source provenance drift")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.sustained_grip_lift_h20.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("followup_horizon") != 20 or
            payload.get("sustained_grip_lift") is not True or
            payload.get("delta_z_action") != .2 or
            payload.get("second_delta_action") != .1 or
            payload.get("finger_indices") != list(FINGER_SYNERGY_INDICES)):
        raise ValueError("sustained-option schema drift")
    rows = payload["records"]
    assignment = rows["assignment"].reshape(-1)
    if not torch.isin(assignment, torch.tensor([-1, 0, 1])).all():
        raise ValueError("invalid sustained option arm")
    wrist_sum = rows["option_wrist_increment_sum"].reshape(-1)
    finger_sum = rows["option_finger_increment_sum"]
    if (not torch.equal(rows["option_steps"].reshape(-1), (assignment != 0).long() * 10) or
            not torch.allclose(finger_sum[assignment == -1],
                               torch.zeros_like(finger_sum[assignment == -1])) or
            not torch.allclose(wrist_sum[assignment == 0],
                               torch.zeros_like(wrist_sum[assignment == 0])) or
            (wrist_sum[assignment != 0] < -1e-5).any() or
            (wrist_sum[assignment != 0] > 1.00001).any() or
            (finger_sum[assignment == 1] < -1e-5).any() or
            (finger_sum[assignment == 1] > 2.00001).any()):
        raise ValueError("invalid actual bounded option increments")
    first = rows["executed_action"] - rows["base_action"]
    base = rows["base_action"]
    selected = assignment != 0
    grip = assignment == 1
    expected = torch.zeros_like(first)
    expected[selected, 2] = (base[selected, 2] + .1).clamp(-1, 1) - base[selected, 2]
    indices = list(FINGER_SYNERGY_INDICES)
    expected[:, indices] = ((base[:, indices] + grip[:, None].float() * .2).clamp(-1, 1)
                            - base[:, indices])
    if not torch.allclose(first, expected, atol=1e-5):
        raise ValueError("first executed option action mismatch")
    result = estimate(rows, horizon=20, num_envs=128)
    report = {key: value for key, value in result.items()
              if key not in ("outcome", "contact_fraction", "assignment",
                             "strata", "num_envs")}
    report.update(schema="ref2dex.sustained_grip_lift_effect_report.v1",
                  classification="Probe", input_sha256=sha256(args.input),
                  environment_cluster_95ci_mm=bootstrap_ci(
                      result, draws=500, seed=240990),
                  contact_fraction_cluster_95ci=bootstrap_ci(
                      result, draws=500, seed=240991, metric="contact_fraction"),
                  executed_option={
                      "lift_only_wrist_sum_mean": float(wrist_sum[assignment == -1].mean()),
                      "grip_lift_wrist_sum_mean": float(wrist_sum[assignment == 1].mean()),
                      "grip_finger_sum_mean": float(finger_sum[assignment == 1].mean()),
                      "grip_finger_saturation_fraction": float(
                          (finger_sum[assignment == 1] < 2 - 1e-4).float().mean())})
    report["continue_to_replication"] = (
        all(value["treated"] >= 50 for value in report["by_object"].values()) and
        sum(value["contact_supported_dz_plus_minus_mm"] > 0
            for value in report["by_object"].values()) >= 2 and
        report["pooled_contact_supported_dz_plus_minus_mm"] >= 10 and
        report["pooled_contact_fraction_plus_minus"] >= 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_object": report["by_object"],
                      "pooled_mm": report["pooled_contact_supported_dz_plus_minus_mm"],
                      "ci_mm": report["environment_cluster_95ci_mm"],
                      "contact_fraction": report["pooled_contact_fraction_plus_minus"],
                      "executed_option": report["executed_option"],
                      "continue_to_replication": report["continue_to_replication"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
