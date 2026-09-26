"""Analyze the single-airplane sustained grip+lift physical gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.CmResidual.randomized_action import FINGER_SYNERGY_INDICES
from src.task.CmResidual.tools.analyze_crossobject_action import bootstrap_ci, estimate
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


CHECKPOINT_SHA = "5d1f50a21409df5a09f0a471d115d2f06eedce6bc53855a81acf3cc268e6e1a7"
SOURCE_MANIFEST = Path(
    "outputs/CmResidual/agent_s1_airplane_source_v1/manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    manifest = json.loads((args.input.parent / "run_manifest.json").read_text())
    source_sha = sha256(SOURCE_MANIFEST)
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("source_partition") != "train" or
            manifest.get("source_actor_role") != "self_trained" or
            manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
            manifest.get("motion_manifest_sha256") != source_sha or
            manifest.get("transitions_sha256") != sha256(args.input) or
            manifest.get("source_objects") != ["airplane"] or
            manifest.get("seed") != 191):
        raise ValueError("single-airplane option provenance drift")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.sustained_grip_lift_h20.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("followup_horizon") != 20 or
            payload.get("sustained_grip_lift") is not True or
            payload.get("delta_z_action") != .2 or
            payload.get("second_delta_action") != .1 or
            payload.get("finger_indices") != list(FINGER_SYNERGY_INDICES)):
        raise ValueError("single-airplane option schema drift")
    rows = payload["records"]
    assignment = rows["assignment"].reshape(-1)
    if not torch.isin(assignment, torch.tensor([-1, 0, 1])).all():
        raise ValueError("invalid option assignment")
    wrist_sum = rows["option_wrist_increment_sum"].reshape(-1)
    finger_sum = rows["option_finger_increment_sum"]
    selected = assignment != 0
    grip = assignment == 1
    first = rows["executed_action"] - rows["base_action"]
    expected = torch.zeros_like(first)
    base = rows["base_action"]
    expected[selected, 2] = (base[selected, 2] + .1).clamp(-1, 1) - base[selected, 2]
    indices = list(FINGER_SYNERGY_INDICES)
    expected[:, indices] = ((base[:, indices] + grip[:, None].float() * .2).clamp(-1, 1)
                            - base[:, indices])
    if (not torch.equal(rows["option_steps"].reshape(-1), selected.long() * 10) or
            not torch.allclose(first, expected, atol=1e-5) or
            not torch.allclose(finger_sum[assignment == -1],
                               torch.zeros_like(finger_sum[assignment == -1])) or
            not torch.allclose(wrist_sum[selected], torch.ones_like(wrist_sum[selected]),
                               atol=1e-5) or
            not torch.allclose(finger_sum[grip],
                               torch.full_like(finger_sum[grip], 2.), atol=1e-5)):
        raise ValueError("executed sustained option dose mismatch")
    result = estimate(rows, horizon=20, num_envs=128)
    report = {key: value for key, value in result.items()
              if key not in ("outcome", "contact_fraction", "assignment",
                             "strata", "num_envs")}
    ci = bootstrap_ci(result, draws=1000, seed=241191)
    contact_ci = bootstrap_ci(result, draws=1000, seed=241192,
                              metric="contact_fraction")
    item = report["by_object"]["0"]
    plus = int(item["plus"])
    minus = int(item["minus"])
    gate = bool(plus >= 100 and minus >= 100 and
                report["pooled_contact_supported_dz_plus_minus_mm"] >= 5 and
                ci[0] > 0 and report["pooled_contact_fraction_plus_minus"] >= 0)
    report.update(
        schema="ref2dex.s1_task_aligned_option_effect.v1",
        classification="Probe", input_sha256=sha256(args.input),
        source_manifest_sha256=source_sha,
        environment_cluster_95ci_mm=ci,
        contact_fraction_cluster_95ci=contact_ci,
        executed_option={
            "lift_only_wrist_sum_mean": float(wrist_sum[assignment == -1].mean()),
            "grip_lift_wrist_sum_mean": float(wrist_sum[grip].mean()),
            "grip_finger_sum_mean": float(finger_sum[grip].mean()),
        },
        continue_to_seed192_replication=gate,
        limits=["One actor checkpoint and one trajectory.",
                "This physical opportunity test is not evidence of Cm utility."],
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plus": plus, "minus": minus,
                      "pooled_mm": report["pooled_contact_supported_dz_plus_minus_mm"],
                      "ci_mm": ci,
                      "contact_fraction": report["pooled_contact_fraction_plus_minus"],
                      "continue_to_seed192_replication": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
