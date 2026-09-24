"""Audit expert-origin option availability on self-trained actor states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.CmResidual.tools.analyze_crossobject_action import bootstrap_ci, estimate
from src.task.CmResidual.tools.analyze_finger_primer_lift import SELF_TRAINED_SHA
from src.task.CmResidual.tools.probe_expert_crossobject_cm import OFFICIAL_SHA, SPLIT_SHA
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
            manifest.get("source_actor_role") != "self_trained" or
            manifest.get("candidate_actor_role") != "official_data_collector" or
            manifest.get("source_checkpoint_sha256") != SELF_TRAINED_SHA or
            manifest.get("candidate_checkpoint_sha256") != OFFICIAL_SHA or
            manifest.get("motion_manifest_sha256") != SPLIT_SHA or
            manifest.get("transitions_sha256") != sha256(args.input)):
        raise ValueError("dual-policy option provenance drift")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.expert_option_h20.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("source_actor_role") != "self_trained" or
            payload.get("candidate_actor_role") != "official_data_collector" or
            payload.get("option_horizon") != 10 or
            payload.get("followup_horizon") != 20):
        raise ValueError("dual-policy option schema drift")
    rows = payload["records"]
    a = rows["assignment"].reshape(-1)
    if (not torch.isin(a, torch.tensor([-1, 0, 1])).all() or
            not torch.equal(rows["option_steps"].reshape(-1), (a != 0).long() * 10) or
            not torch.allclose(rows["executed_action"][a == 1],
                               rows["expert_action"][a == 1]) or
            not torch.allclose(rows["executed_action"][a != 1],
                               rows["base_action"][a != 1])):
        raise ValueError("executed dual-policy arm mismatch")
    effect = estimate(rows, horizon=20, num_envs=128)
    report = {key: value for key, value in effect.items()
              if key not in ("outcome", "contact_fraction", "assignment",
                             "strata", "num_envs")}
    report.update(schema="ref2dex.expert_option_availability_report.v1",
                  classification="Probe", input_sha256=sha256(args.input),
                  environment_cluster_95ci_mm=bootstrap_ci(
                      effect, draws=500, seed=241000),
                  contact_fraction_cluster_95ci=bootstrap_ci(
                      effect, draws=500, seed=241001, metric="contact_fraction"),
                  candidate_action_gap_l2_mean=float(
                      (rows["expert_action"] - rows["base_action"])[a != 0]
                      .norm(dim=-1).mean()),
                  option_action_gap_l2_mean_per_step=float(
                      rows["option_action_gap_sum"][a != 0].mean() / 10))
    report["continue_to_replication"] = (
        all(value["treated"] >= 50 for value in report["by_object"].values()) and
        sum(value["contact_supported_dz_plus_minus_mm"] > 0
            for value in report["by_object"].values()) >= 2 and
        report["pooled_contact_supported_dz_plus_minus_mm"] >= 10 and
        report["pooled_contact_fraction_plus_minus"] >= .05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled_mm": report["pooled_contact_supported_dz_plus_minus_mm"],
                      "ci_mm": report["environment_cluster_95ci_mm"],
                      "contact_fraction": report["pooled_contact_fraction_plus_minus"],
                      "by_object": report["by_object"],
                      "action_gap": report["candidate_action_gap_l2_mean"],
                      "continue_to_replication": report["continue_to_replication"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
