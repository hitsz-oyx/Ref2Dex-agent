"""Audit two-step finger-primer plus common-lift randomized effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.CmResidual.randomized_action import FINGER_SYNERGY_INDICES
from src.task.CmResidual.tools.analyze_crossobject_action import bootstrap_ci, estimate
from src.task.CmResidual.tools.probe_expert_crossobject_cm import SPLIT_SHA
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


SELF_TRAINED_SHA = "3212bcc195d374a4d1cb31b21051e4a0a93f63fc345a0fe987279a562092dccb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or args.num_envs != 64:
        parser.error("new output and 64 environments required")
    source_manifest = json.loads((args.input.parent / "run_manifest.json").read_text())
    if (source_manifest.get("run_status") != "COMPLETED" or
            source_manifest.get("source_partition") != "train" or
            source_manifest.get("source_actor_role") != "self_trained" or
            source_manifest.get("checkpoint_sha256") != SELF_TRAINED_SHA or
            source_manifest.get("motion_manifest_sha256") != SPLIT_SHA or
            source_manifest.get("transitions_sha256") != sha256(args.input)):
        raise ValueError("self-trained finger-primer source provenance drift")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.randomized_finger_primer_lift_h10.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("followup_horizon") != 10 or
            payload.get("finger_primer_lift") is not True or
            payload.get("delta_z_action") != .2 or
            payload.get("second_delta_action") != .1):
        raise ValueError("finger-primer source schema drift")
    rows = payload["records"]
    assignments = rows["assignment"].reshape(-1)
    first = rows["executed_action"] - rows["base_action"]
    expected = torch.zeros_like(first)
    expected[:, list(FINGER_SYNERGY_INDICES)] = .2 * assignments[:, None]
    second = rows["second_executed_action"] - rows["second_base_action"]
    second_expected = torch.zeros_like(second)
    second_expected[assignments != 0, 2] = .1
    if (not torch.allclose(first, expected, atol=1e-5) or
            not torch.allclose(second, second_expected, atol=1e-5)):
        raise ValueError("first/second executed treatment dose mismatch")
    result = estimate(rows, horizon=10, num_envs=args.num_envs)
    report = {key: value for key, value in result.items()
              if key not in ("outcome", "contact_fraction", "assignment",
                             "strata", "num_envs")}
    report.update(schema="ref2dex.finger_primer_lift_effect_report.v1",
                  classification="Probe", input_sha256=sha256(args.input),
                  environment_cluster_95ci_mm=bootstrap_ci(
                      result, draws=500, seed=240989),
                  contact_fraction_cluster_95ci=bootstrap_ci(
                      result, draws=500, seed=240990, metric="contact_fraction"))
    positives = sum(value["contact_supported_dz_plus_minus_mm"] > 0
                    for value in report["by_object"].values())
    counts = all(value["treated"] >= 50 for value in report["by_object"].values())
    report["continue_to_independent_train_seed"] = (
        counts and positives >= 2 and
        report["pooled_contact_supported_dz_plus_minus_mm"] >= 5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_object": report["by_object"],
                      "pooled_mm": report["pooled_contact_supported_dz_plus_minus_mm"],
                      "ci_mm": report["environment_cluster_95ci_mm"],
                      "contact_fraction": report["pooled_contact_fraction_plus_minus"],
                      "continue_to_independent_train_seed":
                      report["continue_to_independent_train_seed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
