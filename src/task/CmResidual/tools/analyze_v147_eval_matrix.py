"""Verify V1.47's fixed heldout s3 matrix and seed-cluster bootstrap."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

from src.task.CmResidual.tools.analyze_v145_replication import cluster_bootstrap, sha256


SEEDS = tuple(range(109, 114))
ARMS = ("mixed", "control")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads(args.parent_manifest.read_text())
    if (parent.get("run_status") != "COMPLETED" or
            parent.get("seeds") != list(SEEDS) or
            parent.get("repeats_per_seed") != 2 or
            len(parent.get("completed_runs", [])) != 20):
        raise ValueError("V1.47 parent manifest is incomplete/drifted")
    records = {}
    summaries = {arm: [] for arm in ARMS}
    elapsed = {arm: [] for arm in ARMS}
    for entry in parent["completed_runs"]:
        key = (entry["seed"], entry["repeat"], entry["arm"])
        if (key in records or key[0] not in SEEDS or key[1] not in (0, 1) or
                key[2] not in ARMS or not 0 <= entry["successes"] <= 64):
            raise ValueError(f"invalid/duplicate V1.47 entry {key}")
        child = json.loads(Path(entry["run_manifest"]).read_text())
        if (child.get("run_status") != "COMPLETED" or
                child.get("checkpoint_sha256") != parent["checkpoint_sha256"][key[2]] or
                child.get("input_manifest_sha256") !=
                parent["evaluation_input_manifest_sha256"] or
                child.get("early_termination_disabled") is not True or
                child["summary"].get("selector_enabled") is not False or
                round(child["summary"]["lift_success_rate"] * 64) != entry["successes"]):
            raise ValueError(f"invalid child manifest {key}")
        seconds = (datetime.fromisoformat(entry["completed_at"]) -
                   datetime.fromisoformat(entry["created_at"])).total_seconds()
        records[key] = int(entry["successes"])
        summaries[key[2]].append(child["summary"])
        elapsed[key[2]].append(seconds)
    expected = {(seed, repeat, arm) for seed in SEEDS for repeat in (0, 1)
                for arm in ARMS}
    if set(records) != expected:
        raise ValueError("missing V1.47 matrix entries")
    rows = []
    differences = []
    for seed in SEEDS:
        values = {arm: [records[(seed, repeat, arm)] for repeat in (0, 1)]
                  for arm in ARMS}
        difference = sum(values["mixed"]) - sum(values["control"])
        differences.append(difference / 128)
        rows.append({"seed": seed, "mixed": values["mixed"],
                     "control": values["control"],
                     "mixed_minus_control_successes_per_128": difference})
    interval = cluster_bootstrap(differences, seed=147)
    totals = {arm: sum(value for (_, _, name), value in records.items()
                       if name == arm) for arm in ARMS}
    difference_fraction = (totals["mixed"] - totals["control"]) / 640
    benefit = (difference_fraction >= 0.08 and all(value > 0 for value in differences)
               and interval[0] > 0)
    stable = all(records[(seed, repeat, "mixed")] >= 58 for seed in SEEDS
                 for repeat in (0, 1))
    report = {
        "schema": "ref2dex.v147_multimotion_analysis.v1",
        "run_status": "COMPLETED", "work_version": "V1.47",
        "parent_manifest": str(args.parent_manifest.resolve()),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "seed_rows": rows, "totals_per_640": totals,
        "mixed_minus_control_fraction": difference_fraction,
        "seed_cluster_bootstrap_95_fraction": list(interval),
        "bootstrap_seed": 147, "bootstrap_repetitions": 10000,
        "mixed_benefit_gate": benefit,
        "mixed_stability_gate_each_run_58_of_64": stable,
        "mean_max_contact_lift_m": {
            arm: statistics.mean(s["mean_max_contact_lift_m"] for s in summaries[arm])
            for arm in ARMS},
        "mean_hand_object_contact_fraction": {
            arm: statistics.mean(s["mean_hand_object_contact_fraction"] for s in summaries[arm])
            for arm in ARMS},
        "median_wall_seconds_including_setup": {
            arm: statistics.median(elapsed[arm]) for arm in ARMS},
        "official_policy_checkpoint": None,
        "cm_loaded_at_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "totals_per_640", "mixed_minus_control_fraction",
        "seed_cluster_bootstrap_95_fraction", "mixed_benefit_gate",
        "mixed_stability_gate_each_run_58_of_64")}, sort_keys=True))


if __name__ == "__main__":
    main()
