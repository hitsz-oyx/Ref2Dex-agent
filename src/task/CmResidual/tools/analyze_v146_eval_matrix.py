"""Verify V1.46's fixed e300 Cm-on/off heldout matrix and cluster bootstrap."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

from src.task.CmResidual.tools.analyze_v145_replication import cluster_bootstrap, sha256


SEEDS = tuple(range(104, 109))
REPEATS = (0, 1)
ARMS = ("on", "off")
CHECKPOINTS = {
    "on": "278fc7a1b65837de7495d0e5da12a78ab177937a14210ae733134940deac15b2",
    "off": "36ff2ac7ffd4433b5f60b32dce7603cff5db24a0bab9866b9ab469d1ce645929",
}


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
            parent.get("checkpoint_sha256") != CHECKPOINTS or
            len(parent.get("completed_runs", [])) != 20):
        raise ValueError("V1.46 parent manifest is incomplete/drifted")
    records = {}
    summaries = {arm: [] for arm in ARMS}
    elapsed = {arm: [] for arm in ARMS}
    for entry in parent["completed_runs"]:
        key = (entry["seed"], entry["repeat"], entry["arm"])
        if (key in records or key[0] not in SEEDS or key[1] not in REPEATS or
                key[2] not in ARMS or not 0 <= entry["successes"] <= 64):
            raise ValueError(f"invalid/duplicate V1.46 entry {key}")
        child = json.loads(Path(entry["run_manifest"]).read_text())
        if (child.get("run_status") != "COMPLETED" or
                child.get("checkpoint_sha256") != CHECKPOINTS[key[2]] or
                child.get("input_manifest_sha256") != parent["input_manifest_sha256"] or
                child.get("early_termination_disabled") is not True or
                child["summary"].get("selector_enabled") is not False or
                round(child["summary"]["lift_success_rate"] * 64) != entry["successes"]):
            raise ValueError(f"invalid child manifest {key}")
        seconds = (datetime.fromisoformat(entry["completed_at"]) -
                   datetime.fromisoformat(entry["created_at"])).total_seconds()
        records[key] = int(entry["successes"])
        summaries[key[2]].append(child["summary"])
        elapsed[key[2]].append(seconds)
    expected = {(seed, repeat, arm) for seed in SEEDS for repeat in REPEATS
                for arm in ARMS}
    if set(records) != expected:
        raise ValueError("missing V1.46 matrix entries")
    rows = []
    differences = []
    for seed in SEEDS:
        values = {arm: [records[(seed, repeat, arm)] for repeat in REPEATS]
                  for arm in ARMS}
        difference = sum(values["on"]) - sum(values["off"])
        differences.append(difference / 128)
        rows.append({"seed": seed, "on": values["on"], "off": values["off"],
                     "on_minus_off_successes_per_128": difference})
    interval = cluster_bootstrap(differences, seed=146)
    totals = {arm: sum(value for (seed, repeat, name), value in records.items()
                       if name == arm) for arm in ARMS}
    difference_fraction = (totals["on"] - totals["off"]) / 640
    cm_benefit = (difference_fraction >= 0.08 and all(value > 0 for value in differences)
                  and interval[0] > 0)
    stable = all(records[(seed, repeat, "on")] >= 58 for seed in SEEDS
                 for repeat in REPEATS)
    report = {
        "schema": "ref2dex.v146_cm_train_reward_analysis.v1",
        "run_status": "COMPLETED", "work_version": "V1.46",
        "parent_manifest": str(args.parent_manifest.resolve()),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "seed_rows": rows, "totals_per_640": totals,
        "cm_on_minus_off_fraction": difference_fraction,
        "seed_cluster_bootstrap_95_fraction": list(interval),
        "bootstrap_seed": 146, "bootstrap_repetitions": 10000,
        "cm_benefit_gate": cm_benefit,
        "cm_on_stability_gate_each_run_58_of_64": stable,
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
        "totals_per_640", "cm_on_minus_off_fraction",
        "seed_cluster_bootstrap_95_fraction", "cm_benefit_gate",
        "cm_on_stability_gate_each_run_58_of_64")}, sort_keys=True))


if __name__ == "__main__":
    main()
