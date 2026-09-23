"""Audit V1.51 strict heldout matrix and seed-cluster bootstrap."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

from src.task.CmResidual.tools.analyze_v145_replication import cluster_bootstrap, sha256
from src.task.CmResidual.tools.run_v151_eval_matrix import SEEDS, OFF_SHA, INPUT_SHA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads(args.parent_manifest.read_text())
    expected = parent.get("checkpoint_sha256", {})
    if (parent.get("run_status") != "COMPLETED" or
            parent.get("seeds") != list(SEEDS) or
            parent.get("repeats_per_seed") != 2 or
            parent.get("input_manifest_sha256") != INPUT_SHA or
            expected.get("off") != OFF_SHA or len(expected.get("on", "")) != 64 or
            len(parent.get("completed_runs", [])) != 20):
        raise ValueError("V1.51 parent manifest is incomplete or drifted")
    records = {}
    summaries = {"on": [], "off": []}
    elapsed = {"on": [], "off": []}
    for entry in parent["completed_runs"]:
        key = (entry["seed"], entry["repeat"], entry["arm"])
        if (key in records or key[0] not in SEEDS or key[1] not in (0, 1) or
                key[2] not in ("on", "off") or not 0 <= entry["successes"] <= 64):
            raise ValueError(f"invalid or duplicate V1.51 entry {key}")
        child = json.loads(Path(entry["run_manifest"]).read_text())
        if (child.get("run_status") != "COMPLETED" or
                child.get("checkpoint_sha256") != expected[key[2]] or
                child.get("input_manifest_sha256") != INPUT_SHA or
                child.get("early_termination_disabled") is not True or
                child["summary"].get("selector_enabled") is not False or
                round(child["summary"]["lift_success_rate"] * 64) != entry["successes"]):
            raise ValueError(f"invalid child manifest {key}")
        records[key] = int(entry["successes"])
        summaries[key[2]].append(child["summary"])
        elapsed[key[2]].append((datetime.fromisoformat(entry["completed_at"]) -
                                datetime.fromisoformat(entry["created_at"])).total_seconds())
    if set(records) != {(seed, repeat, arm) for seed in SEEDS
                        for repeat in (0, 1) for arm in ("on", "off")}:
        raise ValueError("missing V1.51 matrix entries")
    rows, differences = [], []
    for seed in SEEDS:
        on = [records[(seed, repeat, "on")] for repeat in (0, 1)]
        off = [records[(seed, repeat, "off")] for repeat in (0, 1)]
        difference = (sum(on) - sum(off)) / 128
        rows.append({"seed": seed, "on": on, "off": off,
                     "on_minus_off_successes_per_128": sum(on) - sum(off)})
        differences.append(difference)
    interval = cluster_bootstrap(differences, seed=151)
    totals = {arm: sum(value for (_, _, name), value in records.items() if name == arm)
              for arm in ("on", "off")}
    difference = (totals["on"] - totals["off"]) / 640
    report = {
        "schema": "ref2dex.v151_cm_ppo_weight_analysis.v1",
        "run_status": "COMPLETED", "work_version": "V1.51",
        "parent_manifest": str(args.parent_manifest.resolve()),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "seed_rows": rows, "totals_per_640": totals,
        "cm_on_minus_off_fraction": difference,
        "seed_cluster_bootstrap_95_fraction": list(interval),
        "bootstrap_seed": 151, "bootstrap_repetitions": 10000,
        "cm_benefit_gate": difference >= 0.08 and all(x > 0 for x in differences) and interval[0] > 0,
        "cm_on_stability_gate_each_run_58_of_64": all(
            records[(seed, repeat, "on")] >= 58 for seed in SEEDS for repeat in (0, 1)),
        "mean_max_contact_lift_m": {
            arm: statistics.mean(s["mean_max_contact_lift_m"] for s in summaries[arm])
            for arm in ("on", "off")},
        "mean_hand_object_contact_fraction": {
            arm: statistics.mean(s["mean_hand_object_contact_fraction"] for s in summaries[arm])
            for arm in ("on", "off")},
        "median_wall_seconds_including_setup": {
            arm: statistics.median(elapsed[arm]) for arm in ("on", "off")},
        "official_policy_checkpoint": None,
        "cm_loaded_at_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "totals_per_640", "cm_on_minus_off_fraction", "seed_cluster_bootstrap_95_fraction",
        "cm_benefit_gate", "cm_on_stability_gate_each_run_58_of_64")}, sort_keys=True))


if __name__ == "__main__":
    main()
