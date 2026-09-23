"""Audit V1.52 three-arm heldout test with seed-cluster uncertainty."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

from src.task.CmResidual.tools.analyze_v145_replication import cluster_bootstrap, sha256
from src.task.CmResidual.tools.run_v152_eval_matrix import SEEDS, FIXED_SHA, INPUT_SHA


ARMS = ("on", "off", "placebo")


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
            parent.get("arms") != list(ARMS) or
            parent.get("input_manifest_sha256") != INPUT_SHA or
            any(expected.get(arm) != digest for arm, digest in FIXED_SHA.items()) or
            len(expected.get("placebo", "")) != 64 or
            len(parent.get("completed_runs", [])) != 30):
        raise ValueError("V1.52 parent manifest is incomplete or drifted")
    records = {}
    summaries = {arm: [] for arm in ARMS}
    elapsed = {arm: [] for arm in ARMS}
    for entry in parent["completed_runs"]:
        key = (entry["seed"], entry["repeat"], entry["arm"])
        if (key in records or key[0] not in SEEDS or key[1] not in (0, 1) or
                key[2] not in ARMS or not 0 <= entry["successes"] <= 64):
            raise ValueError(f"invalid or duplicate V1.52 entry {key}")
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
                        for repeat in (0, 1) for arm in ARMS}:
        raise ValueError("missing V1.52 matrix entries")
    rows = []
    on_placebo, on_off = [], []
    for seed in SEEDS:
        values = {arm: [records[(seed, repeat, arm)] for repeat in (0, 1)]
                  for arm in ARMS}
        rows.append({"seed": seed, **values,
                     "on_minus_placebo_successes_per_128": sum(values["on"]) -
                     sum(values["placebo"]),
                     "on_minus_off_successes_per_128": sum(values["on"]) -
                     sum(values["off"])})
        on_placebo.append(rows[-1]["on_minus_placebo_successes_per_128"] / 128)
        on_off.append(rows[-1]["on_minus_off_successes_per_128"] / 128)
    interval_placebo = cluster_bootstrap(on_placebo, seed=152)
    interval_off = cluster_bootstrap(on_off, seed=153)
    totals = {arm: sum(value for (_, _, name), value in records.items() if name == arm)
              for arm in ARMS}
    difference_placebo = (totals["on"] - totals["placebo"]) / 640
    difference_off = (totals["on"] - totals["off"]) / 640
    report = {
        "schema": "ref2dex.v152_cm_ppo_placebo_analysis.v1",
        "run_status": "COMPLETED", "work_version": "V1.52",
        "parent_manifest": str(args.parent_manifest.resolve()),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "seed_rows": rows, "totals_per_640": totals,
        "on_minus_placebo_fraction": difference_placebo,
        "on_minus_placebo_seed_cluster_bootstrap_95_fraction": list(interval_placebo),
        "on_minus_off_fraction": difference_off,
        "on_minus_off_seed_cluster_bootstrap_95_fraction": list(interval_off),
        "bootstrap_seeds": {"on_placebo": 152, "on_off": 153},
        "bootstrap_repetitions": 10000,
        "learned_sample_alignment_gate": (
            difference_placebo >= 0.08 and all(x > 0 for x in on_placebo) and
            interval_placebo[0] > 0),
        "on_off_replication_gate": (
            difference_off >= 0.08 and all(x > 0 for x in on_off) and
            interval_off[0] > 0),
        "cm_on_stability_gate_each_run_58_of_64": all(
            records[(seed, repeat, "on")] >= 58 for seed in SEEDS for repeat in (0, 1)),
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
        "totals_per_640", "on_minus_placebo_fraction",
        "on_minus_placebo_seed_cluster_bootstrap_95_fraction",
        "learned_sample_alignment_gate", "on_minus_off_fraction",
        "on_minus_off_seed_cluster_bootstrap_95_fraction", "on_off_replication_gate",
        "cm_on_stability_gate_each_run_58_of_64")}, sort_keys=True))


if __name__ == "__main__":
    main()
