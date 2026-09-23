"""Verify and summarize the preregistered V1.45 seed-cluster replication."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics


SEEDS = tuple(range(99, 104))
REPEATS = (0, 1)
POLICIES = ("route", "baseline")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_bootstrap(differences: list[float], *, seed: int = 145,
                      repetitions: int = 10000) -> tuple[float, float]:
    if len(differences) != 5 or repetitions != 10000:
        raise ValueError("V1.45 bootstrap must use five seeds and 10000 draws")
    rng = random.Random(seed)
    draws = sorted(sum(differences[rng.randrange(5)] for _ in range(5)) / 5
                   for _ in range(repetitions))
    return draws[249], draws[9749]


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
        raise ValueError("V1.45 parent manifest is incomplete")
    records = {}
    elapsed = {name: [] for name in POLICIES}
    for entry in parent["completed_runs"]:
        key = (entry["seed"], entry["repeat"], entry["policy"])
        if (key in records or key[0] not in SEEDS or key[1] not in REPEATS or
                key[2] not in POLICIES or not 0 <= entry["successes"] <= 64):
            raise ValueError(f"invalid/duplicate run {key}")
        manifest = json.loads(Path(entry["run_manifest"]).read_text())
        if (manifest.get("run_status") != "COMPLETED" or
                manifest.get("cm_used_by_policy") is not False or
                manifest.get("mode") != "router" or
                round(manifest["summary"]["lift_success_rate"] * 64) != entry["successes"]):
            raise ValueError(f"invalid child manifest {key}")
        map_sha = parent["checkpoint_and_map_sha256"][f"{key[2]}_map"]
        if manifest["router_map_file"]["sha256"] != map_sha:
            raise ValueError(f"route map drift in {key}")
        seconds = (datetime.fromisoformat(entry["completed_at"]) -
                   datetime.fromisoformat(entry["created_at"])).total_seconds()
        elapsed[key[2]].append(seconds)
        records[key] = int(entry["successes"])
    expected = {(seed, repeat, policy) for seed in SEEDS for repeat in REPEATS
                for policy in POLICIES}
    if set(records) != expected:
        raise ValueError("missing V1.45 matrix entries")
    rows = []
    differences = []
    for seed in SEEDS:
        values = {policy: [records[(seed, repeat, policy)] for repeat in REPEATS]
                  for policy in POLICIES}
        difference = sum(values["route"]) - sum(values["baseline"])
        differences.append(difference / 128)
        rows.append({"seed": seed, "route": values["route"],
                     "baseline": values["baseline"],
                     "route_minus_baseline_successes_per_128": difference})
    interval = cluster_bootstrap(differences)
    totals = {policy: sum(value for (seed, repeat, name), value in records.items()
                          if name == policy) for policy in POLICIES}
    difference_fraction = (totals["route"] - totals["baseline"]) / 640
    stable = all(records[(seed, repeat, "route")] >= 58 for seed in SEEDS
                 for repeat in REPEATS)
    improvement_gate = (difference_fraction >= 0.08 and all(value > 0 for value in differences)
                        and interval[0] > 0)
    report = {
        "schema": "ref2dex.v145_replication_analysis.v1",
        "run_status": "COMPLETED", "work_version": "V1.45",
        "parent_manifest": str(args.parent_manifest.resolve()),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "seed_rows": rows, "totals_per_640": totals,
        "route_minus_baseline_fraction": difference_fraction,
        "seed_cluster_bootstrap_95_fraction": list(interval),
        "bootstrap_seed": 145, "bootstrap_repetitions": 10000,
        "route_stability_gate_each_run_58_of_64": stable,
        "route_improvement_gate": improvement_gate,
        "median_wall_seconds_including_setup": {
            policy: statistics.median(elapsed[policy]) for policy in POLICIES},
        "official_policy_checkpoint": None, "cm_used_by_policy": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"totals_per_640": totals,
                      "route_minus_baseline_fraction": difference_fraction,
                      "seed_cluster_bootstrap_95_fraction": interval,
                      "route_stability_gate_each_run_58_of_64": stable,
                      "route_improvement_gate": improvement_gate}, sort_keys=True))


if __name__ == "__main__":
    main()
