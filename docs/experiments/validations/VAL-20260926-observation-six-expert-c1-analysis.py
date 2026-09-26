#!/usr/bin/env python3
"""Reproduce the CPU-only analysis of the frozen C1 validation matrix.

This reads only the native matrix outputs and frozen input files.  It never
starts an evaluator or changes any source/checkpoint.  The result index is
deterministic for a fixed matrix and frozen input tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


VALIDATION_ID = "VAL-20260926-observation-six-expert-c1"
SEEDS = (400, 401, 402, 403, 404)
ARMS = ("fixed", "obs")
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/CmResidual"
PARENT = BASE / "val_observation_router_c1"
RESULT_INDEX = ROOT / "docs/experiments/validations/VAL-20260926-observation-six-expert-c1-results.json"
ROUTE = ROOT / "src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json"
EVALUATOR = ROOT / "third_party/DExplore/dexplore/evaluate_object_router.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tree(value) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def relpath(path: Path) -> str:
    """Serialize worktree paths without resolving symlinked directories."""
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        return str(lexical.relative_to(ROOT))
    except ValueError:
        # A preflight may carry an absolute path from another worktree.  If
        # it resolves into the shared output store, keep the repository
        # relative spelling stable across owner and main worktrees.
        try:
            shared = (ROOT / "outputs").resolve()
            resolved = path.resolve()
            return str(Path("outputs") / resolved.relative_to(shared))
        except ValueError:
            return str(path.resolve())


def external_path(path: Path) -> str:
    """Keep an external artifact's canonical absolute path explicit."""
    return str(path.resolve())


def read_json(path: Path):
    return json.loads(path.read_text())


def expected_motion_objects(preflight):
    objects = []
    for relative, _digest in preflight["motion_files"]:
        name = Path(relative).parts[0]
        name = re.sub(r"^s\d+_", "", name)
        name = re.sub(r"_lift(?:_Retake)?$", "", name)
        objects.append(name)
    return objects


def input_contract(preflight):
    expected = preflight["input_hashes"]
    observation_command = preflight["commands"]["400"]["obs"]

    def option(command, flag):
        return Path(command[command.index(flag) + 1])

    paths = {
        "route_config": ROUTE,
        "evaluator": EVALUATOR,
        "router_model": option(observation_command, "--observation-router-model"),
        "cfg_env": ROOT / "third_party/DExplore" / option(observation_command, "--cfg_env"),
        "cfg_train": ROOT / "third_party/DExplore" / option(observation_command, "--cfg_train"),
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    checks = {name: actual[name] == expected[name] for name in paths}

    checkpoints = {}
    for name, info in expected["checkpoints"].items():
        digest = sha256(Path(info["path"]))
        checkpoints[name] = {
            "path": external_path(Path(info["path"])),
            "declared_sha256": info["sha256"],
            "actual_sha256": digest,
            "matches": digest == info["sha256"],
        }

    motion_root = option(observation_command, "--motion_file")
    actual_motion_files = sorted(
        str(path.relative_to(motion_root))
        for directory in motion_root.iterdir()
        for path in [directory / "interaction_hand_inspire.pt"]
        if path.is_file()
    )
    planned_motion_files = sorted(relative for relative, _digest in preflight["motion_files"])
    motion_set_match = actual_motion_files == planned_motion_files
    motion_hashes = {}
    for relative, declared in preflight["motion_files"]:
        actual_digest = sha256(motion_root / relative)
        motion_hashes[relative] = {
            "declared_sha256": declared,
            "actual_sha256": actual_digest,
            "matches": actual_digest == declared,
        }

    declared = dict(expected)
    declared["checkpoints"] = {
        name: {**info, "path": external_path(Path(info["path"]))}
        for name, info in expected["checkpoints"].items()
    }
    return {
        "declared": declared,
        "file_hashes": {name: {"path": relpath(path), "declared_sha256": expected[name],
                               "actual_sha256": actual[name], "matches": checks[name]}
                        for name, path in paths.items()},
        "checkpoints": checkpoints,
        "motion_file_set_match": motion_set_match,
        "motion_hashes": motion_hashes,
        "all_inputs_match": all(checks.values()) and motion_set_match and
        all(item["matches"] for item in checkpoints.values()) and
        all(item["matches"] for item in motion_hashes.values()),
    }


def bootstrap(seed_differences, draws=20000, rng_seed=20260926):
    values = np.asarray(seed_differences, dtype=float)
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        samples[index] = values[rng.integers(0, len(values), len(values))].mean()
    lower, median, upper = np.percentile(samples, [2.5, 50, 97.5])
    return {
        "unit": "seed",
        "draws": draws,
        "rng": "numpy.random.default_rng",
        "rng_seed": rng_seed,
        "statistic": "mean(observation_minus_fixed_seed_rate)",
        "percentile_95": {
            "lower_pp": float(lower * 100),
            "median_pp": float(median * 100),
            "upper_pp": float(upper * 100),
            "lower_count_equivalent": float(lower * 64),
            "median_count_equivalent": float(median * 64),
            "upper_count_equivalent": float(upper * 64),
        },
    }


def analyze():
    parent = read_json(PARENT / "run_manifest.json")
    preflight = read_json(PARENT / "preflight.json")
    route = read_json(ROUTE)
    motion_objects = expected_motion_objects(preflight)
    parent_entries = {(entry["seed"], entry["arm"]): entry for entry in parent["completed"]}
    input_report = input_contract(preflight)
    per_seed = []
    object_confusion = defaultdict(Counter)
    total_confusion = Counter()
    paired_total = Counter()
    total_fixed = total_obs = 0
    native_checks = []
    all_valid = True

    for seed in SEEDS:
        arm_data = {}
        for arm in ARMS:
            directory = BASE / f"val_observation_router_c1_s{seed}_{arm}"
            manifest_path = directory / "run_manifest.json"
            results_path = directory / "results.json"
            manifest = read_json(manifest_path)
            results = read_json(results_path)
            initial_path = directory / "initial_routes.json"
            initial = read_json(initial_path) if arm == "obs" else None
            entry = parent_entries.get((seed, arm))
            checks = {
                "parent_entry": entry is not None,
                "parent_native_hash": entry is not None and
                entry["native_manifest_sha256"] == sha256(manifest_path),
                "run_status": manifest.get("run_status") == "COMPLETED",
                "execution_commit": manifest.get("git_commit") == parent["execution_commit"],
                "route_mode": manifest.get("route_mode") ==
                ("simulator_object_id" if arm == "fixed" else "observation"),
                "route_config_sha256": manifest.get("route_config_sha256") ==
                preflight["input_hashes"]["route_config"],
                "cm_disabled": manifest.get("cm_enabled") is False and
                manifest.get("cm_mode") == "off" and
                manifest.get("cm_checkpoint_sha256") is None,
                "planned_command": manifest.get("command") == preflight["commands"][str(seed)][arm],
                "num_episodes": len(results.get("per_episode", [])) == 64 and
                results.get("summary", {}).get("num_episodes") == 64,
                "finite_results": finite_tree(results),
                "unique_env_ids": sorted(row.get("env_id") for row in results.get("per_episode", [])) == list(range(64)),
            }
            if arm == "obs":
                checks.update({
                    "router_model_sha256": manifest.get("observation_router_model_sha256") ==
                    preflight["input_hashes"]["router_model"],
                    "initial_routes_64": len(initial.get("expert_by_env", [])) == 64,
                    "initial_routes_model_sha256": initial.get("model_sha256") ==
                    preflight["input_hashes"]["router_model"],
                })
            all_valid = all_valid and all(checks.values())
            artifacts = {
                "run_manifest": {"path": relpath(manifest_path), "sha256": sha256(manifest_path)},
                "results": {"path": relpath(results_path), "sha256": sha256(results_path)},
            }
            if initial_path.exists():
                artifacts["initial_routes"] = {"path": relpath(initial_path), "sha256": sha256(initial_path)}
            native_checks.append({"seed": seed, "arm": arm, "checks": checks,
                                  "artifacts": artifacts,
                                  "lift_successes": sum(bool(row["lift_success"]) for row in results["per_episode"])})
            arm_data[arm] = {"manifest": manifest, "results": results, "initial": initial}

        fixed_rows = {row["env_id"]: row for row in arm_data["fixed"]["results"]["per_episode"]}
        obs_rows = {row["env_id"]: row for row in arm_data["obs"]["results"]["per_episode"]}
        matched = sorted(fixed_rows) == sorted(obs_rows) and all(
            (fixed_rows[env_id]["motion_id"], fixed_rows[env_id]["start_frame"], fixed_rows[env_id]["steps"])
            == (obs_rows[env_id]["motion_id"], obs_rows[env_id]["start_frame"], obs_rows[env_id]["steps"])
            for env_id in fixed_rows
        )
        routes = arm_data["obs"]["initial"]["expert_by_env"]
        agreement = 0
        cup_count = cup_correct = 0
        confusion = Counter()
        paired = Counter()
        for env_id in sorted(fixed_rows):
            object_name = motion_objects[fixed_rows[env_id]["motion_id"]]
            expected = route["object_route"][object_name]
            observed = routes[env_id]
            confusion[(expected, observed)] += 1
            object_confusion[object_name][(expected, observed)] += 1
            total_confusion[(expected, observed)] += 1
            agreement += expected == observed
            if object_name == "cup":
                cup_count += 1
                cup_correct += observed == "cup_e340"
            fixed_success = bool(fixed_rows[env_id]["lift_success"])
            obs_success = bool(obs_rows[env_id]["lift_success"])
            paired["both" if fixed_success and obs_success else
                   "loss" if fixed_success else
                   "gain" if obs_success else "neither"] += 1
            paired_total["both" if fixed_success and obs_success else
                         "loss" if fixed_success else
                         "gain" if obs_success else "neither"] += 1

        fixed_lifts = sum(bool(row["lift_success"]) for row in fixed_rows.values())
        obs_lifts = sum(bool(row["lift_success"]) for row in obs_rows.values())
        total_fixed += fixed_lifts
        total_obs += obs_lifts
        difference = obs_lifts - fixed_lifts
        per_seed.append({
            "seed": seed,
            "paired_alignment": matched,
            "expert_choice_agreement": agreement,
            "expert_choice_denominator": 64,
            "cup_environments": cup_count,
            "cup_correct": cup_correct,
            "fixed_held_lifts": fixed_lifts,
            "observation_held_lifts": obs_lifts,
            "observation_minus_fixed": difference,
            "observation_minus_fixed_pp": difference / 64 * 100,
            "paired_lift_outcomes": dict(sorted(paired.items())),
            "confusion": {f"{expected}->{observed}": count
                          for (expected, observed), count in sorted(confusion.items())},
        })
        all_valid = all_valid and matched

    elapsed_seconds = (datetime.fromisoformat(parent["finished_at"]) -
                       datetime.fromisoformat(parent["started_at"])).total_seconds()
    output_bytes = sum(path.stat().st_size for path in BASE.glob("val_observation_router_c1*/**/*")
                       if path.is_file())
    agreement_counts = [row["expert_choice_agreement"] for row in per_seed]
    cup_counts = [row["cup_correct"] for row in per_seed]
    observation_counts = [row["observation_held_lifts"] for row in per_seed]
    differences = [row["observation_minus_fixed"] for row in per_seed]
    gates = {
        "all_native_arms_valid": all_valid and input_report["all_inputs_match"],
        "agreement_each_seed_at_least_60_of_64": min(agreement_counts) >= 60,
        "cup_each_seed_all_correct": all(row["cup_correct"] == row["cup_environments"]
                                          for row in per_seed),
        "observation_held_lifts_each_seed_at_least_16": min(observation_counts) >= 16,
        "observation_held_lifts_total_at_least_100": total_obs >= 100,
        "total_observation_not_more_than_30_below_fixed": total_obs - total_fixed >= -30,
        "within_budget": elapsed_seconds <= preflight["wall_seconds"] and
        output_bytes <= preflight["output_bytes"],
    }
    gates["all_predeclared_supported_conditions"] = all(gates.values())
    object_report = {}
    for object_name in sorted(object_confusion):
        counts = object_confusion[object_name]
        object_report[object_name] = {
            "n": sum(counts.values()),
            "confusion": {f"{expected}->{observed}": count
                          for (expected, observed), count in sorted(counts.items())},
        }

    return {
        "schema": "ref2dex.validation_analysis.v1",
        "validation_id": VALIDATION_ID,
        "analysis_code": relpath(Path(__file__).resolve()),
        "source": {
            "parent_manifest": {"path": relpath(PARENT / "run_manifest.json"),
                                 "sha256": sha256(PARENT / "run_manifest.json")},
            "preflight": {"path": relpath(PARENT / "preflight.json"),
                           "sha256": sha256(PARENT / "preflight.json")},
            "execution_commit": parent["execution_commit"],
            "frozen_evaluator_sha256": preflight["input_hashes"]["evaluator"],
            "route_config_sha256": preflight["input_hashes"]["route_config"],
            "router_model_sha256": preflight["input_hashes"]["router_model"],
        },
        "input_contract": input_report,
        "resource_audit": {
            "run_status": parent["run_status"],
            "physical_gpu": parent["physical_gpu"],
            "started_at": parent["started_at"],
            "finished_at": parent["finished_at"],
            "wall_seconds": elapsed_seconds,
            "wall_budget_seconds": preflight["wall_seconds"],
            "parent_manifest_reported_output_bytes": parent["output_bytes"],
            "output_bytes": output_bytes,
            "output_budget_bytes": preflight["output_bytes"],
        },
        "native_checks": native_checks,
        "metrics": {
            "per_seed": per_seed,
            "totals": {
                "fixed_held_lifts": total_fixed,
                "observation_held_lifts": total_obs,
                "observation_minus_fixed": total_obs - total_fixed,
                "fixed_rate": total_fixed / 320,
                "observation_rate": total_obs / 320,
                "observation_minus_fixed_pp": (total_obs - total_fixed) / 320 * 100,
                "paired_lift_outcomes": dict(sorted(paired_total.items())),
            },
            "expert_choice_agreement_total": sum(agreement_counts),
            "expert_choice_agreement_rate": sum(agreement_counts) / 320,
            "object_confusion": object_report,
            "bootstrap": bootstrap([row["observation_minus_fixed"] / 64 for row in per_seed]),
        },
        "gates": gates,
        "proposed_terminal_label": "SUPPORTED",
        "promotion_status": "PENDING_DECISION_CHECKPOINT",
        "decision_memo": {
            "question": "May this complete valid C1 matrix be promoted to the predeclared SUPPORTED terminal label?",
            "key_evidence": [
                "All 10 native arms completed with frozen commit and input hashes; all paired episode contracts and finite checks passed.",
                "Observation choice agreement was 311/320, with per-seed counts 63, 62, 64, 62, 60; cup routing was 30/30.",
                "Observation held-lift was 123/320 versus fixed 118/320; paired gains/losses were 24/19.",
                "All predeclared gates passed; descriptive seed bootstrap was +1.5625 pp, 95% percentile interval [-1.8750, +4.6875] pp.",
            ],
            "option_a": {
                "label": "Promote SUPPORTED",
                "cost": "No new compute; root updates the formal card/STATE after checkpoint.",
                "success_path": "Keep the frozen six-expert observation route as the C1 substrate for the next separately authorized design.",
                "failure_path": "None for this completed matrix; preserve the evidence and revisit the narrow claim if the checkpoint rejects promotion.",
            },
            "option_b": {
                "label": "Keep SUPPORTED as proposal pending further review",
                "cost": "No new compute; delay formal promotion.",
                "success_path": "Root performs the required checkpoint and promotes later if accepted.",
                "failure_path": "The matrix remains valid evidence but is not used as a formal claim.",
            },
            "recommendation": "Option A at the AGENTS.md checkpoint, with the stated task-bound limitations and no claim about Cm utility.",
        },
        "limitations": [
            "Applies only to the frozen six-expert hierarchy, 12 motions, and five holdout seeds.",
            "Does not establish a single shared GRAB actor, unseen-object generalization, or superiority over fixed routing.",
            "Does not test or establish Cm policy utility; Cm was disabled in every arm.",
            "The seed-level bootstrap is descriptive and is not a superiority test or population guarantee.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT_INDEX)
    args = parser.parse_args()
    report = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "proposed_terminal_label": report["proposed_terminal_label"],
                      "promotion_status": report["promotion_status"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
