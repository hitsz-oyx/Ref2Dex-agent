#!/usr/bin/env python3
"""Audit curated TRAIN windows against the inference scheduler contract."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(ROOT / "src/task/ptv3-shared-wam-v5/src"))
sys.path.insert(0, str(ROOT / "src/task/wam-rl-policy-v3/src"))

from ptv3_shared_wam_v6.data import V6Windows
from ptv3_shared_wam_v6.scheduler_cache_identity import (
    GOAL_SEMANTICS, HORIZON_STEPS, SCHEDULER_HZ, cache_identity,
    condition_tensor_hash, scheduler_conditions_for_ticks,
)
from ptv3_shared_wam_v5.storage import load_array, read_json
from ptv3_shared_wam_v5.data import V5Windows


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True).strip()
    except Exception:
        return None


def _geometry(dataset, row):
    source_id = int(row["source_id"])
    sequence = dataset.meta["sequences"][int(row["sequence_id"])]
    name = sequence["objects"][int(row["slot"])]
    key = f"s{source_id}:{name}" if source_id < 3 else name
    entry = dataset.geometry[key]
    path = dataset.geometry_root / entry["path"]
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def _row_index(dataset):
    return {tuple(int(x) for x in row): i for i, row in enumerate(np.asarray(dataset.index))}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument("--data-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    curated = read_json(args.curated_manifest)
    rows = [r for r in curated["selection"]["train_window_rows"] if r.get("admitted") is True]
    # The current curated payload has no planned scheduler reference path.  It
    # only has observed object poses, which are labels and cannot be promoted
    # to an inference instruction.  Keep the replay implementation above for
    # a future reference-bound recipe, but make this receipt conservative.
    scheduler_reference_available = False
    dataset = None
    indices = {}
    legacy_dataset = None
    legacy_indices = {}
    try:
        legacy_recipe = Path(curated["data_root"]) / "recipe" / "manifest.json"
        legacy_dataset = V5Windows(legacy_recipe, "train")
        legacy_indices = _row_index(legacy_dataset)
    except Exception:
        legacy_dataset = None
    grouped = defaultdict(list)
    for row in rows:
        key = (int(row["sequence_id"]), int(row["slot"]))
        grouped[key].append(row)

    identities, excluded = [], []
    legacy_hashes, scheduler_hashes = [], []
    abs_diffs, differing = [], 0
    scheduler_errors = Counter()
    for (sequence_id, slot), group in sorted(grouped.items()):
        source_id = int(group[0]["source_id"])
        ticks = [int(row["tick"]) for row in group]
        reason = "scheduler_reference_path_absent; observed object_poses are labels"
        scheduler_errors[reason] += len(group)
        for row in group:
            excluded.append(dict(source_id=source_id, sample_id=[sequence_id, slot, int(row["tick"])],
                                 window_start_frame=int(row["tick"]), reason=reason))
        if not scheduler_reference_available:
            continue
        sequence, _ = dataset._sequence(sequence_id)
        geometry = _geometry(dataset, group[0])
        conditions = scheduler_conditions_for_ticks(sequence, geometry, source_id, slot, ticks)
        for row in group:
            tick = int(row["tick"])
            key = (sequence_id, slot, tick)
            if key not in indices:
                reason = "curated row absent from frozen TRAIN index"
                scheduler_errors[reason] += 1
                excluded.append(dict(source_id=source_id, sample_id=list(key),
                                     window_start_frame=tick, reason=reason))
                continue
            sample = dataset[indices[key]]
            condition = conditions[tick]
            identity = cache_identity(source_id, sample["sample_id"], tick, condition)
            identities.append(dict(identity=identity["identity"], source_id=source_id,
                                   sample_id=identity["sample_id"], window_start_frame=tick,
                                   goal_semantics=GOAL_SEMANTICS,
                                   horizon_steps=HORIZON_STEPS, scheduler_hz=SCHEDULER_HZ,
                                   condition_sha256=identity["condition_sha256"]))
            scheduler_hashes.append(identity["condition_sha256"])
            legacy = np.asarray(sample["goal_object_delta"], dtype=np.float32)
            legacy_hashes.append(condition_tensor_hash(legacy))
            delta = np.abs(condition.astype(np.float64) - legacy.astype(np.float64))
            abs_diffs.append(float(delta.max()))
            if not np.array_equal(condition, legacy):
                differing += 1

    # Hash the existing endpoint payload separately.  This is an audit of the
    # old contract, never a scheduler admission signal.
    if legacy_dataset is not None:
        for row in rows:
            key = (int(row["sequence_id"]), int(row["slot"]), int(row["tick"]))
            if key in legacy_indices:
                legacy_hashes.append(condition_tensor_hash(
                    legacy_dataset[legacy_indices[key]]["goal_object_delta"]))

    hash_counts = Counter(scheduler_hashes)
    total = len(rows)
    eligible = len(identities)
    unique_hashes = len(hash_counts)
    receipt = dict(
        schema="ref2dex.v6.scheduler-cache-identity-receipt.v1",
        status="BLOCKED_NO_SCHEDULER_REFERENCE" if excluded else "PASS",
        experiment_id="P-20261023-scheduler-cache-identities",
        task="ptv3-shared-wam-v6",
        curated_manifest=str(args.curated_manifest.resolve()),
        data_recipe=str(args.data_recipe.resolve()),
        scheduler_reference_available=scheduler_reference_available,
        git_commit=_git_commit(),
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                         device="CPU/IO only", gpu_used=False),
        contract=dict(goal_semantics=GOAL_SEMANTICS, horizon_steps=HORIZON_STEPS,
                      scheduler_hz=SCHEDULER_HZ,
                      scheduler_path=[
                          "wam_rl_policy_v3.goal_scheduler.OrderedGoalScheduler",
                          "wam_rl_policy_v3.goal_scheduler.update_schedulers",
                      ],
                      semantic_implementations=[
                          "src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/goal_scheduler.py"
                      ]),
        counts=dict(curated_admitted_windows=total, scheduler_eligible_windows=eligible,
                    excluded_unrecomputable_windows=len(excluded),
                    scheduler_condition_hash_unique_count=unique_hashes,
                    scheduler_condition_hash_duplicate_count=eligible - unique_hashes,
                    scheduler_condition_hash_repeat_rate=(
                        float(eligible - unique_hashes) / eligible if eligible else 0.0),
                    legacy_endpoint_hash_unique_count=len(set(legacy_hashes)),
                    legacy_endpoint_hash_duplicate_count=total-len(set(legacy_hashes)),
                    scheduler_vs_legacy_differing_windows=differing,
                    scheduler_vs_legacy_max_abs_delta=max(abs_diffs) if abs_diffs else None),
        identities=identities,
        excluded_windows=excluded,
        exclusion_reasons=dict(scheduler_errors),
        timing=dict(elapsed_s=time.monotonic()-started),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "counts": receipt["counts"],
                      "excluded_reasons": receipt["exclusion_reasons"],
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
