#!/usr/bin/env python3
"""Audit and materialize the identity-agnostic local tau training curation.

This is an offline audit only.  It never edits caches or starts training.  Phase
labels are derived from the frozen proximity-event sidecars and are explicitly
not physical force/contact labels.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PHASES = ("approach", "close", "contact", "hold", "unknown")
SOFT_CONTACT_THRESHOLD = 0.25
SOFT_CONTACT_QUERIES = 3
EVENT_HORIZON = 24


def read_json(path: Path):
    return json.loads(path.read_text())


def phase_for(event: Dict[str, np.ndarray], slot: int, tick: int) -> str:
    valid = bool(event["event_valid"][tick, slot])
    state = int(event["state"][tick, slot])
    if not valid or state < 0:
        return "unknown"
    kind = int(event["event_type"][tick, slot])
    delta = int(event["event_delta_steps"][tick, slot])
    if state == 0:
        return "close" if kind == 1 and delta <= EVENT_HORIZON else "approach"
    if state == 1:
        return "contact" if kind == 2 and delta <= EVENT_HORIZON else "hold"
    return "unknown"


def add_counter(counter: Dict[str, int], key: str, value: int = 1) -> None:
    counter[key] = int(counter.get(key, 0) + value)


def pct(n: int, d: int) -> float:
    return float(n / d * 100.0) if d else 0.0


def candidate_windows(data_root: Path, key: str, split: str) -> np.ndarray:
    path = data_root / "windows" / f"{key}-{split}.npy"
    return np.load(path) if path.exists() else np.empty((0, 2), dtype=np.int32)


def final_index_by_sequence(data_root: Path, manifest: Dict, split: str):
    values = np.load(data_root / "recipe" / f"{split}.npy")
    grouped = collections.defaultdict(list)
    for seq_id, slot, tick in values:
        grouped[int(seq_id)].append((int(slot), int(tick)))
    return grouped


def source_summary(manifest: Dict, data_root: Path, split: str) -> Dict[str, Dict[str, object]]:
    seqs = manifest["sequences"]
    index = np.load(data_root / "recipe" / f"{split}.npy")
    rows: Dict[str, Dict[str, object]] = {}
    for seq_id, slot, _ in index:
        row = seqs[int(seq_id)]
        sid = str(row["source_id"])
        item = rows.setdefault(sid, {"windows": 0, "sequences": set(), "objects": set()})
        item["windows"] += 1
        item["sequences"].add(row["key"])
        item["objects"].add(row["objects"][int(slot)])
    result = {}
    total = len(index)
    for sid, item in sorted(rows.items(), key=lambda kv: int(kv[0])):
        result[sid] = {
            "windows": int(item["windows"]),
            "window_fraction": pct(item["windows"], total),
            "sequences": len(item["sequences"]),
            "objects": len(item["objects"]),
        }
    return result


def audit(args: argparse.Namespace) -> Dict[str, object]:
    out = args.output
    data = args.data_root
    manifest = read_json(data / "recipe" / "manifest.json")
    if manifest.get("schema") != "ref2dex.wam-v5-sequence-overlay.v1":
        raise ValueError("unexpected recipe schema")
    seqs = manifest["sequences"]
    event_manifest = read_json(data / "events" / "manifest.json")
    event_rows = {r["key"]: r for r in event_manifest["sequences"]}
    final_by_split = {s: final_index_by_sequence(data, manifest, s) for s in ("train", "dev")}
    windows = []
    rejections = {}
    quality = {}
    phase_stats = {}
    phase_objects = {}
    phase_sources = {}
    sequence_records = []
    overlap = {}

    for seq_id, row in enumerate(seqs):
        key = row["key"]
        hand_dir = Path(row["path"])
        contact_dir = Path(row["contact_path"])
        event_dir = Path(event_rows[key]["path"])
        joints = np.load(hand_dir / "joints.npy")
        joint_valid = np.load(hand_dir / "joint_valid.npy")
        object_valid = np.load(hand_dir / "object_pose_valid.npy")
        object_poses = np.load(hand_dir / "object_poses.npy")
        surface_valid = np.load(hand_dir / "surface_valid.npy")
        contact = np.load(contact_dir / "contact.npy")
        contact_valid = np.load(contact_dir / "valid.npy")
        event = {name: np.load(event_dir / f"{name}.npy") for name in
                 ("state", "event_type", "event_delta_steps", "event_valid")}
        seq_splits = {}
        for split in ("train", "dev"):
            candidate = candidate_windows(data, key, split)
            if len(candidate) == 0:
                continue
            final_set = set(final_by_split[split].get(seq_id, ()))
            accepted = 0
            split_rej = collections.Counter()
            phase_count = collections.Counter()
            phase_obj = collections.defaultdict(set)
            phase_src = collections.Counter()
            q = collections.Counter()
            starts = collections.defaultdict(list)
            for slot, tick in candidate:
                slot, tick = int(slot), int(tick)
                history_ok = bool(joint_valid[tick - 3 : tick + 1].all(0).any())
                object_ok = bool(object_valid[tick - 3 : tick + 25, slot].all())
                reasons = []
                if not history_ok:
                    reasons.append("history_missing")
                if not object_ok:
                    reasons.append("object_pose_invalid")
                reason = "+".join(reasons) if reasons else "accepted"
                is_final = (slot, tick) in final_set
                if is_final:
                    accepted += 1
                else:
                    split_rej[reason] += 1
                phase = phase_for(event, slot, tick)
                obj = row["objects"][slot]
                # All quality/phase summaries describe admitted rows. Rejected
                # rows remain in the manifest with their explicit reason.
                if is_final:
                    phase_count[phase] += 1
                    phase_obj[phase].add(obj)
                    phase_src[phase] += 1
                    # These are diagnostics of what the current identity-agnostic
                    # gate does not reject; they do not alter admission.
                    target_contact = contact[tick + 1 : tick + 25, slot]
                    target_valid = contact_valid[tick + 1 : tick + 25, slot]
                    if not target_valid.all():
                        q["contact_invalid_target"] += 1
                    if not surface_valid[tick - 3 : tick + 1].all():
                        q["surface_invalid_history"] += 1
                    positive = ((target_contact >= SOFT_CONTACT_THRESHOLD).sum(-1) >= SOFT_CONTACT_QUERIES)
                    q["positive_contact_target" if positive.any() else "no_positive_contact_target"] += 1
                    if ((contact[tick, slot] >= SOFT_CONTACT_THRESHOLD).sum() >= SOFT_CONTACT_QUERIES):
                        q["contact_current"] += 1
                    object_motion = np.linalg.norm(
                        object_poses[tick + 1 : tick + 25, slot, :3, 3]
                        - object_poses[tick, slot, :3, 3], axis=-1).max()
                    hand_motion = np.linalg.norm(joints[tick + 1 : tick + 25] - joints[tick], axis=-1).max(axis=0).mean()
                    if object_motion < 0.002 and hand_motion < 0.002:
                        q["static_both_lt2mm"] += 1
                    starts[slot].append(tick)
                windows.append({"split": split, "sequence_id": seq_id, "sequence_key": key,
                                "source_id": row["source_id"], "slot": slot, "object": obj,
                                "tick": tick, "phase": phase, "admitted": is_final,
                                "rejection_reason": reason})
            rejections[split] = rejections.get(split, collections.Counter())
            rejections[split].update(split_rej)
            quality[split] = quality.get(split, collections.Counter())
            quality[split].update(q)
            phase_stats[split] = phase_stats.get(split, collections.Counter())
            phase_stats[split].update(phase_count)
            phase_objects.setdefault(split, collections.defaultdict(set))
            phase_sources.setdefault(split, collections.defaultdict(collections.Counter))
            for phase, objects in phase_obj.items():
                phase_objects[split][phase].update(objects)
                phase_sources[split][phase][str(row["source_id"])] += phase_src[phase]
            seq_splits[split] = {"candidate_windows": int(len(candidate)),
                                 "admitted_windows": int(accepted),
                                 "rejected_windows": int(len(candidate) - accepted),
                                 "rejection_reasons": dict(split_rej)}
            for slot, ticks in starts.items():
                ticks = sorted(ticks)
                overlap.setdefault(split, {"exact_duplicate_rows": 0, "near_pairs_delta_lt24": 0,
                                           "windows_in_near_pairs": set(), "adjacent_delta": collections.Counter()})
                if len(ticks) != len(set(ticks)):
                    overlap[split]["exact_duplicate_rows"] += len(ticks) - len(set(ticks))
                overlap[split]["adjacent_delta"].update(b - a for a, b in zip(ticks, ticks[1:]))
                for i, start in enumerate(ticks):
                    j = i + 1
                    while j < len(ticks) and ticks[j] - start < EVENT_HORIZON:
                        overlap[split]["near_pairs_delta_lt24"] += 1
                        overlap[split]["windows_in_near_pairs"].add((seq_id, slot, start))
                        overlap[split]["windows_in_near_pairs"].add((seq_id, slot, ticks[j]))
                        j += 1
        sequence_records.append({"sequence_id": seq_id, "key": key, "source_id": row["source_id"],
                                 "objects": row["objects"], "splits": seq_splits})

    totals = {s: int(len(np.load(data / "recipe" / f"{s}.npy"))) for s in ("train", "dev")}
    phase_report = {}
    for split in ("train", "dev"):
        phase_report[split] = {}
        for phase in PHASES:
            n = int(phase_stats.get(split, {}).get(phase, 0))
            phase_report[split][phase] = {
                "windows": n, "window_fraction": pct(n, totals[split]),
                "objects": len(phase_objects.get(split, {}).get(phase, set())),
                "source_windows": dict(phase_sources.get(split, {}).get(phase, {})),
            }

    source_report = {s: source_summary(manifest, data, s) for s in ("train", "dev")}
    seq_train = {int(x) for x in np.load(data / "recipe" / "train.npy")[:, 0]}
    seq_dev = {int(x) for x in np.load(data / "recipe" / "dev.npy")[:, 0]}
    for split in overlap:
        overlap[split]["windows_in_near_pairs"] = len(overlap[split]["windows_in_near_pairs"])
        overlap[split]["near_window_fraction"] = pct(overlap[split]["windows_in_near_pairs"], totals[split])
        overlap[split]["adjacent_delta"] = dict(sorted(overlap[split]["adjacent_delta"].items()))

    curated = {
        "schema": "ref2dex.tau-curated-manifest.v1",
        "status": "AUDITED_NO_TRAINING",
        "task": "ptv3-shared-wam-v6",
        "experiment_id": "P-20261023-tau-training-set-curation",
        "identity_agnostic_rules": {
            "history": "H4: any hand/joint remains valid across t-3..t",
            "future_object": "object_pose_valid for the selected slot across t-3..t+24",
            "split": "sequence-disjoint TRAIN/DEV; no object-name rule",
            "phase_audit": "proximity sidecar: 3 queries, enter .8/leave .5, persistence 3; labels are geometric, not force contact",
            "static_audit": "diagnostic only: max object and mean hand displacement <2mm over K24; never an admission rule",
        },
        "source_summary": source_report,
        "phase_coverage": phase_report,
        "admission_rejections": {
            split: {"candidate_windows": int(sum(r["splits"].get(split, {}).get("candidate_windows", 0) for r in sequence_records)),
                    "admitted_windows": totals[split], "rejected_windows": int(sum(rejections.get(split, {}).values())),
                    "reason_counts": dict(rejections.get(split, {})),
                    "reason_fraction_of_rejected": {k: pct(v, sum(rejections.get(split, {}).values())) for k, v in rejections.get(split, {}).items()}}
            for split in ("train", "dev")
        },
        "quality_not_rejected": {s: dict(quality.get(s, {})) for s in ("train", "dev")},
        "overlap": overlap,
        "sequence_split_overlap": len(seq_train & seq_dev),
        "sequence_records": sequence_records,
        "windows": windows,
        "goal_contract": {
            "current_training": "goal_object_delta = object pose at realized t+24 minus pose at t",
            "deployment_current": "scheduler spatial_path_waypoint with no endpoint-time guarantee",
            "must_recompute_before_training": [
                "all goal_object_delta/goal_object_points derived at read time from the repaired scheduler-style condition",
                "event/task condition sidecars and their cache identities",
                "recipe manifest, train/dev index identities, TRAIN-only normalizer JSON",
                "held-out evaluation panel inputs and baseline receipts",
            ],
            "preserve": ["trajectory_target K24", "contact labels and masks", "sequence-disjoint split"],
            "estimated_cost": "CPU/IO pass over 1,177 caches and 37,028 windows; no raw-data copy. Any scheduler-conditioned WAM forward and new training require a fresh run and weights-only initialization.",
        },
        "training_recipe": {
            "architecture_loss": "unchanged v6 shared WAM / strict FP32 / joint21 H4 K24 / same trajectory, scene, contact and event losses",
            "data_only_changes": ["identity-agnostic H4/object-valid admission", "phase/source/sequence-balanced sampling with caps", "repaired goal contract"],
            "training_allowed_now": False,
            "devices_after_contract": [4, 5, 6, 7],
        },
        "reproduction": {
            "command": "python3 src/task/ptv3-shared-wam-v6/tools/audit/curate_tau_training_set.py --output outputs/ptv3-shared-wam-v6/tau-curated-20261023-r1/curated-manifest.json",
            "inputs": str((data / "recipe" / "manifest.json").resolve()),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")
    return curated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("outputs/ptv3-shared-wam-v6/local-data-20261021-r2/data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ptv3-shared-wam-v6/tau-curated-20261023-r1/curated-manifest.json"))
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "train": result["admission_rejections"]["train"],
                      "dev": result["admission_rejections"]["dev"],
                      "phase": result["phase_coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
