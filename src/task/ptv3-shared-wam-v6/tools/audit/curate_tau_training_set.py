#!/usr/bin/env python3
"""Audit and curate a small, identity-agnostic local tau TRAIN panel.

This is a CPU/IO-only pass. It never converts source archives, edits caches, or
starts training. Missing candidate sidecars are explicit audit outcomes: the
brief19 KeyError failure must not turn missing data into admitted windows.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

PHASES = ("approach", "close", "contact", "hold", "unknown")
SOFT_CONTACT_THRESHOLD = 0.25
SOFT_CONTACT_QUERIES = 3
EVENT_HORIZON = 24
HISTORY = 4
DEFAULT_TARGET_WINDOWS = 6000
DEFAULT_ANCHOR_WINDOWS = 512
GIB = 1024 ** 3


def read_json(path: Path):
    return json.loads(path.read_text())


def pct(n: int, d: int) -> float:
    return float(n / d * 100.0) if d else 0.0


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _event_value(event: Mapping[str, np.ndarray], name: str, tick: int, slot: int, default: int = -1) -> int:
    """Read both old (T,1) and new (T,S) event sidecars."""
    values = np.asarray(event[name])
    if values.ndim == 1:
        return int(values[tick]) if 0 <= tick < len(values) else default
    return int(values[tick, slot]) if 0 <= tick < values.shape[0] and 0 <= slot < values.shape[1] else default


def phase_for(event: Mapping[str, np.ndarray], slot: int, tick: int) -> str:
    valid = bool(_event_value(event, "event_valid", tick, slot, 0))
    state = _event_value(event, "state", tick, slot, -1)
    if not valid or state < 0:
        return "unknown"
    kind = _event_value(event, "event_type", tick, slot, -1)
    delta = _event_value(event, "event_delta_steps", tick, slot, EVENT_HORIZON + 1)
    if state == 0:
        return "close" if kind == 1 and delta <= EVENT_HORIZON else "approach"
    if state == 1:
        return "contact" if kind == 2 and delta <= EVENT_HORIZON else "hold"
    return "unknown"


def candidate_windows(data_root: Path, row: Mapping[str, object], split: str) -> Tuple[np.ndarray, Optional[str]]:
    """Load a candidate sidecar without assuming ``windows[split]`` exists."""
    key = str(row.get("key", "unknown"))
    paths: List[Path] = [data_root / "windows" / f"{key}-{split}.npy"]
    declared = row.get("windows", {})
    if isinstance(declared, Mapping) and split in declared:
        raw = Path(str(declared[split]))
        paths.append(raw if raw.is_absolute() else data_root / raw)
    seen = set()
    for path in paths:
        path = path.resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if not path.exists():
            continue
        try:
            values = np.asarray(np.load(path, allow_pickle=False))
        except Exception as exc:
            return np.empty((0, 2), dtype=np.int32), f"candidate_load_error:{type(exc).__name__}"
        if values.ndim != 2 or values.shape[1] != 2:
            return np.empty((0, 2), dtype=np.int32), "candidate_shape_invalid"
        return values.astype(np.int32, copy=False), None
    return np.empty((0, 2), dtype=np.int32), "candidate_sidecar_missing"


def final_index_by_sequence(data_root: Path, split: str):
    values = np.load(data_root / "recipe" / f"{split}.npy", allow_pickle=False)
    grouped = collections.defaultdict(set)
    for seq_id, slot, tick in values:
        grouped[int(seq_id)].add((int(slot), int(tick)))
    return grouped


def source_summary(manifest: Mapping[str, object], data_root: Path, split: str) -> Dict[str, Dict[str, object]]:
    seqs = manifest["sequences"]
    index = np.load(data_root / "recipe" / f"{split}.npy", allow_pickle=False)
    rows: Dict[str, Dict[str, object]] = {}
    for seq_id, slot, _ in index:
        row = seqs[int(seq_id)]
        sid = str(row["source_id"])
        item = rows.setdefault(sid, {"windows": 0, "sequences": set(), "objects": set()})
        item["windows"] += 1
        item["sequences"].add(row["key"])
        objects = row.get("objects", [])
        if int(slot) < len(objects):
            item["objects"].add(objects[int(slot)])
    total = len(index)
    return {sid: {"windows": int(item["windows"]), "window_fraction": pct(item["windows"], total),
                  "sequences": len(item["sequences"]), "objects": len(item["objects"])}
            for sid, item in sorted(rows.items(), key=lambda kv: int(kv[0]))}


def rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    r = np.asarray(rotations, dtype=np.float64)
    trace = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def geometry_key(source_id: int, object_name: str) -> str:
    return f"s{source_id}:{object_name}" if source_id < 3 else object_name


def load_geometry_index(data_root: Path) -> Dict[str, Path]:
    manifest = read_json(data_root / "geometry" / "manifest.json")
    # Geometry manifest paths are relative to the geometry manifest directory
    # and already contain its nested ``geometry/`` component.
    return {str(row["key"]): data_root / "geometry" / str(row["path"]) for row in manifest.get("objects", [])}


def object_extent(geometry_index: Mapping[str, Path], key: str, cache: MutableMapping[str, Dict[str, float]]) -> Dict[str, float]:
    if key in cache:
        return cache[key]
    path = geometry_index.get(key)
    if path is None or not path.exists():
        result = {"diag_m": float("nan"), "max_extent_m": float("nan"), "volume_m3": float("nan")}
    else:
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["surface"], dtype=np.float64)
        extent = np.ptp(points, axis=0)
        result = {"diag_m": float(np.linalg.norm(extent)), "max_extent_m": float(np.max(extent)),
                  "volume_m3": float(np.prod(extent))}
    cache[key] = result
    return result


def row_features(row: Mapping[str, object], slot: int, tick: int, joints: np.ndarray,
                 object_poses: np.ndarray, geometry_index: Mapping[str, Path],
                 extent_cache: MutableMapping[str, Dict[str, float]]) -> Dict[str, float]:
    pose0 = np.asarray(object_poses[tick, slot], dtype=np.float64)
    future = np.asarray(object_poses[tick + 1:tick + 25, slot], dtype=np.float64)
    translation = np.linalg.norm(future[:, :3, 3] - pose0[:3, 3], axis=-1)
    relative = np.einsum("ij,tjk->tik", pose0[:3, :3].T, future[:, :3, :3])
    rotation = rotation_angle_deg(relative).reshape(-1)
    hand_future = np.asarray(joints[tick + 1:tick + 25], dtype=np.float64)
    hand_delta = np.linalg.norm(hand_future - np.asarray(joints[tick], dtype=np.float64), axis=-1)
    objects = row.get("objects", [])
    object_name = str(objects[slot])
    ext = object_extent(geometry_index, geometry_key(int(row["source_id"]), object_name), extent_cache)
    return {"geometry_diag_m": ext["diag_m"], "geometry_max_extent_m": ext["max_extent_m"],
            "geometry_volume_m3": ext["volume_m3"],
            "object_translation_range_m": float(np.nanmax(translation)) if len(translation) else 0.0,
            "object_rotation_range_deg": float(np.nanmax(rotation)) if len(rotation) else 0.0,
            "hand_motion_range_m": float(np.nanmax(hand_delta)) if len(hand_delta) else 0.0}


def add_bins(rows: Sequence[MutableMapping[str, object]], field: str, name: str) -> Dict[str, float]:
    values = np.asarray([float(r["features"].get(field, np.nan)) for r in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        for row in rows:
            row["features"][name] = "unknown"
        return {"q33": float("nan"), "q66": float("nan")}
    q33, q66 = np.quantile(finite, [1 / 3, 2 / 3])
    for row in rows:
        value = float(row["features"].get(field, np.nan))
        row["features"][name] = "unknown" if not np.isfinite(value) else ("low" if value <= q33 else "mid" if value <= q66 else "high")
    return {"q33": float(q33), "q66": float(q66)}


def near_duplicate_fraction(rows: Sequence[Mapping[str, object]]) -> float:
    by_key = collections.defaultdict(list)
    for row in rows:
        by_key[(row["sequence_key"], int(row["slot"]))].append(int(row["tick"]))
    marked = set()
    for key, ticks in by_key.items():
        ordered = sorted(ticks)
        for i, start in enumerate(ordered):
            for end in ordered[i + 1:]:
                if end - start >= EVENT_HORIZON:
                    break
                marked.add((key, start)); marked.add((key, end))
    return pct(len(marked), len(rows))


def allocate_counts(total: int, counts: Mapping[object, int], weights: Optional[Mapping[object, float]] = None) -> Dict[object, int]:
    if total <= 0 or not counts:
        return {key: 0 for key in counts}
    weights = weights or {key: 1.0 for key in counts}
    active = [key for key, count in counts.items() if count > 0]
    result = {key: 0 for key in counts}
    remaining = min(int(total), sum(int(counts[key]) for key in active))
    if not active:
        return result
    denom = sum(float(weights.get(key, 1.0)) for key in active)
    raw = {key: remaining * float(weights.get(key, 1.0)) / denom for key in active}
    for key in active:
        result[key] = min(int(counts[key]), int(math.floor(raw[key])))
    left = remaining - sum(result.values())
    while left:
        candidates = [key for key in active if result[key] < counts[key]]
        if not candidates:
            break
        key = max(candidates, key=lambda k: (raw[k] - result[k], str(k)))
        result[key] += 1
        left -= 1
    return result


def choose_rows(rows: Sequence[MutableMapping[str, object]], target: int,
                phase_weights: Mapping[str, float]) -> List[MutableMapping[str, object]]:
    """Select by phase/source, spreading geometry and motion bins."""
    if target >= len(rows):
        return list(rows)
    by_phase = collections.defaultdict(list)
    for row in rows:
        by_phase[str(row["phase"])].append(row)
    phase_quota = allocate_counts(target, {phase: len(by_phase.get(phase, [])) for phase in PHASES}, phase_weights)
    selected: List[MutableMapping[str, object]] = []
    selected_ids = set()
    for phase in PHASES:
        pool = by_phase.get(phase, [])
        if not pool or not phase_quota.get(phase):
            continue
        by_source = collections.defaultdict(list)
        for row in pool:
            by_source[int(row["source_id"])].append(row)
        source_quota = allocate_counts(phase_quota[phase], {sid: len(v) for sid, v in by_source.items()},
                                       {sid: math.sqrt(len(v)) for sid, v in by_source.items()})
        for sid in sorted(by_source):
            group = sorted(by_source[sid], key=lambda r: (str(r["features"].get("geometry_bin", "unknown")),
                str(r["features"].get("motion_bin", "unknown")), str(r["sequence_key"]), int(r["slot"]), int(r["tick"])))
            quota = source_quota.get(sid, 0)
            captures = collections.defaultdict(list)
            for row in group:
                captures[(row["sequence_key"], int(row["slot"]))].append(row)
            captures = {key: values for key, values in sorted(captures.items())}
            pointers = {key: 0 for key in captures}
            last_tick: Dict[Tuple[str, int], int] = {}
            picked = 0
            while picked < quota:
                progressed = False
                for capture, values in captures.items():
                    while pointers[capture] < len(values):
                        row = values[pointers[capture]]; pointers[capture] += 1
                        rid = (row["sequence_key"], int(row["slot"]), int(row["tick"]))
                        if rid in selected_ids:
                            continue
                        previous = last_tick.get(capture)
                        if previous is not None and int(row["tick"]) - previous < EVENT_HORIZON:
                            continue
                        selected.append(row); selected_ids.add(rid); last_tick[capture] = int(row["tick"])
                        picked += 1; progressed = True
                        break
                    if picked >= quota:
                        break
                if not progressed:
                    break
            if picked < quota:
                for row in group:
                    if picked >= quota:
                        break
                    rid = (row["sequence_key"], int(row["slot"]), int(row["tick"]))
                    if rid not in selected_ids:
                        selected.append(row); selected_ids.add(rid); picked += 1
    if len(selected) < min(target, len(rows)):
        rest = [row for row in rows if (row["sequence_key"], int(row["slot"]), int(row["tick"])) not in selected_ids]
        rest.sort(key=lambda r: (str(r["phase"]), int(r["source_id"]), str(r["sequence_key"]), int(r["slot"]), int(r["tick"])))
        selected.extend(rest[:min(target, len(rows)) - len(selected)])
    return selected[:target]


def coverage(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    phase_windows = collections.Counter(str(row["phase"]) for row in rows)
    result = {
        "windows": len(rows),
        "unique_sequences": len({row["sequence_key"] for row in rows}),
        "unique_objects": len({(row["source_id"], row["object"]) for row in rows}),
        "sources": len({int(row["source_id"]) for row in rows}),
        "phase_windows": {phase: int(phase_windows.get(phase, 0)) for phase in PHASES},
        "phase_window_fraction": {phase: pct(int(phase_windows.get(phase, 0)), len(rows)) for phase in PHASES},
        "phase_unique_sequences": {phase: len({row["sequence_key"] for row in rows if row["phase"] == phase}) for phase in PHASES},
        "phase_unique_objects": {phase: len({(row["source_id"], row["object"]) for row in rows if row["phase"] == phase}) for phase in PHASES},
        "geometry_bins": dict(collections.Counter(str(row["features"].get("geometry_bin", "unknown")) for row in rows)),
        "motion_bins": dict(collections.Counter(str(row["features"].get("motion_bin", "unknown")) for row in rows)),
        "rotation_bins": dict(collections.Counter(str(row["features"].get("rotation_bin", "unknown")) for row in rows)),
        "source_windows": {str(k): int(v) for k, v in sorted(collections.Counter(int(row["source_id"]) for row in rows).items())},
        "near_duplicate_window_fraction": near_duplicate_fraction(rows),
    }
    for field in ("geometry_diag_m", "object_translation_range_m", "object_rotation_range_deg", "hand_motion_range_m"):
        values = np.asarray([float(row["features"].get(field, np.nan)) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[field] = {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())} if len(values) else None
    return result


def load_sequence_arrays(row: Mapping[str, object], event_row: Optional[Mapping[str, object]]):
    hand_dir = Path(str(row["path"])); contact_dir = Path(str(row["contact_path"]))
    arrays = {name: np.load(path, mmap_mode="r") for name, path in {
        "joints": hand_dir / "joints.npy", "joint_valid": hand_dir / "joint_valid.npy",
        "object_valid": hand_dir / "object_pose_valid.npy", "object_poses": hand_dir / "object_poses.npy",
        "surface_valid": hand_dir / "surface_valid.npy", "contact": contact_dir / "contact.npy",
        "contact_valid": contact_dir / "valid.npy"}.items()}
    event = {}
    if event_row is not None:
        event_dir = Path(str(event_row["path"]))
        event = {name: np.load(event_dir / f"{name}.npy", mmap_mode="r") for name in
                 ("state", "event_type", "event_delta_steps", "event_valid")}
    return arrays, event


def sequence_batch_summary(args: argparse.Namespace, local_manifest: Mapping[str, object]) -> Dict[str, object]:
    plan_path = args.plan.resolve()
    if not plan_path.exists():
        return {"status": "UNAVAILABLE_NO_PLAN", "requested_sequences": int(args.sequence_target), "selected_sequences": 0}
    plan = read_json(plan_path)
    local_keys = {str(row["key"]) for row in local_manifest.get("sequences", [])}
    pending = [row for row in plan.get("jobs", []) if str(row.get("key")) not in local_keys]
    by_source = collections.defaultdict(list)
    for row in pending:
        by_source[int(row["source_id"])].append(row)
    quotas = allocate_counts(min(args.sequence_target, len(pending)), {sid: len(v) for sid, v in by_source.items()},
                             {sid: math.sqrt(len(v)) for sid, v in by_source.items()}) if args.sequence_target else {}
    selected = []
    for sid in sorted(by_source):
        values = sorted(by_source[sid], key=lambda r: (str(r.get("source_sequence", r.get("key"))), str(r.get("key"))))
        selected.extend(values[:quotas.get(sid, 0)])
    selected = sorted(selected, key=lambda r: str(r.get("key")))
    # The compact plan preserves the original NAS paths, but the payloads for
    # this local workspace are mirrored under the repository outputs tree (for
    # sources 0--2) or recorded in source_row.source_path (for the packed
    # sources).  Report availability separately from conversion: a readable
    # raw payload is not yet an admitted sequence or effective coverage.
    repo_root = Path(__file__).resolve().parents[5]
    def payload_path(job: Mapping[str, object]) -> Optional[Path]:
        sid = int(job.get("source_id", -1))
        if sid < 3:
            raw = str(job.get("sequence_folder", ""))
            if "/outputs/" in raw:
                return repo_root / "outputs" / raw.split("/outputs/", 1)[1]
            return Path(raw)
        source_row = job.get("source_row", {})
        if isinstance(source_row, Mapping) and source_row.get("source_path"):
            return Path(str(source_row["source_path"]))
        return Path(str(job.get("payload", "")))
    available = [row for row in selected if (payload_path(row) or Path(".")).exists()]
    missing = [row for row in selected if row not in available]
    inventory = None
    inventory_path = getattr(args, "inventory", None)
    if inventory_path is not None and Path(inventory_path).exists():
        audit = read_json(Path(inventory_path))
        by_key = {str(row.get("key")): row for row in audit.get("rows", [])}
        values = [int(by_key[str(row["key"])]["estimated_bytes"])
                  for row in selected if str(row["key"]) in by_key]
        inventory = {"path": str(Path(inventory_path).resolve()), "status": audit.get("status"),
                     "checked_sequences": int(audit.get("checked_sequences", 0)),
                     "selected_sequences_with_estimates": len(values),
                     "estimated_bytes": int(sum(values)),
                     "estimated_gib": float(sum(values) / GIB)}
    return {"status": "READY_FOR_BATCH_CONVERSION" if not missing else "PARTIAL_PAYLOADS_AVAILABLE", "plan": str(plan_path),
            "pending_sequences_in_plan": len(pending), "requested_sequences": int(args.sequence_target),
            "selected_sequences": len(selected),
            "source_counts": {str(k): int(v) for k, v in sorted(collections.Counter(int(r["source_id"]) for r in selected).items())},
            "selected_keys": [str(row["key"]) for row in selected],
            "payloads_available": len(available), "payloads_missing": len(missing),
            "missing_keys": [str(row["key"]) for row in missing],
            "inventory": inventory,
            "coverage": "UNKNOWN_UNTIL_CONVERSION; do not call sequence count effective coverage",
            "conversion_started": False,
            "reason": "raw payload paths are resolvable; conversion and admission remain a separate bounded batch"}


def audit(args: argparse.Namespace) -> Dict[str, object]:
    out = args.output.resolve(); data = args.data_root.resolve()
    manifest = read_json(data / "recipe" / "manifest.json")
    if manifest.get("schema") != "ref2dex.wam-v5-sequence-overlay.v1":
        raise ValueError("unexpected recipe schema")
    seqs = manifest["sequences"]
    event_rows = {r["key"]: r for r in read_json(data / "events" / "manifest.json").get("sequences", [])}
    final_by_split = {s: final_index_by_sequence(data, s) for s in ("train", "dev")}
    geometry_index = load_geometry_index(data); extent_cache = {}
    windows: List[Dict[str, object]] = []; admitted = {"train": [], "dev": []}
    rejections = {"train": collections.Counter(), "dev": collections.Counter()}
    missing = {"train": collections.Counter(), "dev": collections.Counter()}; quality = {"train": collections.Counter(), "dev": collections.Counter()}
    sequence_records = []
    for seq_id, row in enumerate(seqs):
        key = str(row["key"]); event_row = event_rows.get(key); splits = {}
        for split in ("train", "dev"):
            candidate, candidate_reason = candidate_windows(data, row, split)
            if candidate_reason is not None:
                missing[split][candidate_reason] += 1
            if not len(candidate):
                if candidate_reason is not None:
                    splits[split] = {"candidate_windows": 0, "admitted_windows": 0, "rejected_windows": 0,
                                     "candidate_status": candidate_reason, "rejection_reasons": {}}
                continue
            arrays, event = load_sequence_arrays(row, event_row); final_set = final_by_split[split].get(seq_id, set()); local_rej = collections.Counter()
            for slot, tick in candidate:
                slot, tick = int(slot), int(tick); reasons = []
                if slot < 0 or slot >= arrays["object_valid"].shape[1]: reasons.append("object_pose_invalid")
                else:
                    # Match the frozen recipe verdict: numpy's clipped slices
                    # are audited as-is.  Candidate rows that run past the
                    # available tail therefore receive the same history/object
                    # reason as the original preparation audit, rather than a
                    # new third admission rule.
                    history = np.asarray(arrays["joint_valid"][max(0, tick - 3):tick + 1])
                    object_valid = np.asarray(arrays["object_valid"][max(0, tick - 3):tick + 25, slot])
                    if not history.all(0).any(): reasons.append("history_missing")
                    if not object_valid.all(): reasons.append("object_pose_invalid")
                reason = "+".join(reasons) if reasons else "accepted"
                # The recipe index is the frozen verdict from the preparation
                # audit.  Recompute the reasons for audit visibility, but do
                # not silently change that verdict in this reporting pass.
                is_final = (slot, tick) in final_set
                if not is_final: local_rej[reason] += 1
                objects = row.get("objects", []); obj = str(objects[slot]) if 0 <= slot < len(objects) else "unknown"
                record = {"split": split, "sequence_id": int(seq_id), "sequence_key": key, "source_id": int(row["source_id"]),
                          "slot": slot, "object": obj, "tick": tick, "phase": phase_for(event, slot, tick) if event else "unknown",
                          "admitted": bool(is_final), "rejection_reason": reason}
                if is_final:
                    if not reasons:
                        record["features"] = row_features(row, slot, tick, arrays["joints"], arrays["object_poses"], geometry_index, extent_cache)
                        target_contact = np.asarray(arrays["contact"][tick + 1:tick + 25, slot]); target_valid = np.asarray(arrays["contact_valid"][tick + 1:tick + 25, slot])
                        if not target_valid.all(): quality[split]["contact_invalid_target"] += 1
                        if not np.asarray(arrays["surface_valid"][tick - 3:tick + 1]).all(): quality[split]["surface_invalid_history"] += 1
                        positive = ((target_contact >= SOFT_CONTACT_THRESHOLD).sum(-1) >= SOFT_CONTACT_QUERIES)
                        quality[split]["positive_contact_target" if positive.any() else "no_positive_contact_target"] += 1
                        if (np.asarray(arrays["contact"][tick, slot]) >= SOFT_CONTACT_THRESHOLD).sum() >= SOFT_CONTACT_QUERIES: quality[split]["contact_current"] += 1
                    else:
                        record["features"] = {"geometry_diag_m": float("nan"), "geometry_max_extent_m": float("nan"),
                                               "geometry_volume_m3": float("nan"), "object_translation_range_m": float("nan"),
                                               "object_rotation_range_deg": float("nan"), "hand_motion_range_m": float("nan")}
                        record["admission_rule_diagnostics"] = reasons
                    admitted[split].append(record)
                windows.append(record)
            rejections[split].update(local_rej)
            splits[split] = {"candidate_windows": int(len(candidate)), "admitted_windows": int(sum(1 for r in windows if r["sequence_id"] == seq_id and r["split"] == split and r["admitted"])),
                             "rejected_windows": int(sum(local_rej.values())), "candidate_status": "loaded", "rejection_reasons": dict(local_rej)}
        sequence_records.append({"sequence_id": int(seq_id), "key": key, "source_id": int(row["source_id"]), "objects": row.get("objects", []), "splits": splits})
    for rows in admitted.values():
        add_bins(rows, "geometry_diag_m", "geometry_bin"); add_bins(rows, "object_translation_range_m", "motion_bin"); add_bins(rows, "object_rotation_range_deg", "rotation_bin")
    phase_weights = {"approach": 0.20, "close": 0.20, "contact": 0.30, "hold": 0.29, "unknown": 0.01}
    curated_train = choose_rows(admitted["train"], min(max(args.target_windows, 0), len(admitted["train"])), phase_weights)
    anchor_train = choose_rows(admitted["train"], min(max(args.anchor_windows, 0), len(admitted["train"])), {phase: 0.20 for phase in PHASES})
    anchor_rows = [{k: r[k] for k in ("sequence_id", "sequence_key", "source_id", "slot", "object", "tick", "phase")} for r in anchor_train]
    anchor_id = identity(anchor_rows)
    phase_report = {}
    for split, rows in admitted.items():
        cov = coverage(rows)
        phase_report[split] = {phase: {"windows": cov["phase_windows"][phase], "window_fraction": cov["phase_window_fraction"][phase],
                                       "unique_sequences": cov["phase_unique_sequences"][phase], "unique_objects": cov["phase_unique_objects"][phase]} for phase in PHASES}
    sequence_batch = sequence_batch_summary(args, manifest)
    # ``out.parent`` is intentionally created only after the preflight.  Query
    # the already existing data filesystem instead of failing on a new output
    # directory.
    free_bytes = shutil.disk_usage(data).free
    requested_estimate_bytes = int(args.estimated_additional_gib * GIB)
    inventory_estimate_bytes = int((sequence_batch.get("inventory") or {}).get("estimated_bytes", 0))
    estimate_bytes = max(requested_estimate_bytes, inventory_estimate_bytes)
    curated = {"schema": "ref2dex.tau-curated-manifest.v2", "status": "CURATED_LOCAL_WINDOWS_NO_TRAINING", "task": "ptv3-shared-wam-v6",
        "experiment_id": "P-20261023-tau-training-set-curation", "git_commit": args.git_commit, "data_root": str(data),
        "environment": {"host": platform.node(), "python": sys.version, "numpy": np.__version__, "device": "CPU/IO only", "gpu_used": False},
        "identity_agnostic_rules": {"admission": [
            {"rule": "H4 hand validity", "meaning": "at least one hand/joint valid through t-3..t", "counted_in": "history_missing"},
            {"rule": "K24 object validity", "meaning": "selected slot pose valid through t-3..t+24", "counted_in": "object_pose_invalid"}],
            "selection": "phase/source/geometry extent bin/motion range bin; object name is never a selection key or threshold",
            "phase": "proximity sidecar with frozen enter/leave/persistence; geometric proximity, not force-contact truth",
            "near_duplicate": "24-frame spacing preferred during selection; residual near repeats are reported"},
        "brief19_failure": {"occurred": True, "failure": "candidate_windows direct lookup raised KeyError for a missing manifest key",
            "repair": "filename/declared-path resolution with explicit missing/invalid counts", "old_reported_verdict": "33,672 TRAIN and 3,356 DEV admitted windows; no 6k selection"},
        "storage_preflight": {"filesystem_free_before_bytes": int(free_bytes), "estimated_additional_conversion_bytes": estimate_bytes,
            "estimated_additional_conversion_gib": float(args.estimated_additional_gib),
            "inventory_estimated_additional_gib": float(inventory_estimate_bytes / GIB),
            "preflight_estimate_gib": float(estimate_bytes / GIB), "safety_floor_bytes": 20 * GIB,
            "estimated_free_after_bytes": int(free_bytes - estimate_bytes), "write_preflight_pass": bool(free_bytes - estimate_bytes >= 20 * GIB),
            "conversion_started": False, "conversion_status": sequence_batch["status"]},
        "source_summary": {split: source_summary(manifest, data, split) for split in ("train", "dev")}, "phase_coverage_all_admitted": phase_report,
        "effective_coverage": {"curated_train": coverage(curated_train), "fixed_train_anchor": coverage(anchor_train),
            "added_sequences_vs_existing_local_cache": 0,
            "pending_sequence_payloads_available": int(sequence_batch.get("payloads_available", 0)),
            "interpretation": "coverage is measured in the 1,177 prepared sequences; pending sequence conversion remains unmeasured"},
        "admission_rejections": {split: {"candidate_windows": int(sum(r["splits"].get(split, {}).get("candidate_windows", 0) for r in sequence_records)),
            "admitted_windows": len(admitted[split]), "rejected_windows": int(sum(rejections[split].values())), "reason_counts": dict(rejections[split]),
            "missing_candidate_sidecars": dict(missing[split]), "reason_fraction_of_rejected": {k: pct(v, sum(rejections[split].values())) for k, v in rejections[split].items()}} for split in ("train", "dev")},
        "quality_not_rejected": {split: dict(quality[split]) for split in ("train", "dev")},
        "selection": {"requested_train_windows": int(args.target_windows), "selected_train_windows": len(curated_train), "target_tolerance": "6000±10%",
            "phase_weights": phase_weights, "fixed_anchor_windows": len(anchor_train), "anchor_id": anchor_id,
            "anchor_selection_pool": "existing local admitted TRAIN; sequence_key/slot/tick stable across expansion",
            "train_window_rows": curated_train, "anchor_rows": anchor_rows},
        "pending_sequence_batch": sequence_batch,
        "goal_contract": {"status": "HOLD_UNTIL_SCHEDULER_CACHE_IDENTITIES", "current_training_goal": "scheduler spatial_path_waypoint",
            "deployment_goal": "scheduler spatial_path_waypoint", "object_goal_valid_hard_code": "removed_and_verified",
            "train_infer_goal_semantics": "spatial_path_waypoint", "contract_receipt": "docs/audit/v6-goal-contract-20261023.md",
            "scheduler_conditioned_cache_identities": "not_ready", "training_allowed": False},
        "exposure": {"prior_flow_matching_updates": 11380, "prior_global_batch": 288, "prior_window_exposures_estimate": 11380 * 288,
            "prior_local_train_windows": len(admitted["train"]), "prior_uniform_exposure_per_window_estimate": (11380 * 288 / len(admitted["train"])) if admitted["train"] else None,
            "same_updates_batch_on_curated_6k_exposure_per_window_estimate": (11380 * 288 / len(curated_train)) if curated_train else None,
            "warning": "updates and global batch determine exposure; poorer TRAIN fit after a larger pool is not evidence that data is invalid"},
        "audit_rows": windows, "sequence_records": sequence_records,
        "reproduction": {"command": "python3 src/task/ptv3-shared-wam-v6/tools/audit/curate_tau_training_set.py --output outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/curated-manifest.json",
                         "inputs": str((data / "recipe" / "manifest.json").resolve())}}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")
    (out.parent / "train-anchor.json").write_text(json.dumps({"schema": "ref2dex.tau-train-anchor.v1", "anchor_id": anchor_id,
        "status": "FIXED_BEFORE_EXPANSION", "source_manifest": str((data / "recipe" / "manifest.json").resolve()), "rows": anchor_rows}, indent=2, sort_keys=True) + "\n")
    return curated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("outputs/ptv3-shared-wam-v6/local-data-20261021-r2/data"))
    parser.add_argument("--plan", type=Path, default=Path("tmp/wam-v5/plan-compact.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/curated-manifest.json"))
    parser.add_argument("--target-windows", type=int, default=DEFAULT_TARGET_WINDOWS); parser.add_argument("--anchor-windows", type=int, default=DEFAULT_ANCHOR_WINDOWS)
    parser.add_argument("--sequence-target", type=int, default=6000); parser.add_argument("--estimated-additional-gib", type=float, default=23.0)
    parser.add_argument("--inventory", type=Path, default=None, help="optional read-only header inventory for the pending batch")
    parser.add_argument("--git-commit", default="working-tree")
    args = parser.parse_args(); result = audit(args)
    print(json.dumps({"status": result["status"], "output": str(args.output), "selected_train_windows": result["selection"]["selected_train_windows"],
                      "anchor_windows": result["selection"]["fixed_anchor_windows"], "pending_sequence_batch": result["pending_sequence_batch"]["selected_sequences"],
                      "coverage": result["effective_coverage"]["curated_train"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
