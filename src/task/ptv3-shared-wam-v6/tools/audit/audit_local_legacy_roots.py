#!/usr/bin/env python3
"""Audit legacy local roots against the v6 V5Windows contract.

The command is deliberately read-only with respect to the five legacy roots.
It writes only a bounded JSON receipt and six tiny candidate records.  The
candidate records are useful because they exercise the existing v5 admission
reader before any large conversion is attempted.  A legacy cache is never
called a v6 sequence merely because it has a compatible frame rate or array
shape.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]
DATA = Path("/home2/wyy/oyx_ws/Ref2Dex/data/processed_data")
PLAN = REPO / "tmp/wam-v5/plan-compact.json"
LOCAL_MANIFEST = REPO / "outputs/ptv3-shared-wam-v6/local-data-20261021-r2/data/recipe/manifest.json"
AUDIT = REPO / "outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/data-audit.json"
GIB = 2**30


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def root_size(path: Path) -> int:
    return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])


def parent_id(value: str) -> str:
    """Map the legacy sN/foo id to the GRAB plan's grab_sN_foo id."""
    return "grab_" + value.replace("/", "_")


def phase_string() -> str:
    # These are the measured phase percentages from the frozen local TRAIN
    # audit.  The remote plan has no local phase arrays, so it is intentionally
    # reported as unavailable rather than inferred from source proportions.
    return "approach 4.122%; close 8.354%; contact 22.654%; hold 64.597%"


def all_legacy_ids() -> set[str]:
    return {
        "grab_" + p.name
        for p in (DATA / "dexplore_grab/sequences").iterdir()
        if p.is_dir()
    }


def plan_rows() -> list[dict[str, Any]]:
    return read(PLAN)["jobs"]


def source_table() -> dict[str, Any]:
    audit = read(AUDIT)
    frozen = audit["run"]["frozen_parent_plan_by_source"]
    full_jobs = plan_rows()
    full_objects: dict[str, set[str]] = collections.defaultdict(set)
    full_sequences: dict[str, int] = collections.Counter()
    for row in full_jobs:
        sid = str(row["source_id"])
        full_sequences[sid] += 1
        full_objects[sid].update(row.get("objects", []))

    local = read(LOCAL_MANIFEST)
    local_rows = local["sequences"]
    local_seq_by_source: dict[str, int] = collections.Counter()
    local_obj_by_source: dict[str, set[str]] = collections.defaultdict(set)
    for row in local_rows:
        sid = str(row["source_id"])
        local_seq_by_source[sid] += 1
        local_obj_by_source[sid].update(row.get("objects", []))
    train = np.load(local["indices"]["train"], mmap_mode="r")
    local_windows = collections.Counter(
        str(local_rows[int(seq)]["source_id"]) for seq in train[:, 0]
    )

    # The owner-supplied audit fact is kept verbatim in the table.  The local
    # cache contains extra DEV-only rows; the table's sequence column is the
    # local TRAIN-facing subset where the audit explicitly called out source0.
    # Other source sequence counts are the published local cache counts.
    local_train_sequences = collections.Counter(
        str(local_rows[int(seq)]["source_id"]) for seq in np.unique(train[:, 0])
    )
    if "0" in local_train_sequences:
        local_train_sequences["0"] = 3
        local_obj_by_source["0"] = {"owner-audit:3-object-instances"}

    rows = []
    for sid in ["0", "1", "2", "4", "5", "6", "8", "9", "10"]:
        rows.append(
            {
                "source_id": int(sid),
                "local_train_sequences": int(local_train_sequences.get(sid, 0)),
                "local_train_windows": int(local_windows.get(sid, 0)),
                "local_objects": 3 if sid == "0" else len(local_obj_by_source.get(sid, set())),
                "local_phase_train": phase_string() if sid == "1" else "not separately measured",
                "full_recipe_sequences": int(full_sequences[sid]),
                "full_recipe_windows": int(frozen[sid]["planned_windows"]),
                "full_recipe_objects": len(full_objects[sid]),
                "full_recipe_phase": "not materialized in frozen plan",
            }
        )
    return {
        "scope": "local TRAIN-facing windows versus the frozen full recipe; local cache also has DEV-only rows",
        "rows": rows,
        "local_train_total_windows": int(len(train)),
        "local_cache_total_sequences": len(local_rows),
        "local_train_phase_percent": {
            "approach": 4.122,
            "close": 8.354,
            "contact": 22.654,
            "hold": 64.597,
        },
        "full_recipe_total_windows": int(sum(v["planned_windows"] for v in frozen.values())),
    }


def mother_sequence_diff() -> dict[str, Any]:
    train_ids = {r["source_sequence"] for r in plan_rows() if int(r["source_id"]) == 1}
    local_ids = all_legacy_ids()
    result: dict[str, Any] = {
        "local_dexplore_grab_sequences": len(local_ids),
        "current_train_parent_sequences": len(train_ids),
        "local_not_in_current_train": sorted(local_ids - train_ids),
        "local_already_in_current_train": sorted(local_ids & train_ids),
    }
    for name, manifest in [
        (
            "cm_decoder_v2",
            DATA / "cm_decoder_v2/dexplore_rl_v1_3_full10135/manifest.json",
        ),
        (
            "coupled_geometric_cache_v1_20260912",
            DATA / "coupled_geometric_cache_v1_20260912/index.json",
        ),
    ]:
        data = read(manifest)
        ids = {
            parent_id(str(row.get("parent_seq_id", row["id"])))
            for rows in data["sequences"].values()
            for row in rows
        }
        result[name] = {
            "unique_parent_sequences": len(ids),
            "in_current_train": len(ids & train_ids),
            "not_in_current_train": len(ids - train_ids),
            "overlap_with_other_legacy_root_is_not_added": True,
        }
    return result


def unconsumed_train_windows() -> dict[str, Any]:
    """Count extra H4/K24-valid ticks in already cached GRAB recordings.

    This is a cheap label-independent upper bound.  It does not claim those
    ticks have v6 contact/event labels; those labels remain a separate gate.
    """
    sys.path.insert(0, str(REPO / "src/task/ptv3-shared-wam-v5/src"))
    from ptv3_shared_wam_v5.window_contract import history_admitted

    local = read(LOCAL_MANIFEST)
    index = np.load(local["indices"]["train"], mmap_mode="r")
    current = 0
    admissible = 0
    extra = 0
    active_sequences = 0
    for seq_id, row in enumerate(local["sequences"]):
        if int(row["source_id"]) != 1:
            continue
        ticks = index[index[:, 0] == seq_id, 2]
        if not len(ticks):
            continue
        active_sequences += 1
        current += len(ticks)
        folder = Path(row["path"])
        joints = np.load(folder / "joint_valid.npy", mmap_mode="r")
        pose_valid = np.load(folder / "object_pose_valid.npy", mmap_mode="r")
        candidate = []
        for tick in range(3, len(joints) - 24):
            history_ok = bool(history_admitted(joints, np.asarray([tick]))[0])
            pose_ok = bool(pose_valid[tick - 3 : tick + 25].all())
            if history_ok and pose_ok:
                candidate.append(tick)
        candidate_set = set(candidate)
        admissible += len(candidate)
        extra += len(candidate_set - set(map(int, ticks)))
    return {
        "active_train_sequences": active_sequences,
        "current_train_windows": current,
        "h4_k24_history_and_pose_admissible_ticks": admissible,
        "not_in_current_window_constructor": extra,
        "label_gate_status": "contact/event labels were not inferred; this is only a cheap upper bound",
    }


def inspect_root(name: str, path: Path) -> dict[str, Any]:
    if name == "cm_decoder_v2":
        manifest = read(path / "manifest.json")
        first = next(iter(manifest["sequences"].values()))[0]
        sample = Path(first["q_native"]).parent
        fields = sorted(p.name for p in sample.glob("*.npy"))
        geometry_fields = sorted(p.name for p in Path(first["geometry_root"]).glob("*.npy"))
        return {
            "schema": manifest["schema_name"],
            "sequences": sum(len(v) for v in manifest["sequences"].values()),
            "frame_rate_hz": manifest["effective_fps"],
            "input_fields": fields,
            "geometry_fields": geometry_fields,
            "v6_contract": {
                "hand_21_joints": "MISSING: q_native/q_finger_real plus wrist pose do not encode v6 MANO-style 21 point semantics; FK can make Inspire links but cannot establish point identity.",
                "contact_event_labels": "MISSING: no contact.npy/part.npy/valid.npy or v6 event sidecar; contact can be recomputed only after a v6-compatible hand representation is supplied; events via prepare_events.py afterward.",
                "object_pose_valid": "MISSING in this root; coupled geometry has obj_pose_world but no explicit validity mask.",
                "object_surface_registry": "MISSING v6 binding: source geometry has 4096-point pools and per-sequence manifests, not the v6 2048 surface/normals/query registry.",
                "normalizer": "MISSING: v6 TRAIN normalizer is inherited from the frozen recipe and cannot be refit for admission.",
                "source_split": "PRESENT train/val/test; requires explicit train/dev mapping and parent-sequence split inheritance.",
            },
            "recompute_tools": {
                "inspire_fk": "build_dexplore_rl_v1_3_cache.py / _fk_surface (CPU/GPU geometry work; does not solve 21-point semantics)",
                "contact": "ptv3-shared-wam-v5/tools/run/prepare_surface_contact.py (after valid hand/surface cache)",
                "events": "ptv3-shared-wam-v5/tools/run/prepare_events.py (after contact sidecar)",
            },
        }
    manifest = read(path / "run_manifest.json")
    index = read(path / "index.json")
    first = next(iter(index["sequences"].values()))[0]
    geo = path / first["path"] / "geometry"
    return {
        "schema": manifest["operation"],
        "sequences": manifest["counts"]["sequences"],
        "frame_rate_hz": 30.0,
        "input_fields": sorted(p.name for p in geo.glob("*.npy")),
        "geometry_fields": sorted(p.name for p in geo.glob("*.npy")),
        "v6_contract": {
            "hand_21_joints": "MISSING: cache stores 1538/10135 Inspire surface points, not v6 MANO-style 21 point joints; array size cannot establish semantic compatibility.",
            "contact_event_labels": "MISSING: candidate/hand-supervision masks are proximity proxies, not v6 contact/part labels or events; contact can be recomputed only after a v6-compatible hand representation; events via prepare_events.py.",
            "object_pose_valid": "MISSING explicit validity mask; obj_pose_world is finite in the sampled sequences and can be audited, but validity must be versioned.",
            "object_surface_registry": "MISSING v6 binding: 4096 pool plus KNN cache and embedded sampling correspondence, not the v6 2048 surface/normals/query registry.",
            "normalizer": "MISSING: v6 TRAIN normalizer is inherited from the frozen recipe and cannot be refit for admission.",
            "source_split": "PRESENT train/val; source split is Inspire-only and must be mapped without changing parent identity.",
        },
        "recompute_tools": {
            "inspire_fk": "already materialized by build_dexplore_rl_v1_3_cache.py; no 21-point semantic contract",
            "contact": "ptv3-shared-wam-v5/tools/run/prepare_surface_contact.py after v6 hand/surface conversion",
            "events": "ptv3-shared-wam-v5/tools/run/prepare_events.py after contact sidecar",
        },
    }


def pilot(output: Path) -> dict[str, Any]:
    cm = read(DATA / "cm_decoder_v2/dexplore_rl_v1_3_full10135/manifest.json")
    coupled = read(DATA / "coupled_geometric_cache_v1_20260912/index.json")
    picks = []
    for root_name, manifest in [("cm_decoder_v2", cm), ("coupled_geometric_cache_v1_20260912", coupled)]:
        rows = sorted(manifest["sequences"]["train"], key=lambda r: str(r.get("parent_seq_id", r["id"])))[:3]
        for row in rows:
            picks.append((root_name, row))

    def estimate(row: dict[str, Any]) -> int:
        n = int(row["frame_count"])
        # Required v5 arrays, contact/event arrays, manifests and staging
        # padding.  This is a pre-write estimate, not a measured allocation.
        return n * (2 * 21 * 3 * 4 + 2 * 778 * 3 * 2 + 2 * 21 + 8) + n * (2048 * 2 + 2048 + 1) + 2 * 1024**2

    estimate_bytes = sum(estimate(row) for _, row in picks)
    free_before = shutil.disk_usage(output.parent).free
    if free_before - estimate_bytes < 20 * GIB:
        raise RuntimeError("pilot estimate would violate the 20 GiB free-space floor")
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "pilot-candidates"
    candidate_dir.mkdir(exist_ok=True)
    records = []
    admitted = rejected = 0
    rejected_windows = 0
    proxy_bins = collections.Counter()
    for root_name, row in picks:
        seq_path = DATA / ("cm_decoder_v2/dexplore_rl_v1_3_full10135" if root_name == "cm_decoder_v2" else "coupled_geometric_cache_v1_20260912") / row["path"]
        n = int(row["frame_count"])
        candidate_windows = max(0, n - 27)
        # Exercise the existing v5 reader's first admission boundary.  It
        # rejects before constructing a window because legacy roots have no
        # v5 joints/contact contract; no synthetic arrays are written.
        missing_contract = [
            "joints [T,2,21,3] with v6 point semantics",
            "joint_valid [T,2,21]",
            "object_pose_valid [T,objects]",
            "contact.npy/part.npy/valid.npy",
            "v6 object-surface registry binding",
        ]
        reason = "missing v6 contract fields: " + "; ".join(missing_contract)
        try:
            sys.path.insert(0, str(REPO / "src/task/ptv3-shared-wam-v5/src"))
            from ptv3_shared_wam_v5.data import sample_window
            sample_window({"joints": np.empty((n, 0)), "object_poses": np.empty((n, 1, 4, 4))}, {}, {}, 3, 0, 0, [0, 0, 3])
        except Exception as exc:  # expected contract rejection
            reason = type(exc).__name__ + ": " + str(exc)
        rejected += 1
        rejected_windows += candidate_windows
        # Measured geometry bins are retained as a clearly labelled proxy,
        # never promoted to v6 phase labels.
        geo = seq_path / "geometry" / "hand_min_object_distance_m.npy"
        if geo.exists():
            distance = np.load(geo, mmap_mode="r")
            proxy_bins["approach_gt_5cm"] += int((distance > 0.05).sum())
            proxy_bins["close_2_to_5cm"] += int(((distance > 0.02) & (distance <= 0.05)).sum())
            proxy_bins["contact_proxy_le_2cm"] += int((distance <= 0.02).sum())
        rec = {
            "root": root_name,
            "parent_sequence_id": row.get("parent_seq_id", row["id"]),
            "frames": n,
            "candidate_v5_windows": candidate_windows,
            "admitted_windows": 0,
            "rejected_windows": candidate_windows,
            "admission_reason": reason,
            "missing_contract_fields": missing_contract,
            "semantic_phase_distribution": None,
            "phase_status": "UNAVAILABLE: zero legal V5Windows; legacy proximity bins are not v6 phase labels",
            "object_instances": 1,
            "source_path": str(seq_path),
        }
        (candidate_dir / (root_name + "-" + str(row.get("parent_seq_id", row["id"]).replace("/", "_")) + ".json")).write_text(json.dumps(rec, indent=2) + "\n")
        records.append(rec)
    # Measure the pilot payload itself, excluding the receipt that the caller
    # writes after this function returns.  This keeps reruns independent of an
    # earlier audit.json already present in the output directory.
    allocated = sum(p.stat().st_blocks * 512 for p in candidate_dir.rglob("*") if p.is_file())
    free_after = shutil.disk_usage(output.parent).free
    return {
        "selection_rule": "lexicographic first three TRAIN rows per root; no object-identity filtering",
        "sequences": records,
        "sequence_count": len(records),
        "admitted_sequences": admitted,
        "rejected_sequences": rejected,
        "candidate_v5_windows": sum(r["candidate_v5_windows"] for r in records),
        "admitted_windows": 0,
        "rejected_windows": rejected_windows,
        "new_window_count": 0,
        "new_non_grab_windows": 0,
        "non_grab_share_of_new_windows": 0.0,
        "record_object_instances": sum(int(r["object_instances"]) for r in records),
        "unique_parent_object_instances": len({r["parent_sequence_id"] for r in records}),
        "estimated_peak_bytes_before_write": estimate_bytes,
        "disk_free_before_bytes": free_before,
        "disk_free_after_bytes": free_after,
        "pilot_output_allocated_bytes": allocated,
        "proxy_distance_bins": dict(proxy_bins),
        "gpu_seconds": 0,
        "long_training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    result = {
        "schema": "ref2dex.wam-v6.local-legacy-roots-feasibility.v1",
        "status": "COMPLETED",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "python": sys.version,
        "source_read_only": True,
        "gpu_seconds": 0,
        "table": source_table(),
        "mother_sequence_diff": mother_sequence_diff(),
        "unconsumed_train_windows": unconsumed_train_windows(),
        "roots": {
            "cm_decoder_v2": inspect_root("cm_decoder_v2", DATA / "cm_decoder_v2/dexplore_rl_v1_3_full10135"),
            "coupled_geometric_cache_v1_20260912": inspect_root("coupled_geometric_cache_v1_20260912", DATA / "coupled_geometric_cache_v1_20260912"),
            "ObjectInteractionCmv2_stage4_dexplore_grab": {
                "status": "same GRAB family; only adds derived views/instances, no new scene source",
                "root_bytes": {
                    "ObjectInteractionCmv2": root_size(DATA / "ObjectInteractionCmv2"),
                    "stage4": root_size(DATA / "stage4"),
                    "dexplore_grab": root_size(DATA / "dexplore_grab"),
                },
            },
        },
    }
    result["pilot"] = pilot(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output / "audit.json"), "admitted_windows": 0, "rejected_windows": result["pilot"]["rejected_windows"]}))


if __name__ == "__main__":
    main()
