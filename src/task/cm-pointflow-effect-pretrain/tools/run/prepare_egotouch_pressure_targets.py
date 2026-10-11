#!/usr/bin/env python3
"""Prepare masked, side-specific EgoTouch pressure windows.

The source files are already acquired and hash-checked by the raw-sensor
manifest.  This adapter deliberately keeps the processed pressure grid and its
finite-cell mask; it never rebuilds a sensor mapping from the 256 byte channels
and never uses a recording-wide future maximum as an input statistic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np


HISTORY = 4
HORIZON = 8
TAU = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_rows(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected non-empty rows: {path}")
    return rows


def read_record(source: Path, entry: dict) -> dict:
    record = source / "raw" / entry["record"]
    pressure_rows = read_json_rows(record / "jq_pressure.json")
    hand_rows = read_json_rows(record / "wilor_hands.json")
    if len(pressure_rows) != len(hand_rows):
        raise ValueError(f"row count mismatch: {entry['record']}")
    frame_ids = np.asarray([r["frame_index"] for r in pressure_rows], dtype=np.int64)
    hand_ids = np.asarray([r["frame_index"] for r in hand_rows], dtype=np.int64)
    if not np.array_equal(frame_ids, hand_ids) or not np.array_equal(
        frame_ids, np.arange(len(frame_ids), dtype=np.int64)
    ):
        raise ValueError(f"frame IDs are not contiguous/aligned: {entry['record']}")
    timestamps = np.asarray([r["ts"] for r in pressure_rows], dtype=np.float64)
    if not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        raise ValueError(f"pressure timestamps are not strictly increasing: {entry['record']}")
    raw_sensor = np.asarray(
        [r["sensor_left"] + r["sensor_right"] for r in pressure_rows], dtype=np.float32
    )
    if raw_sensor.shape != (len(pressure_rows), 512) or not np.isfinite(raw_sensor).all():
        raise ValueError(f"raw sensor contract differs: {entry['record']}")
    if raw_sensor.min() < 0 or raw_sensor.max() > 255:
        raise ValueError(f"raw sensor range differs: {entry['record']}")

    grid_path = record / "pressure_grids.npz"
    with np.load(grid_path, allow_pickle=False) as grids:
        grid = np.stack(
            [np.asarray(grids["left_pressure_grid"], dtype=np.float32),
             np.asarray(grids["right_pressure_grid"], dtype=np.float32)],
            axis=1,
        )
        metadata = {
            key: grids[key].item()
            for key in ("grid_size", "tactile_max", "bend_max", "separate_normalization",
                        "baseline_corrected_left", "baseline_corrected_right")
            if key in grids.files and grids[key].ndim == 0
        }
    if grid.shape != (len(pressure_rows), 2, 21, 21):
        raise ValueError(f"unexpected pressure grid shape: {entry['record']} {grid.shape}")
    finite = np.isfinite(grid)
    if not finite.any(axis=(2, 3)).all():
        raise ValueError(f"a side has no measured pressure cells: {entry['record']}")
    if not np.isfinite(grid[finite]).all():
        raise ValueError(f"non-finite pressure values survived mask: {entry['record']}")
    if np.nanmin(grid) < 0 or np.nanmax(grid) > 1:
        raise ValueError(f"processed pressure is outside [0,1]: {entry['record']}")

    hand = np.full((len(hand_rows), 2, 21, 3), np.nan, dtype=np.float32)
    hand_valid = np.zeros((len(hand_rows), 2), dtype=bool)
    for i, row in enumerate(hand_rows):
        for side, key in enumerate(("left_pos", "right_pos")):
            value = np.asarray(row.get(key), dtype=np.float32)
            if value.shape == (21, 3) and np.isfinite(value).all():
                hand[i, side] = value
                hand_valid[i, side] = True

    starts = []
    for side in range(2):
        for start in range(len(grid) - HISTORY - HORIZON + 1):
            stop = start + HISTORY + HORIZON
            # A side-specific pressure sequence is retained even when the other
            # hand is absent.  Past hand shape is optional and explicitly masked.
            if finite[start:stop, side].any(axis=(1, 2)).all():
                starts.append([int(start), int(side)])

    return {
        "record": entry["record"],
        "split": entry["split"],
        "pressure": np.nan_to_num(grid, nan=0.0).astype(np.float32),
        "raw_sensor": raw_sensor,
        "pressure_valid": finite,
        "contact_proxy": finite & (np.nan_to_num(grid, nan=0.0) >= TAU),
        "hand": hand,
        "hand_valid": hand_valid,
        "timestamps": timestamps,
        "source_frame_ids": frame_ids,
        "metadata": metadata,
        "window_starts": starts,
        "source_files": {
            name: sha256(record / name)
            for name in ("jq_pressure.json", "wilor_hands.json", "pressure_grids.npz")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "COMPLETED_DIVERSE_RAW_SAMPLE":
        raise ValueError("source acquisition is not complete")
    manifest = {
        "schema": "ref2dex.egotouch-pressure-targets.v1",
        "status": "PREPARING",
        "training_allowed": False,
        "source_manifest_sha256": sha256(args.source / "manifest.json"),
        "source_revision": source_manifest.get("revision"),
        "protocol": {
            "history": HISTORY,
            "horizon": HORIZON,
            "tau": TAU,
            "pressure_scale": "processed_grid_normalized_0_1",
            "window_unit": "original source rows; timestamps retained",
            "input_future_normalization": False,
            "contact_proxy_is_physical_gt": False,
            "side_specific_windows": True,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "records": [],
    }
    try:
        for entry in source_manifest["selection"]:
            result = read_record(args.source, entry)
            output_name = entry["record"].replace("/", "__") + ".npz"
            np.savez_compressed(
                args.output / output_name,
                pressure=result["pressure"],
                raw_sensor=result["raw_sensor"],
                pressure_valid=result["pressure_valid"],
                contact_proxy=result["contact_proxy"],
                hand=result["hand"],
                hand_valid=result["hand_valid"],
                timestamps=result["timestamps"],
                source_frame_ids=result["source_frame_ids"],
            )
            manifest["records"].append(
                {
                    "record": result["record"],
                    "split": result["split"],
                    "file": output_name,
                    "sha256": sha256(args.output / output_name),
                    "frames": int(len(result["pressure"])),
                    "side_windows": {
                        "left": sum(side == 0 for _, side in result["window_starts"]),
                        "right": sum(side == 1 for _, side in result["window_starts"]),
                    },
                    "measured_cells_per_frame": [
                        int(x)
                        for x in result["pressure_valid"].sum(axis=(2, 3)).mean(axis=0)
                    ],
                    "positive_fraction_measured": [
                        float(
                            result["contact_proxy"][:, side].sum()
                            / result["pressure_valid"][:, side].sum()
                        )
                        for side in range(2)
                    ],
                    "source_files": result["source_files"],
                    "normalization_metadata": result["metadata"],
                }
            )
            print(
                f"{result['split']} {result['record']}: {len(result['pressure'])} frames, "
                f"left/right windows={manifest['records'][-1]['side_windows']['left']}/"
                f"{manifest['records'][-1]['side_windows']['right']}",
                flush=True,
            )
        fit = sum(r["split"] == "fit" for r in manifest["records"])
        held = sum(r["split"] == "held_task" for r in manifest["records"])
        manifest["counts"] = {"fit_records": fit, "held_records": held}
        manifest["status"] = (
            "QUALIFIED_PRESSURE_PROBE_ONLY" if fit >= 3 and held >= 3 else "INSUFFICIENT_RECORDS"
        )
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
