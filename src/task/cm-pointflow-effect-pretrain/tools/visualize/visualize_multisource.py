#!/usr/bin/env python3
"""Viser viewer for the four-domain PointWorld WM30 corpus.

The viewer is intentionally read-only.  It reads the frozen mixed manifest and
the source sequence arrays directly, so the rendered objects and hands retain
the same provenance as the training windows.  The supported domains are
OakInk2, GRAB, ARCTIC, and ContactPose.

Typical usage from the repository root::

    conda run -n graspenv python \
        src/task/cm-pointflow-effect-pretrain/tools/visualize/visualize_multisource.py

Use ``--check-only`` for a small, headless contract check.  The interactive
viewer exposes source, split, sequence, anchor frame, future offset, playback,
and visibility controls through Viser.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATA_RELATIVE = Path("outputs/cm-pointflow-effect-pretrain/mixed-wm30-20261007")
DOMAIN_NAMES = ("oakink2", "grab", "arctic", "contactpose")
EXPECTED_MIXED_SCHEMA = "pointworld-multisource.wm30.v1"
EXPECTED_SOURCE_SCHEMA = "ref2dex.native-wm30.v1"
EXPECTED_OAKINK_SCHEMA = "oakink.wm30.geometry-centers.v2"
HISTORY = 4
HORIZON = 24

SOURCE_COLORS = {
    "oakink2": np.asarray([235, 145, 45], dtype=np.uint8),
    "grab": np.asarray([90, 170, 245], dtype=np.uint8),
    "arctic": np.asarray([170, 105, 235], dtype=np.uint8),
    "contactpose": np.asarray([50, 195, 145], dtype=np.uint8),
}
CONTEXT_COLOR = np.asarray([145, 150, 160], dtype=np.uint8)
RIGHT_HAND_COLOR = np.asarray([70, 145, 245], dtype=np.uint8)
LEFT_HAND_COLOR = np.asarray([80, 215, 125], dtype=np.uint8)
FUTURE_OBJECT_COLOR = np.asarray([245, 205, 75], dtype=np.uint8)
FUTURE_RIGHT_COLOR = np.asarray([185, 100, 245], dtype=np.uint8)
FUTURE_LEFT_COLOR = np.asarray([245, 105, 155], dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_data_root() -> Path:
    """Find the shared output root when this code lives in a worktree.

    Git worktrees contain the source tree but generally do not contain the
    ignored ``outputs/`` directory.  Prefer an explicit environment override,
    then the current worktree, then sibling Ref2Dex-agent worktrees.
    """

    override = os.environ.get("REF2DEX_POINTWORLD_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidates = [REPO_ROOT / DEFAULT_DATA_RELATIVE]
    for sibling in sorted(REPO_ROOT.parent.glob("Ref2Dex-agent*")):
        candidates.append(sibling / DEFAULT_DATA_RELATIVE)
    for candidate in candidates:
        if (candidate / "processed" / "manifest.json").is_file():
            return candidate.resolve()
    # Keep the first path in the error message if acquisition has not been
    # prepared yet; DomainCatalog will report the missing manifest clearly.
    return candidates[0].resolve()


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    return np.ascontiguousarray(points @ pose[:3, :3].T + pose[:3, 3])


def _finite(name: str, value: np.ndarray) -> None:
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _sample_points(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return np.asarray(points, dtype=np.float32)
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float32)


@dataclass(frozen=True)
class SequenceRecord:
    source: str
    sequence: str
    split: str
    path: Path
    objects: Tuple[str, ...]
    frames: int
    windows: int

    @property
    def label(self) -> str:
        object_label = ", ".join(self.objects[:2])
        if len(self.objects) > 2:
            object_label += f" +{len(self.objects) - 2}"
        return f"{self.sequence} | {object_label} | {self.frames}f"


class SequenceData:
    """Lazy, read-only access to one source sequence and its canonical clouds."""

    def __init__(self, record: SequenceRecord):
        self.record = record
        self.meta = _json(record.path / "meta.json")
        self.arrays = {
            key: np.load(record.path / f"{key}.npy", mmap_mode="r")
            for key in ("hand", "hand_valid", "poses", "pose_valid", "frame_ids", "centers")
        }
        for key, value in self.arrays.items():
            _finite(f"{record.source}/{record.sequence}/{key}", value)
        self._canonical: Dict[str, np.ndarray] = {}
        self._index_cache: Dict[str, np.ndarray] = {}
        self._sequence_indices: Optional[Dict[str, int]] = None

        expected_frames = int(self.arrays["hand"].shape[0])
        if expected_frames != record.frames:
            raise ValueError(
                f"frame count mismatch for {record.source}/{record.sequence}: "
                f"manifest={record.frames}, hand={expected_frames}"
            )
        if self.arrays["hand"].shape[1:] != (2, 11, 3):
            raise ValueError(f"expected hand shape [frames,2,11,3], got {self.arrays['hand'].shape}")
        if self.arrays["hand_valid"].shape != (expected_frames, 2):
            raise ValueError(f"invalid hand_valid shape: {self.arrays['hand_valid'].shape}")
        if self.arrays["poses"].shape != (expected_frames, len(record.objects), 4, 4):
            raise ValueError(f"invalid pose shape: {self.arrays['poses'].shape}")
        if self.arrays["pose_valid"].shape != (expected_frames, len(record.objects)):
            raise ValueError(f"invalid pose_valid shape: {self.arrays['pose_valid'].shape}")
        if self.arrays["centers"].shape != (len(record.objects), 3):
            raise ValueError(f"invalid centers shape: {self.arrays['centers'].shape}")
        if expected_frames < HISTORY + HORIZON:
            raise ValueError(f"sequence is too short for WM30 windows: {expected_frames} frames")

    @property
    def frame_count(self) -> int:
        return int(self.arrays["hand"].shape[0])

    @property
    def minimum_anchor(self) -> int:
        return HISTORY - 1

    @property
    def maximum_anchor(self) -> int:
        return self.frame_count - HORIZON

    def canonical(self, object_name: str) -> np.ndarray:
        if object_name not in self._canonical:
            path = self.record.path.parent.parent / "canonical" / f"{object_name}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"canonical cloud is missing: {path}")
            with np.load(path) as payload:
                if "points" not in payload:
                    raise ValueError(f"canonical file has no points array: {path}")
                points = np.asarray(payload["points"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                raise ValueError(f"invalid canonical point shape at {path}: {points.shape}")
            if len(points) != 512:
                raise ValueError(f"expected 512 canonical points at {path}, got {len(points)}")
            _finite(str(path), points)
            self._canonical[object_name] = points
        return self._canonical[object_name]

    def object_points(self, frame: int, only_valid: bool = True) -> Tuple[List[np.ndarray], List[int]]:
        poses = self.arrays["poses"][frame]
        valid = self.arrays["pose_valid"][frame]
        clouds: List[np.ndarray] = []
        indices: List[int] = []
        for object_index, object_name in enumerate(self.record.objects):
            if only_valid and not bool(valid[object_index]):
                continue
            clouds.append(_transform(self.canonical(object_name), poses[object_index]))
            indices.append(object_index)
        return clouds, indices

    def hand_points(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(self.arrays["hand"][frame], dtype=np.float32)
        valid = np.asarray(self.arrays["hand_valid"][frame], dtype=bool)
        return points, valid

    def window_row(self, split: str, preferred_anchor: int) -> Optional[Tuple[int, int, int, int]]:
        index_path = self.record.path.parent.parent / f"index_{split}.npy"
        if not index_path.is_file():
            return None
        rows = self._index_cache.get(split)
        if rows is None:
            rows = np.load(index_path, mmap_mode="r")
            if rows.ndim != 2 or rows.shape[1] != 4:
                raise ValueError(f"invalid source index shape at {index_path}: {rows.shape}")
            self._index_cache[split] = rows
        sequence_ids = self._sequence_index_map()
        sequence_index = sequence_ids.get(self.record.sequence)
        if sequence_index is None:
            return None
        matches = rows[(rows[:, 0] == sequence_index) & (rows[:, 1] == preferred_anchor)]
        if len(matches) == 0:
            matches = rows[(rows[:, 0] == sequence_index)]
        if len(matches) == 0:
            return None
        return tuple(int(x) for x in matches[0])

    def _sequence_index_map(self) -> Dict[str, int]:
        if self._sequence_indices is None:
            manifest = _json(self.record.path.parent.parent / "manifest.json")
            self._sequence_indices = {
                str(name): index for index, name in enumerate(manifest.get("sequences", []))
            }
        return self._sequence_indices


class DomainCatalog:
    """Catalog of the four registered source domains."""

    def __init__(self, mixed_root: Path):
        self.mixed_root = Path(mixed_root).expanduser().resolve()
        manifest_path = self.mixed_root / "processed" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"mixed manifest does not exist: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = _json(manifest_path)
        if self.manifest.get("schema") != EXPECTED_MIXED_SCHEMA:
            raise ValueError(
                f"expected {EXPECTED_MIXED_SCHEMA}, got {self.manifest.get('schema')!r}"
            )
        if self.manifest.get("status") != "COMPLETED":
            raise ValueError("mixed corpus is not marked COMPLETED")
        if (self.manifest.get("fps"), self.manifest.get("history"), self.manifest.get("horizon")) != (30, 4, 24):
            raise ValueError("mixed corpus does not satisfy the 30Hz/history4/horizon24 contract")
        descriptors = {str(item["name"]): item for item in self.manifest.get("sources", [])}
        if tuple(descriptors) != DOMAIN_NAMES:
            raise ValueError(f"expected source order {DOMAIN_NAMES}, got {tuple(descriptors)}")
        self.domains = {
            name: self._build_domain(name, descriptors[name]) for name in DOMAIN_NAMES
        }

    def _build_domain(self, name: str, descriptor: Mapping[str, Any]) -> List[SequenceRecord]:
        root = _resolve(str(descriptor["root"]), self.mixed_root)
        source_manifest_path = root / "processed" / "manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"{name} source manifest does not exist: {source_manifest_path}")
        expected_hash = descriptor.get("manifest_sha256")
        if expected_hash and _sha256(source_manifest_path) != expected_hash:
            raise ValueError(f"source manifest drift for {name}: {source_manifest_path}")
        source_manifest = _json(source_manifest_path)
        schema = source_manifest.get("schema")
        if name == "oakink2":
            if schema != EXPECTED_OAKINK_SCHEMA:
                raise ValueError(f"unexpected OakInk2 schema: {schema!r}")
            sequences = [str(value) for value in source_manifest.get("sequences", [])]
            split_sequences = source_manifest.get("split_sequences", {})
            record_meta = {}
        else:
            if schema != EXPECTED_SOURCE_SCHEMA:
                raise ValueError(f"unexpected {name} schema: {schema!r}")
            record_meta = {str(item["sequence"]): item for item in source_manifest.get("records", [])}
            # GRAB and ARCTIC share one native pack; each mixed descriptor
            # owns only the records whose provenance source matches its name.
            sequences = [
                sequence for sequence, item in record_meta.items()
                if str(item.get("source")) == name
            ]
            split_sequences = source_manifest.get("splits", {})
        split_by_sequence = {
            str(sequence): str(split)
            for split, values in split_sequences.items()
            for sequence in values
        }
        records: List[SequenceRecord] = []
        for sequence in sequences:
            split = split_by_sequence.get(sequence)
            if split is None:
                continue
            path = root / "processed" / "sequences" / sequence
            meta = record_meta.get(sequence)
            if meta is None:
                meta = _json(path / "meta.json")
            objects = tuple(str(value) for value in meta.get("objects", []))
            if not objects:
                raise ValueError(f"sequence has no objects: {path}")
            frames = int(meta.get("frames", 0))
            index = descriptor.get("indices", {}).get(split, {})
            records.append(
                SequenceRecord(
                    source=name,
                    sequence=sequence,
                    split=split,
                    path=path,
                    objects=objects,
                    frames=frames,
                    windows=int(index.get("windows", 0)),
                )
            )
        if not records:
            raise ValueError(f"no sequences found for source {name}")
        return records

    def records(self, source: str, split: str) -> List[SequenceRecord]:
        if source not in self.domains:
            raise KeyError(source)
        return [record for record in self.domains[source] if split == "all" or record.split == split]

    def first_record(self, source: str, split: str) -> SequenceRecord:
        records = self.records(source, split)
        if not records:
            raise ValueError(f"no {split} sequences available for {source}")
        return records[0]


def _choose_record(records: Sequence[SequenceRecord], sequence: Optional[str], index: int) -> SequenceRecord:
    if sequence:
        matches = [record for record in records if record.sequence == sequence]
        if not matches:
            raise ValueError(f"sequence {sequence!r} is not available in the selected source/split")
        return matches[0]
    return records[int(index) % len(records)]


def _concat_clouds(clouds: Iterable[np.ndarray]) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float32) for value in clouds if len(value)]
    if not values:
        return np.zeros((1, 3), dtype=np.float32)
    return np.concatenate(values, axis=0)


def _render_sample(
    data: SequenceData,
    frame: int,
    target_object: int,
    point_limit: int,
) -> Dict[str, Any]:
    object_clouds, object_indices = data.object_points(frame)
    object_points = _concat_clouds(object_clouds)
    object_colors = np.concatenate(
        [
            np.broadcast_to(
                SOURCE_COLORS[data.record.source]
                if object_index == target_object
                else CONTEXT_COLOR,
                (len(cloud), 3),
            )
            for cloud, object_index in zip(object_clouds, object_indices)
        ],
        axis=0,
    ) if object_clouds else np.zeros((1, 3), dtype=np.uint8)
    if len(object_points) > point_limit:
        indices = np.linspace(0, len(object_points) - 1, point_limit, dtype=np.int64)
        object_points = object_points[indices]
        object_colors = object_colors[indices]
    hands, hand_valid = data.hand_points(frame)
    return {
        "object_points": object_points,
        "object_colors": object_colors,
        "hands": hands,
        "hand_valid": hand_valid,
        "object_indices": object_indices,
    }


def _validate_record(record: SequenceRecord, preferred_anchor: int) -> Dict[str, Any]:
    data = SequenceData(record)
    anchor = min(max(int(preferred_anchor), data.minimum_anchor), data.maximum_anchor)
    row = data.window_row(record.split, anchor)
    target_object = int(row[1]) if row is not None else 0
    if target_object < 0 or target_object >= len(record.objects):
        raise ValueError(f"invalid anchor object {target_object} for {record.sequence}")
    current = _render_sample(data, anchor, target_object, point_limit=8192)
    future = _render_sample(data, anchor + HORIZON, target_object, point_limit=8192)
    return {
        "source": record.source,
        "sequence": record.sequence,
        "split": record.split,
        "frames": data.frame_count,
        "anchor": anchor,
        "target_object": target_object,
        "objects": list(record.objects),
        "current_object_points": int(len(current["object_points"])),
        "current_hand_valid": current["hand_valid"].astype(int).tolist(),
        "future_object_points": int(len(future["object_points"])),
        "future_hand_valid": future["hand_valid"].astype(int).tolist(),
        "window_row": list(row) if row is not None else None,
    }


def run_check(catalog: DomainCatalog, split: str, sequence_index: int, frame: int) -> None:
    results = []
    for source in DOMAIN_NAMES:
        records = catalog.records(source, split)
        if not records:
            results.append({"source": source, "split": split, "status": "NO_SEQUENCES"})
            continue
        record = _choose_record(records, None, sequence_index)
        results.append(_validate_record(record, frame))
    print(json.dumps({"mixed_manifest": str(catalog.manifest_path), "domains": results}, indent=2, ensure_ascii=False))


def _gui_options(records: Sequence[SequenceRecord]) -> Tuple[str, ...]:
    return tuple(record.label for record in records)


def run_server(args: argparse.Namespace, catalog: DomainCatalog) -> None:
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is required for the interactive viewer; use the graspenv environment") from exc

    source = args.source
    split = args.split
    records = catalog.records(source, split)
    if not records:
        raise ValueError(f"no sequences available for source={source!r}, split={split!r}")
    record = _choose_record(records, args.sequence, args.sequence_index)
    data = SequenceData(record)
    initial_frame = min(max(int(args.frame), data.minimum_anchor), data.maximum_anchor)
    row = data.window_row(record.split, initial_frame)
    target_object = int(row[1]) if row is not None else 0
    state: Dict[str, Any] = {"source": source, "split": split, "records": records, "record": record,
                             "data": data, "target_object": target_object, "playing": False}
    lock = threading.RLock()

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    source_gui = server.gui.add_dropdown("数据域", options=DOMAIN_NAMES, initial_value=source)
    split_gui = server.gui.add_dropdown("split", options=("all", "train", "val", "test"), initial_value=split)
    sequence_gui = server.gui.add_dropdown("序列", options=_gui_options(records), initial_value=record.label)
    previous_sequence = server.gui.add_button("上一条序列")
    next_sequence = server.gui.add_button("下一条序列")
    frame_gui = server.gui.add_slider("anchor 帧", min=data.minimum_anchor, max=data.maximum_anchor,
                                      step=1, initial_value=initial_frame)
    future_gui = server.gui.add_slider("未来偏移", min=0, max=HORIZON, step=1, initial_value=int(args.future_delta))
    play_gui = server.gui.add_button("播放 / 暂停")
    fps_gui = server.gui.add_slider("播放 FPS", min=1.0, max=30.0, step=1.0, initial_value=float(args.fps))
    show_object_gui = server.gui.add_checkbox("显示物体点云", initial_value=True)
    show_hand_gui = server.gui.add_checkbox("显示手点", initial_value=True)
    show_future_gui = server.gui.add_checkbox("显示未来点", initial_value=True)
    show_path_gui = server.gui.add_checkbox("显示目标物体轨迹", initial_value=True)
    point_size_gui = server.gui.add_slider("物体点大小", min=0.0005, max=0.02, step=0.0005, initial_value=float(args.point_size))
    hand_size_gui = server.gui.add_slider("手点大小", min=0.0005, max=0.02, step=0.0005, initial_value=float(args.hand_size))
    status_gui = server.gui.add_markdown("")
    help_gui = server.gui.add_markdown(
        "颜色：当前目标物体为域颜色，其他物体为灰色；当前右手蓝色、左手绿色；未来物体黄色、未来双手为紫/粉色。"
        "anchor 帧固定保留 4 帧历史和 24 帧未来，未来偏移为 0 时隐藏未来点。"
    )
    del help_gui

    handles: Dict[str, Any] = {}

    def set_record(new_record: SequenceRecord, reset_frame: bool = True) -> None:
        state["record"] = new_record
        state["data"] = SequenceData(new_record)
        state["target_object"] = 0
        state["records"] = catalog.records(str(source_gui.value), str(split_gui.value))
        frame_gui.min = state["data"].minimum_anchor
        frame_gui.max = state["data"].maximum_anchor
        if reset_frame:
            frame_gui.value = state["data"].minimum_anchor

    def refresh_record_options() -> None:
        selected = catalog.records(str(source_gui.value), str(split_gui.value))
        if not selected:
            raise ValueError(f"no sequences available for {source_gui.value}/{split_gui.value}")
        state["records"] = selected
        sequence_gui.options = _gui_options(selected)
        sequence_gui.value = selected[0].label
        set_record(selected[0])

    def sequence_from_gui() -> SequenceRecord:
        for candidate in state["records"]:
            if candidate.label == str(sequence_gui.value):
                return candidate
        return state["records"][0]

    def render() -> None:
        with lock:
            data: SequenceData = state["data"]
            record: SequenceRecord = state["record"]
            frame = min(max(int(frame_gui.value), data.minimum_anchor), data.maximum_anchor)
            future_delta = int(future_gui.value)
            future_frame = min(frame + future_delta, data.frame_count - 1)
            row = data.window_row(record.split, frame)
            target_object = int(row[1]) if row is not None else 0
            state["target_object"] = target_object
            current = _render_sample(data, frame, target_object, args.max_points)
            future = _render_sample(data, future_frame, target_object, args.max_points)
            hands = current["hands"]
            hand_valid = current["hand_valid"]
            future_hands = future["hands"]
            future_valid = future["hand_valid"]

            def hand_side(points: np.ndarray, valid: np.ndarray, side: int) -> np.ndarray:
                return np.asarray(points[side], dtype=np.float32) if bool(valid[side]) else np.zeros((0, 3), dtype=np.float32)

            right = hand_side(hands, hand_valid, 0)
            left = hand_side(hands, hand_valid, 1)
            future_right = hand_side(future_hands, future_valid, 0)
            future_left = hand_side(future_hands, future_valid, 1)
            for key, points, colors in (
                ("object", current["object_points"], current["object_colors"]),
                ("right", right, np.broadcast_to(RIGHT_HAND_COLOR, (len(right), 3))),
                ("left", left, np.broadcast_to(LEFT_HAND_COLOR, (len(left), 3))),
                ("future_object", future["object_points"], np.broadcast_to(FUTURE_OBJECT_COLOR, (len(future["object_points"]), 3))),
                ("future_right", future_right, np.broadcast_to(FUTURE_RIGHT_COLOR, (len(future_right), 3))),
                ("future_left", future_left, np.broadcast_to(FUTURE_LEFT_COLOR, (len(future_left), 3))),
            ):
                if key not in handles:
                    handles[key] = server.scene.add_point_cloud(
                        f"/dataset/{key}", points=points if len(points) else np.zeros((1, 3), np.float32),
                        colors=colors if len(points) else np.zeros((1, 3), np.uint8),
                        point_size=float(point_size_gui.value if key in ("object", "future_object") else hand_size_gui.value),
                        point_shape="circle", precision="float32", point_shading="flat",
                    )
                else:
                    handles[key].points = points if len(points) else np.zeros((1, 3), np.float32)
                    handles[key].colors = colors if len(points) else np.zeros((1, 3), np.uint8)
                    handles[key].point_size = float(point_size_gui.value if key in ("object", "future_object") else hand_size_gui.value)
                has_points = len(points) > 0
                handles[key].visible = has_points and (
                    (key == "object" and bool(show_object_gui.value))
                    or (key in ("right", "left") and bool(show_hand_gui.value))
                    or (key.startswith("future_") and bool(show_future_gui.value)
                        and ((key == "future_object" and bool(show_object_gui.value)) or (key != "future_object" and bool(show_hand_gui.value)))
                        and future_delta > 0)
                )

            path = data.arrays["poses"][max(data.minimum_anchor, frame - HISTORY + 1):min(data.frame_count, frame + HORIZON + 1), target_object, :3, 3]
            if "path" not in handles:
                handles["path"] = server.scene.add_spline_catmull_rom("/dataset/target_path", path, color=(255, 180, 50), line_width=2.0)
            else:
                handles["path"].positions = path
            handles["path"].visible = bool(show_path_gui.value)

            source_frame = int(data.arrays["frame_ids"][frame])
            future_source_frame = int(data.arrays["frame_ids"][future_frame])
            status_gui.content = (
                f"**域** `{record.source}`  | **split** `{record.split}`  | **序列** `{record.sequence}`  \n"
                f"**anchor** `{frame + 1}/{data.frame_count}`  | **source frame** `{source_frame}`  | "
                f"**未来** `+{future_frame - frame}` ({(future_frame - frame) / 30.0:.3f}s, source `{future_source_frame}`)  \n"
                f"**对象** `{', '.join(record.objects)}`  | **目标 object index** `{target_object}`  | "
                f"**手有效** `right={int(hand_valid[0])}, left={int(hand_valid[1])}`  | "
                f"**对象点** `{len(current['object_points'])}`  | **future object 点** `{len(future['object_points'])}`  \n"
                f"**窗口** `[{max(0, frame - HISTORY + 1)}, {min(data.frame_count - 1, frame + HORIZON)}]`  | "
                f"**dataset index row** `{list(row) if row is not None else 'not found'}`"
            )

    @source_gui.on_update
    def _(_):
        with lock:
            refresh_record_options(); render()

    @split_gui.on_update
    def _(_):
        with lock:
            refresh_record_options(); render()

    @sequence_gui.on_update
    def _(_):
        with lock:
            set_record(sequence_from_gui()); render()

    @previous_sequence.on_click
    def _(_):
        with lock:
            records = state["records"]; index = records.index(state["record"]); set_record(records[(index - 1) % len(records)]); sequence_gui.value = state["record"].label; render()

    @next_sequence.on_click
    def _(_):
        with lock:
            records = state["records"]; index = records.index(state["record"]); set_record(records[(index + 1) % len(records)]); sequence_gui.value = state["record"].label; render()

    for control in (frame_gui, future_gui, show_object_gui, show_hand_gui, show_future_gui, show_path_gui, point_size_gui, hand_size_gui):
        @control.on_update
        def _(_control_event):
            render()

    @play_gui.on_click
    def _(_):
        state["playing"] = not state["playing"]

    render()
    print(f"PointWorld four-domain viewer: http://localhost:{args.port}", flush=True)
    print(f"source={record.source} split={record.split} sequence={record.sequence}", flush=True)
    while True:
        if state["playing"]:
            with lock:
                next_frame = int(frame_gui.value) + 1
                frame_gui.value = data.minimum_anchor if next_frame > data.maximum_anchor else next_frame
        time.sleep(max(0.01, 1.0 / float(fps_gui.value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="mixed-wm30 output root (default: REF2DEX_POINTWORLD_DATA_ROOT or an existing sibling worktree output)",
    )
    parser.add_argument("--source", choices=DOMAIN_NAMES, default="oakink2")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="train")
    parser.add_argument("--sequence", default=None, help="exact sequence id")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=3, help="initial WM30 anchor frame")
    parser.add_argument("--future-delta", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--hand-size", type=float, default=0.006)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--check-only", action="store_true", help="validate one representative sequence per domain and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else _default_data_root()
    print(f"[visualize] loading mixed manifest: {data_root}", flush=True)
    catalog = DomainCatalog(data_root)
    print(f"[visualize] indexed four domains: {', '.join(DOMAIN_NAMES)}", flush=True)
    if args.check_only:
        run_check(catalog, args.split, args.sequence_index, args.frame)
        return
    run_server(args, catalog)


if __name__ == "__main__":
    main()
