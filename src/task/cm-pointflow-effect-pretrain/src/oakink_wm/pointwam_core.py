"""Small, data-contract-first reconstruction of the PointWAM core pipeline.

This module deliberately stops at the human scene/hand forecaster.  It does
not claim to reproduce PointWAM's private Mosaic3D/Volt implementation or its
robot action retargeter.  The useful boundary here is the one that was missing
from the earlier probes:

``metric scene tracks + semantic hand points + language`` ->
``future scene displacement + future hand displacement``.

The preprocessing functions use NumPy so they can be run without a model or a
GPU.  The forecaster is a small O(N) PyTorch model intended for contract and
learning smoke tests; the official model is sparse and substantially larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SEMANTIC_HAND_POINTS = 10
RAW_CLIP_FRAMES = 11
MODEL_CLIP_FRAMES = 31
SCENE_POINT_CAP = 12_000
SCENE_DEDUP_VOXEL_M = 0.015

# VITRA/MediaPipe-style 21-joint order: wrist, four joints per finger in the
# order thumb, index, middle, ring, little.  PointWAM keeps six palm-side
# points (wrist + five finger bases) and four tips (thumb/index/middle/ring).
VITRA_21_TO_10 = (0, 1, 5, 9, 13, 17, 4, 8, 12, 16)

# EgoDex publishes named transforms rather than a canonical integer skeleton.
# Keeping the names here makes the conversion auditable and avoids pretending
# that an undocumented 25-joint index order is universal.
EGODEX_SEMANTIC_SUFFIXES = (
    "Hand",
    "LittleFingerMetacarpal",
    "RingFingerMetacarpal",
    "MiddleFingerMetacarpal",
    "IndexFingerMetacarpal",
    "ThumbKnuckle",
    "ThumbTip",
    "IndexFingerTip",
    "MiddleFingerTip",
    "RingFingerTip",
)


class PointWAMContractError(ValueError):
    """Raised when an input cannot satisfy the PointWAM data contract."""


def _as_float32(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if not np.isfinite(value[np.isfinite(value)]).all():
        # The expression above intentionally permits NaNs used for invalid
        # hand points.  Infinities are never meaningful metric coordinates.
        if np.isinf(value).any():
            raise PointWAMContractError(f"{name} contains infinite values")
    return value


def _validate_xyz(array: np.ndarray, name: str, ndim: int) -> np.ndarray:
    value = _as_float32(array, name)
    if value.ndim != ndim or value.shape[-1] != 3:
        raise PointWAMContractError(
            f"{name} must have shape (..., 3) with ndim={ndim}, got {value.shape}"
        )
    return value


def semantic_keypoints_from_joints(
    joints: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    source: str = "vitra21",
    joint_map: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a hand skeleton to the ten PointWAM semantic keypoints.

    ``joints`` may have any leading dimensions, followed by ``(J, 3)``.  An
    invalid semantic point remains ``NaN`` in the returned coordinates and is
    represented by ``False`` in the mask.  Zero-filling missing hands would
    create a fake trajectory, so callers must keep the mask.

    The VITRA 21-joint order has a documented default map.  EgoDex's named
    transforms should normally be converted with :func:`load_egodex_hands`;
    an integer map for another skeleton must be supplied explicitly.
    """

    points = _validate_xyz(joints, "joints", joints.ndim)
    if source == "vitra21":
        if points.shape[-2] != 21:
            raise PointWAMContractError(
                f"vitra21 expects 21 joints, got {points.shape[-2]}"
            )
        selected = VITRA_21_TO_10 if joint_map is None else tuple(joint_map)
    elif source in {"custom", "egodex25"}:
        if joint_map is None:
            raise PointWAMContractError(
                f"{source} requires an explicit joint_map; do not assume a 25-joint order"
            )
        selected = tuple(joint_map)
    else:
        raise PointWAMContractError(f"unknown hand source: {source}")
    if len(selected) != SEMANTIC_HAND_POINTS or len(set(selected)) != len(selected):
        raise PointWAMContractError("joint_map must contain ten distinct indices")
    if min(selected) < 0 or max(selected) >= points.shape[-2]:
        raise PointWAMContractError("joint_map contains an out-of-range index")

    out = points[..., selected, :].copy()
    if valid is None:
        out_valid = np.isfinite(out).all(axis=-1)
    else:
        source_valid = np.asarray(valid, dtype=bool)
        if source_valid.shape != points.shape[:-1]:
            raise PointWAMContractError(
                f"valid must have shape {points.shape[:-1]}, got {source_valid.shape}"
            )
        out_valid = source_valid[..., selected] & np.isfinite(out).all(axis=-1)
    out[~out_valid] = np.nan
    return out.astype(np.float32), out_valid.astype(bool)


def load_egodex_hands(
    hdf5_path: str | Path,
    *,
    frame_indices: Sequence[int] | None = None,
    sides: Sequence[str] = ("left", "right"),
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load EgoDex's named hand transforms into ``(T, H, 10, 3)`` points.

    This intentionally reads annotations only; RGB decoding and scene tracking
    remain separate contracts.  It works with an official EgoDex HDF5 file and
    does not require the paired MP4 for the hand branch.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("load_egodex_hands requires h5py") from exc

    path = Path(hdf5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    side_names = tuple(sides)
    if not side_names or any(side not in {"left", "right"} for side in side_names):
        raise PointWAMContractError("sides must contain left/right")

    with h5py.File(path, "r") as root:
        available = root.get("/transforms")
        if available is None:
            raise PointWAMContractError(f"{path} has no /transforms group")
        first_name = f"{side_names[0]}{EGODEX_SEMANTIC_SUFFIXES[0]}"
        if first_name not in available:
            raise PointWAMContractError(f"missing EgoDex transform /transforms/{first_name}")
        total = int(available[first_name].shape[0])
        indices = np.arange(total, dtype=np.int64) if frame_indices is None else np.asarray(frame_indices)
        if indices.ndim != 1 or indices.size == 0 or (indices < 0).any() or (indices >= total).any():
            raise PointWAMContractError("frame_indices must be nonempty and within the episode")

        coordinates = np.full(
            (len(indices), len(side_names), SEMANTIC_HAND_POINTS, 3),
            np.nan,
            dtype=np.float32,
        )
        valid = np.zeros(coordinates.shape[:-1], dtype=bool)
        for hand_id, side in enumerate(side_names):
            for point_id, suffix in enumerate(EGODEX_SEMANTIC_SUFFIXES):
                name = f"{side}{suffix}"
                if name not in available:
                    raise PointWAMContractError(f"missing EgoDex transform /transforms/{name}")
                transform = np.asarray(available[name][indices, :3, 3], dtype=np.float32)
                point_valid = np.isfinite(transform).all(axis=-1)
                confidence = root.get(f"/confidences/{name}")
                if confidence is not None:
                    point_valid &= np.asarray(confidence[indices]) > 0
                coordinates[:, hand_id, point_id] = np.where(
                    point_valid[:, None], transform, np.nan
                )
                valid[:, hand_id, point_id] = point_valid

        attrs = root.attrs
        if attrs.get("llm_type", "") == "reversible":
            direction = str(attrs.get("which_llm_description", "1"))
            key = "llm_description" if direction == "1" else "llm_description2"
        else:
            key = "llm_description"
        language = attrs.get(key, "")
        if isinstance(language, bytes):
            language = language.decode("utf-8", errors="replace")
        return coordinates, valid, str(language)


def _interp_valid(values: np.ndarray, valid: np.ndarray, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample values while preserving invalid gaps."""

    if values.shape[0] < 2:
        raise PointWAMContractError("at least two raw frames are required")
    if valid.shape != values.shape[:-1]:
        raise PointWAMContractError("value and validity shapes disagree")
    source_t = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    flat_values = values.reshape(values.shape[0], -1, 3)
    flat_valid = valid.reshape(valid.shape[0], -1)
    out = np.full((target_frames, flat_values.shape[1], 3), np.nan, dtype=np.float32)
    out_valid = np.zeros((target_frames, flat_values.shape[1]), dtype=bool)
    for target_id, t in enumerate(target_t):
        right = int(np.searchsorted(source_t, t, side="left"))
        right = min(right, values.shape[0] - 1)
        left = max(0, right - 1)
        if right == left:
            out_valid[target_id] = flat_valid[left]
            out[target_id, out_valid[target_id]] = flat_values[left, out_valid[target_id]]
            continue
        alpha = float((t - source_t[left]) / (source_t[right] - source_t[left]))
        keep = flat_valid[left] & flat_valid[right]
        out_valid[target_id] = keep
        out[target_id, keep] = (
            (1.0 - alpha) * flat_values[left, keep] + alpha * flat_values[right, keep]
        )
    return out.reshape((target_frames,) + values.shape[1:]), out_valid.reshape((target_frames,) + valid.shape[1:])


def deduplicate_scene_tracks(
    scene_xyz: np.ndarray,
    scene_valid: np.ndarray,
    *,
    voxel_size_m: float = SCENE_DEDUP_VOXEL_M,
    max_points: int = SCENE_POINT_CAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select first-frame scene tracks using PointWAM's metric voxel contract."""

    tracks = _validate_xyz(scene_xyz, "scene_xyz", 3)
    valid = np.asarray(scene_valid, dtype=bool)
    if valid.shape != tracks.shape[:-1]:
        raise PointWAMContractError("scene_valid shape does not match scene_xyz")
    if voxel_size_m <= 0 or max_points <= 0:
        raise PointWAMContractError("voxel_size_m and max_points must be positive")
    first = valid[0] & np.isfinite(tracks[0]).all(axis=-1)
    candidates = np.flatnonzero(first)
    if candidates.size == 0:
        raise PointWAMContractError("scene has no valid first-frame tracks")
    selected: list[int] = []
    occupied: set[tuple[int, int, int]] = set()
    for index in candidates.tolist():
        key = tuple(np.floor(tracks[0, index] / voxel_size_m).astype(np.int64).tolist())
        if key in occupied:
            continue
        occupied.add(key)
        selected.append(index)
        if len(selected) >= max_points:
            break
    chosen = np.asarray(selected, dtype=np.int64)
    return tracks[:, chosen].copy(), valid[:, chosen].copy(), chosen


@dataclass(frozen=True)
class PointWAMClip:
    """A fully materialized 31-frame human training clip."""

    scene_xyz: np.ndarray  # (31, N, 3), metric world coordinates
    scene_valid: np.ndarray  # (31, N)
    hand_xyz: np.ndarray  # (31, H, 10, 3)
    hand_valid: np.ndarray  # (31, H, 10)
    language: str = ""

    def __post_init__(self) -> None:
        if self.scene_xyz.shape[0] != MODEL_CLIP_FRAMES:
            raise PointWAMContractError("scene clip must contain 31 frames")
        if self.hand_xyz.shape[0] != MODEL_CLIP_FRAMES:
            raise PointWAMContractError("hand clip must contain 31 frames")
        if self.scene_valid.shape != self.scene_xyz.shape[:-1]:
            raise PointWAMContractError("scene validity shape mismatch")
        if self.hand_valid.shape != self.hand_xyz.shape[:-1]:
            raise PointWAMContractError("hand validity shape mismatch")
        if self.hand_xyz.shape[-2] != SEMANTIC_HAND_POINTS:
            raise PointWAMContractError("hand clip must contain ten semantic points")

    @property
    def hand_points(self) -> int:
        return int(np.prod(self.hand_xyz.shape[1:3]))

    def targets(self) -> dict[str, np.ndarray]:
        """Return current-frame anchored displacements and conservative masks."""

        current_scene_valid = self.scene_valid[0]
        current_hand_valid = self.hand_valid[0]
        scene_mask = self.scene_valid & current_scene_valid[None, :]
        hand_mask = self.hand_valid & current_hand_valid[None, :, :]
        scene_anchor = np.nan_to_num(self.scene_xyz[0], nan=0.0)
        hand_anchor = np.nan_to_num(self.hand_xyz[0], nan=0.0)
        scene = np.nan_to_num(self.scene_xyz, nan=0.0) - scene_anchor[None]
        hand = np.nan_to_num(self.hand_xyz, nan=0.0) - hand_anchor[None]
        return {
            "scene_current": scene_anchor.astype(np.float32),
            "scene_valid": current_scene_valid.astype(bool),
            "hand_current": hand_anchor.reshape(-1, 3).astype(np.float32),
            "hand_valid": current_hand_valid.reshape(-1).astype(bool),
            "scene_displacement": scene.astype(np.float32),
            "scene_target_valid": scene_mask.astype(bool),
            "hand_displacement": hand.reshape(MODEL_CLIP_FRAMES, -1, 3).astype(np.float32),
            "hand_target_valid": hand_mask.reshape(MODEL_CLIP_FRAMES, -1).astype(bool),
        }


def build_pointwam_clip(
    scene_xyz: np.ndarray,
    scene_valid: np.ndarray,
    hand_joints: np.ndarray,
    hand_joint_valid: np.ndarray | None = None,
    *,
    source: str = "vitra21",
    joint_map: Sequence[int] | None = None,
    language: str = "",
    target_frames: int = MODEL_CLIP_FRAMES,
    scene_voxel_m: float = SCENE_DEDUP_VOXEL_M,
    max_scene_points: int = SCENE_POINT_CAP,
) -> PointWAMClip:
    """Construct the core clip from raw 10 Hz scene tracks and hand joints."""

    scene = _validate_xyz(scene_xyz, "scene_xyz", 3)
    scene_mask = np.asarray(scene_valid, dtype=bool)
    if scene_mask.shape != scene.shape[:-1]:
        raise PointWAMContractError("scene_valid shape does not match scene_xyz")
    hands, hand_mask = semantic_keypoints_from_joints(
        hand_joints, hand_joint_valid, source=source, joint_map=joint_map
    )
    if scene.shape[0] != hands.shape[0]:
        raise PointWAMContractError("scene and hand clips must share frame count")
    scene, scene_mask, _ = deduplicate_scene_tracks(
        scene, scene_mask, voxel_size_m=scene_voxel_m, max_points=max_scene_points
    )
    scene, scene_mask = _interp_valid(scene, scene_mask, target_frames)
    hands, hand_mask = _interp_valid(hands, hand_mask, target_frames)
    return PointWAMClip(scene, scene_mask, hands, hand_mask, language)


def collate_pointwam_clips(clips: Sequence[PointWAMClip]) -> dict[str, torch.Tensor]:
    """Pad variable scene point counts while retaining every validity mask."""

    if not clips:
        raise PointWAMContractError("at least one clip is required")
    max_scene = max(int(clip.scene_xyz.shape[1]) for clip in clips)
    batch = len(clips)
    hands = int(clips[0].hand_points)
    for clip in clips:
        if clip.hand_points != hands:
            raise PointWAMContractError("all clips must have the same hand count")
    scene_xyz = np.zeros((batch, MODEL_CLIP_FRAMES, max_scene, 3), dtype=np.float32)
    scene_valid = np.zeros((batch, max_scene), dtype=bool)
    scene_target = np.zeros_like(scene_xyz)
    scene_target_valid = np.zeros((batch, MODEL_CLIP_FRAMES, max_scene), dtype=bool)
    hand_xyz = np.zeros((batch, hands, 3), dtype=np.float32)
    hand_valid = np.zeros((batch, hands), dtype=bool)
    hand_target = np.zeros((batch, MODEL_CLIP_FRAMES, hands, 3), dtype=np.float32)
    hand_target_valid = np.zeros((batch, MODEL_CLIP_FRAMES, hands), dtype=bool)
    for row, clip in enumerate(clips):
        targets = clip.targets()
        count = clip.scene_xyz.shape[1]
        scene_xyz[row, :, :count] = np.nan_to_num(clip.scene_xyz, nan=0.0)
        scene_valid[row, :count] = targets["scene_valid"]
        scene_target[row, :, :count] = targets["scene_displacement"]
        scene_target_valid[row, :, :count] = targets["scene_target_valid"]
        hand_xyz[row] = targets["hand_current"]
        hand_valid[row] = targets["hand_valid"]
        hand_target[row] = targets["hand_displacement"]
        hand_target_valid[row] = targets["hand_target_valid"]
    return {
        "scene_xyz": torch.from_numpy(scene_xyz[:, 0]),
        "scene_valid": torch.from_numpy(scene_valid),
        "hand_xyz": torch.from_numpy(hand_xyz),
        "hand_valid": torch.from_numpy(hand_valid),
        "scene_target": torch.from_numpy(scene_target),
        "scene_target_valid": torch.from_numpy(scene_target_valid),
        "hand_target": torch.from_numpy(hand_target),
        "hand_target_valid": torch.from_numpy(hand_target_valid),
    }


def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class _MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class PointWAMCoreForecaster(nn.Module):
    """An O(N) scene/hand joint forecaster for the PointWAM contract.

    This is intentionally smaller than the paper model.  It preserves the
    essential interfaces and separate scene/hand heads while using masked set
    pooling instead of Mosaic3D + Volt so a contract smoke fits on one GPU.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 96,
        horizon: int = MODEL_CLIP_FRAMES,
        language_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or horizon < 1 or language_dim < 0:
            raise ValueError("invalid forecaster dimensions")
        self.horizon = int(horizon)
        self.language_dim = int(language_dim)
        self.scene_encoder = _MLP(3, hidden_dim, hidden_dim)
        self.hand_encoder = _MLP(3, hidden_dim, hidden_dim)
        self.language_encoder = (
            _MLP(language_dim, hidden_dim, hidden_dim) if language_dim else None
        )
        context_in = hidden_dim * (2 + bool(language_dim))
        self.context = _MLP(context_in, hidden_dim, hidden_dim)
        self.scene_refine = _MLP(hidden_dim * 2, hidden_dim, hidden_dim)
        self.hand_refine = _MLP(hidden_dim * 2, hidden_dim, hidden_dim)
        self.time = nn.Embedding(self.horizon, hidden_dim)
        self.scene_head = _MLP(hidden_dim * 2, hidden_dim, 3)
        self.hand_head = _MLP(hidden_dim * 2, hidden_dim, 3)

    def forward(
        self,
        scene_xyz: torch.Tensor,
        scene_valid: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
        language: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if scene_xyz.ndim != 3 or scene_xyz.shape[-1] != 3:
            raise ValueError("scene_xyz must be (B,N,3)")
        if hand_xyz.ndim != 3 or hand_xyz.shape[-1] != 3:
            raise ValueError("hand_xyz must be (B,K,3)")
        if scene_valid.shape != scene_xyz.shape[:2] or hand_valid.shape != hand_xyz.shape[:2]:
            raise ValueError("point validity masks do not match point tensors")
        scene_valid = scene_valid.bool()
        hand_valid = hand_valid.bool()
        scene_xyz = torch.nan_to_num(scene_xyz.float()) * scene_valid[..., None]
        hand_xyz = torch.nan_to_num(hand_xyz.float()) * hand_valid[..., None]
        if language is None:
            language = scene_xyz.new_zeros((scene_xyz.shape[0], self.language_dim))
        if language.ndim != 2 or language.shape[0] != scene_xyz.shape[0] or language.shape[1] != self.language_dim:
            raise ValueError(f"language must be (B,{self.language_dim})")

        scene_tokens = self.scene_encoder(scene_xyz)
        hand_tokens = self.hand_encoder(hand_xyz)
        parts = [_masked_mean(scene_tokens, scene_valid), _masked_mean(hand_tokens, hand_valid)]
        if self.language_encoder is not None:
            parts.append(self.language_encoder(language))
        global_context = self.context(torch.cat(parts, dim=-1))
        scene_tokens = self.scene_refine(
            torch.cat((scene_tokens, global_context[:, None].expand_as(scene_tokens)), dim=-1)
        )
        hand_tokens = self.hand_refine(
            torch.cat((hand_tokens, global_context[:, None].expand_as(hand_tokens)), dim=-1)
        )
        time = self.time.weight[None, :, None]
        scene_hidden = scene_tokens[:, None] + time
        hand_hidden = hand_tokens[:, None] + time
        scene_delta = self.scene_head(torch.cat((scene_hidden, time.expand_as(scene_hidden)), dim=-1))
        hand_delta = self.hand_head(torch.cat((hand_hidden, time.expand_as(hand_hidden)), dim=-1))
        return {"scene_displacement": scene_delta, "hand_displacement": hand_delta}


def pointwam_joint_loss(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    scene_weight: float = 1.0,
    hand_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute masked scene and hand displacement losses separately."""

    terms: dict[str, torch.Tensor] = {}
    for name, weight in (("scene", scene_weight), ("hand", hand_weight)):
        pred = prediction[f"{name}_displacement"]
        truth = target[f"{name}_target"]
        valid = target[f"{name}_target_valid"].bool()
        if pred.shape != truth.shape:
            raise ValueError(f"{name} prediction/target shape mismatch")
        error = F.smooth_l1_loss(pred, truth, reduction="none").mean(-1)
        error = error.masked_select(valid)
        if not error.numel():
            raise PointWAMContractError(f"{name} has no valid target points")
        terms[name] = error.mean()
    terms["total"] = scene_weight * terms["scene"] + hand_weight * terms["hand"]
    return terms["total"], terms


def synthetic_pointwam_clips(
    count: int = 8,
    *,
    scene_points: int = 24,
    raw_frames: int = RAW_CLIP_FRAMES,
    seed: int = 11,
) -> list[PointWAMClip]:
    """Create deterministic metric clips for a contract/optimizer smoke only."""

    if count < 1 or scene_points < 4 or raw_frames < 2:
        raise ValueError("invalid synthetic clip dimensions")
    rng = np.random.default_rng(seed)
    clips: list[PointWAMClip] = []
    t = np.linspace(0.0, 1.0, raw_frames, dtype=np.float32)
    for _ in range(count):
        base_scene = rng.normal(0.0, 0.12, (scene_points, 3)).astype(np.float32)
        scene_velocity = (0.01 + 0.02 * base_scene).astype(np.float32)
        scene = base_scene[None] + t[:, None, None] * scene_velocity[None]
        scene_valid = np.ones((raw_frames, scene_points), dtype=bool)
        base_hand = rng.normal(0.0, 0.08, (2, 21, 3)).astype(np.float32)
        hand_velocity = (0.015 + 0.03 * base_hand).astype(np.float32)
        hands = base_hand[None] + t[:, None, None, None] * hand_velocity[None]
        hand_valid = np.ones((raw_frames, 2, 21), dtype=bool)
        clips.append(
            build_pointwam_clip(
                scene,
                scene_valid,
                hands,
                hand_valid,
                source="vitra21",
                language="synthetic contract smoke",
            )
        )
    return clips

