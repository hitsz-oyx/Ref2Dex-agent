"""Separate observed geometry from offline labels before packing Shared-WAM."""
from dataclasses import dataclass, asdict
import time
import numpy as np

from . import SOURCE_ID
from .geometry import calibrate_scale, current_anchor, lift_tracks, transform_points
from .native_hand import NativeEpisode
from .sample_contract import validate_sample
from .scene_builder import select_queries, build_scene


@dataclass(frozen=True)
class ProcessingConfig:
    scene_points: int = 2048
    queries: int = 64
    erosion: int = 2
    min_current_points: int = 16
    min_endpoint_fraction: float = .8
    min_scale_samples: int = 12
    max_scale_relative_iqr: float = .25
    max_current_depth_median_m: float = .03
    max_current_depth_p95_m: float = .06
    max_rigid_median_m: float = .005
    max_rigid_p95_m: float = .015

    def __post_init__(self):
        if not 32 <= self.scene_points <= 4096 or not 32 <= self.queries <= 128:
            raise ValueError("bounded scene/query counts required")
        if not 3 <= self.min_current_points <= 32 or not 0 < self.min_endpoint_fraction <= 1:
            raise ValueError("invalid current/endpoint admission thresholds")
        if self.erosion < 0 or self.min_scale_samples < 3:
            raise ValueError("invalid erosion/scale sample count")
        for key, value in asdict(self).items():
            if key.startswith("max_") and (not np.isfinite(value) or value <= 0):
                raise ValueError("positive finite quality threshold required: " + key)


class RejectedWindow(ValueError):
    pass


def rigid_residual(current, future):
    a, b = current-current.mean(0), future-future.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ fix @ vt
    return np.linalg.norm(a @ rotation-b, axis=-1)


def process_window(native: NativeEpisode, key, episode_id, tick, rgb, intrinsic,
                   masks, depth_backend, tracker, config=None, depth_to_native=None):
    config = config or ProcessingConfig()
    if rgb.shape[0] != 28 or not 3 <= tick < native.frames-24:
        raise ValueError("native 28-frame H4/K24 clip required")
    timings, began = {}, time.monotonic()
    now = time.monotonic()
    # Pass only <=t RGB and native arrays to target selection. No future array view.
    observed = NativeEpisode(native.hand[tick-3:tick+1].copy(), native.valid[tick-3:tick+1].copy(),
                             native.camera[tick-3:tick+1].copy(), intrinsic.copy(),
                             native.session, native.language)
    target = masks.select_current(key, tick, rgb[3].copy(), observed)
    if not target.get("current_verified") or not target.get("name"):
        raise RejectedWindow("TARGET_UNVERIFIED")
    target_mask, hand_mask = np.asarray(target["target"], bool), np.asarray(target["hand"], bool)
    if target_mask.shape != rgb.shape[1:3] or hand_mask.shape != target_mask.shape:
        raise ValueError("current mask/video geometry mismatch")
    timings["target_s"] = time.monotonic()-now
    now = time.monotonic()
    obs_ids = np.arange(tick-3, tick+1)
    observed_depth = np.asarray(depth_backend.predict(rgb[:4].copy(), obs_ids, "observation"), np.float32)
    if observed_depth.shape != (4, *target_mask.shape):
        raise ValueError("observation depth geometry/clock mismatch")
    observation_scale = calibrate_scale(observed_depth, observed.hand, observed.valid, intrinsic,
        observed.camera, depth_to_native, config.min_scale_samples, config.max_scale_relative_iqr)
    depth_now = observed_depth[-1]*observation_scale["scale"]
    uv, query_world = select_queries(depth_now, target_mask, hand_mask, intrinsic,
        native.camera[tick], config.queries, config.erosion, depth_to_native)
    if len(uv) < config.min_current_points:
        raise RejectedWindow("INSUFFICIENT_CURRENT_TARGET_POINTS")
    # Fixed identity/order chosen now; future quality NEVER selects survivors.
    query_count = min(len(uv), 32)
    anchor = current_anchor(query_world[:query_count])
    scene = build_scene(depth_now, target_mask, hand_mask, intrinsic, native.camera[tick], anchor,
        uv[:query_count], query_world[:query_count], native.hand[tick], native.valid[tick],
        config.scene_points, depth_to_native=depth_to_native)
    hand = native.window(tick, anchor)
    if max(hand["start_hand_valid"].sum(-1)) < 6 or not hand["trajectory_valid"].any():
        raise RejectedWindow("INSUFFICIENT_NATIVE_HAND_SUPERVISION")
    timings["observation_s"] = time.monotonic()-now
    now = time.monotonic()
    tracks, visibility = tracker.track(rgb[3:].copy(), uv.copy())
    tracks, visibility = np.asarray(tracks, np.float32), np.asarray(visibility, bool)
    if tracks.shape != (25, len(uv), 2) or visibility.shape != (25, len(uv)):
        raise ValueError("25 native frames with unchanged query IDs required")
    if not np.allclose(tracks[0], uv, atol=.01):
        raise ValueError("tracker changed current query positions")
    timings["tracking_s"] = time.monotonic()-now
    now = time.monotonic()
    label_ids = np.arange(tick, tick+25)
    label_depth = np.asarray(depth_backend.predict(rgb[3:].copy(), label_ids, "label"), np.float32)
    if label_depth.shape != (25, *target_mask.shape):
        raise ValueError("label depth geometry/clock mismatch")
    label_scale = calibrate_scale(label_depth, native.hand[tick:tick+25], native.valid[tick:tick+25],
        intrinsic, native.camera[tick:tick+25], depth_to_native,
        config.min_scale_samples, config.max_scale_relative_iqr)
    world, valid = lift_tracks(tracks, visibility, label_depth*label_scale["scale"], intrinsic,
                               native.camera[tick:tick+25], depth_to_native)
    identity_masks = masks.label_masks(key, label_ids, target_mask.shape)
    if identity_masks is None:
        raise RejectedWindow("MISSING_FUTURE_TARGET_IDENTITY_MASKS")
    if np.shape(identity_masks) != (25, *target_mask.shape):
        raise ValueError("future target identity mask clock mismatch")
    safe = np.nan_to_num(tracks, nan=-1., posinf=-1., neginf=-1.)
    x, y = np.rint(safe[..., 0]).astype(int), np.rint(safe[..., 1]).astype(int)
    height, width = target_mask.shape
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    identity = np.asarray(identity_masks, bool)[np.arange(25)[:, None],
                                               np.clip(y, 0, height-1), np.clip(x, 0, width-1)]
    valid &= inside & identity
    valid, world = valid[:, :query_count], world[:, :query_count]
    current_common = valid[0]
    if current_common.sum() < config.min_current_points:
        raise RejectedWindow("LABEL_CURRENT_DEPTH_UNAVAILABLE")
    current_error = np.linalg.norm(world[0, current_common]-query_world[:query_count][current_common], axis=-1)
    median, p95 = np.quantile(current_error, [.5, .95])
    if median > config.max_current_depth_median_m or p95 > config.max_current_depth_p95_m:
        raise RejectedWindow("OBSERVATION_LABEL_METRIC_DEPTH_DISAGREEMENT")
    valid &= current_common[None]
    endpoint_fraction = float(valid[-1].sum()/query_count)
    if endpoint_fraction < config.min_endpoint_fraction:
        raise RejectedWindow("INSUFFICIENT_ENDPOINT_COVERAGE")
    residuals = []
    for k in range(1, 25):
        mask = valid[k]
        if mask.sum() >= 4:
            error = rigid_residual(query_world[:query_count][mask], world[k, mask])
            residuals.append(dict(horizon=k, median_m=float(np.median(error)), p95_m=float(np.quantile(error, .95))))
    if not residuals or any(r["median_m"] > config.max_rigid_median_m or
                             r["p95_m"] > config.max_rigid_p95_m for r in residuals):
        raise RejectedWindow("NONRIGID_OR_NOISY_TARGET_TRACKS")
    current = np.zeros((32, 3), np.float32)
    current[:query_count] = transform_points(query_world[:query_count], anchor)
    future = np.zeros((25, 32, 3), np.float32)
    future[:, :query_count] = transform_points(world, anchor)
    known = np.zeros((25, 32), bool)
    known[:, :query_count] = valid
    # Query frame comes from causal observation, not offline future reconstruction.
    future[0] = current
    scene_target = np.where(known[1:, :, None], future[1:]-current[None], 0.)
    endpoint = np.where(known[-1, :, None], future[-1], 0.)
    sample = dict(**scene, **hand, scene_points=current[None], scene_target=scene_target[None],
        scene_valid=known[None, 1:], object_valid=np.ones(1, bool), start_object_points=current.copy(),
        goal_object_points=endpoint, object_goal_valid=known[-1], goal_object_slot=np.array(0, np.int64),
        source_id=np.array(SOURCE_ID, np.int64), sample_id=np.array([episode_id, tick], np.int64))
    validate_sample(sample)
    timings["label_geometry_s"] = time.monotonic()-now
    track_uv = np.zeros((25, 32, 2), np.float32)
    track_uv[:, :query_count] = np.nan_to_num(tracks[:, :query_count])
    meta = dict(episode_key=key, session=native.session, tick=tick, profile="geometry-only",
        fps=30, history=4, horizon=24, observation_frame_ids=obs_ids.tolist(),
        label_frame_ids=label_ids.tolist(), causal_observation=True, training_allowed=True,
        target_name=target["name"], target_method=target.get("method", "backend"),
        native_hand_supervision=True, scene_supervision="tracked-point-pseudolabel",
        anchor_world_to_sample=anchor.tolist(), depth_to_native=np.asarray(
            np.eye(4) if depth_to_native is None else depth_to_native).tolist(),
        observation_calibration=observation_scale, label_calibration=label_scale,
        current_depth_agreement=dict(median_m=float(median), p95_m=float(p95)),
        current_query_count=query_count, candidate_query_count=len(uv),
        endpoint_fraction=endpoint_fraction, endpoint_slots_fraction=float(known[-1].mean()),
        per_horizon_valid=known[1:].sum(-1).tolist(), rigid_residuals=residuals,
        timings=timings, elapsed_s=time.monotonic()-began,
        thresholds=asdict(config))
    return sample, meta, dict(track_uv=track_uv, track_valid=known)
