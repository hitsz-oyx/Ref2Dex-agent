"""Scheduler-conditioned cache identities for the v6 tau gate.

The condition is deliberately produced through the same
``OrderedGoalScheduler``/``update_schedulers`` boundary used by the online
WAM path.  This module contains identity and replay plumbing only; it does
not change the existing v5/v6 reader contract.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
WAM_ROOT = ROOT / "src/task/wam-rl-policy-v3"
V5_ROOT = ROOT / "src/task/ptv3-shared-wam-v5"
sys.path.insert(0, str(WAM_ROOT / "src"))
sys.path.insert(0, str(V5_ROOT / "src"))

from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler, update_schedulers
from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21


GOAL_SEMANTICS = "spatial_path_waypoint"
HORIZON_STEPS = 24
SCHEDULER_HZ = 30
IDENTITY_SCHEMA = "ref2dex.v6.scheduler-cache-identity.v1"


def _canonical_sample_id(sample_id):
    value = np.asarray(sample_id, dtype=np.int64)
    if value.ndim == 0:
        return int(value)
    return [int(x) for x in value.reshape(-1)]


def condition_tensor_hash(condition):
    """Hash the exact float32 tensor bytes, shape and dtype."""
    value = np.ascontiguousarray(np.asarray(condition, dtype=np.float32))
    if value.shape != (32, 3) or not np.isfinite(value).all():
        raise ValueError("scheduler condition must be finite float32 32x3")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def cache_identity(source_id, sample_id, window_start_frame, condition,
                   *, goal_semantics=GOAL_SEMANTICS):
    """Return a stable identity whose key includes source and local sample id."""
    if goal_semantics != GOAL_SEMANTICS:
        raise ValueError("only spatial_path_waypoint is accepted")
    source_id = int(source_id)
    sample_id = _canonical_sample_id(sample_id)
    window_start_frame = int(window_start_frame)
    payload = dict(source_id=source_id, sample_id=sample_id,
                   window_start_frame=window_start_frame,
                   goal_semantics=goal_semantics,
                   horizon_steps=HORIZON_STEPS, scheduler_hz=SCHEDULER_HZ,
                   condition_sha256=condition_tensor_hash(condition))
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                          separators=(",", ":")).encode("utf-8")).hexdigest()
    return dict(schema=IDENTITY_SCHEMA, identity=identity, **payload)


def _canonical_hand(joints, source_id):
    """Map the v5 21-joint hand to the scheduler's canonical 11 points."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 4 or joints.shape[2:] != (21, 3):
        raise ValueError("scheduler replay requires T x hands x 21 x 3 joints")
    # Source 4 has the historical thumb ordering used by the v5 preparation;
    # all other sources use the frozen SPARSE11_IN_21 mapping.
    ids = (0, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20) if int(source_id) == 4 else SPARSE11_IN_21
    return joints[:, 0, ids].astype(np.float64, copy=False)


def scheduler_conditions_for_ticks(sequence, geometry, source_id, slot, ticks):
    """Replay the inference scheduler once and return deltas for selected ticks.

    ``geometry`` is the same v5 geometry entry used by ``sample_window`` and
    contains ``query`` (the 32 same-ID points) and ``surface`` (interaction
    points).  A missing/non-finite observation makes the corresponding window
    ineligible instead of fabricating a target.
    """
    source_id, slot = int(source_id), int(slot)
    ticks = sorted({int(t) for t in ticks})
    if not ticks or min(ticks) < 3:
        raise ValueError("scheduler replay requires admitted H4 ticks")
    # A planned reference path must be supplied under an explicit field.  The
    # observed object trajectory is a label and is intentionally rejected.
    reference = sequence.get("scheduler_reference_poses")
    if reference is None:
        raise ValueError("scheduler reference path absent; observed object_poses are labels")
    poses = np.asarray(reference[:, slot], dtype=np.float64)
    joints = np.asarray(sequence["joints"], dtype=np.float64)
    query = np.asarray(geometry["query"], dtype=np.float64)
    surface = np.asarray(geometry["surface"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("finite object pose reference required")
    if not np.isfinite(poses).all() or not np.isfinite(joints).all():
        raise ValueError("scheduler replay cannot bridge non-finite observations")
    if query.shape != (32, 3) or not np.isfinite(query).all():
        raise ValueError("finite 32-point scheduler query required")
    scheduler = OrderedGoalScheduler(
        poses, np.arange(len(poses), dtype=np.float64) / SCHEDULER_HZ,
        query, interaction_points=surface, goal_type="se3_pose")
    hands = _canonical_hand(joints, source_id)
    if not np.isfinite(hands).all():
        raise ValueError("scheduler replay hand observations are not finite")
    wanted = set(ticks)
    result = {}
    for tick in range(max(ticks) + 1):
        decision = update_schedulers(
            [scheduler], poses[tick:tick + 1],
            np.asarray([tick / float(SCHEDULER_HZ)]), hands[tick:tick + 1],
            fast_geometry=True)[0]
        if tick in wanted:
            condition = np.asarray(decision["goal_delta"], dtype=np.float32)
            if condition.shape != (32, 3) or not np.isfinite(condition).all():
                raise ValueError("scheduler emitted an invalid goal delta")
            if decision.get("goal_semantics") != GOAL_SEMANTICS:
                raise ValueError("scheduler emitted a second/unknown goal semantic")
            result[tick] = condition.copy()
    if set(result) != wanted:
        raise ValueError("scheduler did not emit every requested window condition")
    return result


def scheduler_condition_for_tick(sequence, geometry, source_id, slot, tick):
    return scheduler_conditions_for_ticks(sequence, geometry, source_id, slot, [tick])[int(tick)]
