"""Auditable offline examples for the Stage 1 goal-conditioned policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch

from .goal_policy import ACTIVE_ACTION, DOF, GOAL_HORIZON, goal_features


@dataclass(frozen=True)
class GoalPolicyDataset:
    """Flattened examples plus provenance-preserving environment/tick ids."""

    features: np.ndarray
    labels: np.ndarray
    env_ids: np.ndarray
    ticks: np.ndarray

    def __post_init__(self):
        if self.features.ndim != 2 or self.labels.ndim != 2:
            raise ValueError("dataset arrays must be matrices")
        if len(self.features) != len(self.labels):
            raise ValueError("feature/label row count mismatch")
        if self.labels.shape[1] != DOF or len(self.env_ids) != len(self.features):
            raise ValueError("invalid dataset metadata shape")
        if len(self.ticks) != len(self.features):
            raise ValueError("tick metadata shape mismatch")
        if not all(np.isfinite(x).all() for x in (self.features, self.labels)):
            raise ValueError("dataset contains nonfinite values")


def _array(name: str, value, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError("{} must be finite with {} dimensions".format(name, ndim))
    return result


def build_goal_policy_dataset(trajectory: Mapping[str, np.ndarray],
                              reference_hand, reference_obj,
                              roles: Sequence[str], role: str = "tracker") -> GoalPolicyDataset:
    """Build state/goal/action rows without accepting future robot q.

    ``trajectory`` must contain measured state at T+1 frames and native action at
    T controls.  The only future values used to construct the student input are
    ``reference_hand`` and ``reference_obj``.  Labels are copied from the
    selected role's recorded native action and passive coupled coordinates are
    zeroed to match the native adapter.
    """
    q = _array("dof_position", trajectory["dof_position"], 3)
    dq = _array("dof_velocity", trajectory["dof_velocity"], 3)
    hand = _array("hand_keypoints", trajectory["hand_keypoints"], 4)
    obj = _array("object_pose", trajectory["object_pose"], 4)
    velocity = _array("object_velocity", trajectory["object_velocity"], 3)
    action = _array("action", trajectory["action"], 3)
    if q.shape[-1] != DOF or dq.shape != q.shape:
        raise ValueError("trajectory q/dq must be [T+1, E, 18]")
    if hand.shape[:2] != q.shape[:2] or hand.shape[2:] != (11, 3):
        raise ValueError("trajectory hand shape mismatch")
    if obj.shape[:2] != q.shape[:2] or obj.shape[2:] != (4, 4):
        raise ValueError("trajectory object shape mismatch")
    if velocity.shape[:2] != q.shape[:2] or velocity.shape[2:] != (6,):
        raise ValueError("trajectory object velocity shape mismatch")
    controls, envs = action.shape[:2]
    if q.shape[0] != controls + 1 or action.shape[2] != DOF:
        raise ValueError("trajectory action/state time alignment mismatch")
    roles = list(roles)
    if len(roles) != envs:
        raise ValueError("role metadata does not match trajectory environments")
    selected = np.asarray([i for i, value in enumerate(roles) if value == role], dtype=np.int64)
    if len(selected) < 2:
        raise ValueError("at least two '{}' environments are required".format(role))

    reference_hand = _array("reference_hand", reference_hand, 3)
    reference_obj = _array("reference_obj", reference_obj, 3)
    if reference_hand.shape[1:] != (11, 3) or reference_obj.shape[1:] != (4, 4):
        raise ValueError("reference hand/object shape mismatch")
    if len(reference_hand) != len(reference_obj) or len(reference_hand) < q.shape[0]:
        raise ValueError("reference must cover all trajectory state frames")

    # One common fixed reference is intentionally broadcast over selected
    # environments.  It is a goal, never a future q or command label.
    index = np.minimum(
        np.arange(controls)[:, None] + 1 + np.arange(GOAL_HORIZON)[None, :],
        len(reference_hand) - 1,
    )
    goal_hand = reference_hand[index]  # [T, H, 11, 3]
    goal_obj = reference_obj[index]    # [T, H, 4, 4]
    goal_hand = np.broadcast_to(goal_hand[:, None],
                                (controls, len(selected), GOAL_HORIZON, 11, 3))
    goal_obj = np.broadcast_to(goal_obj[:, None],
                               (controls, len(selected), GOAL_HORIZON, 4, 4))

    previous = np.zeros_like(action[:controls, selected])
    if controls > 1:
        previous[1:] = action[:controls - 1, selected]
    # Reorder to [environment, time] before flattening so an environment split
    # cannot accidentally put adjacent frames in both train and validation.
    env_q = q[:controls, selected].transpose(1, 0, 2)
    env_dq = dq[:controls, selected].transpose(1, 0, 2)
    env_hand = hand[:controls, selected].transpose(1, 0, 2, 3)
    env_obj = obj[:controls, selected].transpose(1, 0, 2, 3)
    env_vel = velocity[:controls, selected].transpose(1, 0, 2)
    env_previous = previous.transpose(1, 0, 2)
    env_goal_hand = goal_hand.transpose(1, 0, 2, 3, 4)
    env_goal_obj = goal_obj.transpose(1, 0, 2, 3, 4)
    with torch.inference_mode():
        features = goal_features(
            torch.from_numpy(env_q.reshape(-1, DOF)),
            torch.from_numpy(env_dq.reshape(-1, DOF)),
            torch.from_numpy(env_hand.reshape(-1, 11, 3)),
            torch.from_numpy(env_obj.reshape(-1, 4, 4)),
            torch.from_numpy(env_vel.reshape(-1, 6)),
            torch.from_numpy(env_previous.reshape(-1, DOF)),
            torch.from_numpy(env_goal_hand.reshape(-1, GOAL_HORIZON, 11, 3)),
            torch.from_numpy(env_goal_obj.reshape(-1, GOAL_HORIZON, 4, 4)),
        ).numpy()
    labels = action[:controls, selected].transpose(1, 0, 2).reshape(-1, DOF).copy()
    passive = sorted(set(range(DOF)) - set(ACTIVE_ACTION))
    labels[:, passive] = 0.0
    return GoalPolicyDataset(
        features=features.astype(np.float32, copy=False),
        labels=labels.astype(np.float32, copy=False),
        env_ids=np.repeat(selected, controls),
        ticks=np.tile(np.arange(controls, dtype=np.int64), len(selected)),
    )


def split_by_environment(dataset: GoalPolicyDataset,
                         validation_fraction: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/validation row masks with whole environments held out."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    environments = np.unique(dataset.env_ids)
    if len(environments) < 2:
        raise ValueError("at least two environments are required for a split")
    count = max(1, int(np.ceil(len(environments) * validation_fraction)))
    train_envs = environments[:-count]
    validation_envs = environments[-count:]
    train = np.isin(dataset.env_ids, train_envs)
    validation = np.isin(dataset.env_ids, validation_envs)
    if not train.any() or not validation.any():
        raise ValueError("environment split produced an empty partition")
    return train, validation

