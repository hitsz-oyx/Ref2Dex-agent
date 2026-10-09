"""Frozen analytic wrist decoder used to isolate the learned finger branch.

The decoder consumes only current live ``q/dq`` and a future hand-keypoint
window. Its coefficients are train-only approach-frame PD statistics; no
future joint state, force, or commanded action is read at execution time.
"""

import json
from pathlib import Path

import numpy as np

from .object_relative_servo import recover_wrist
from .retargeter import closest_euler, wrist_rotation


WRIST_DOF = 6
EXPECTED_INDEPENDENT_DOFS = tuple(range(WRIST_DOF))


def root_template(source_hand, source_q):
    """Calibrate fixed wrist-root points from one source reset frame."""
    hand = np.asarray(source_hand, dtype="float32")
    q = np.asarray(source_q, dtype="float32")
    if hand.shape != (11, 3) or q.shape != (18,):
        raise ValueError("source reset hand/q shapes must be (11,3)/(18,)")
    template = np.einsum("ij,pj->pi", wrist_rotation(q).T, hand - q[:3])
    if not np.isfinite(template).all():
        raise ValueError("nonfinite wrist root template")
    return template.astype("float32")


def recover_wrist_sequence(future_hand, template, reference_q):
    """Recover one absolute wrist state per future hand-keypoint frame."""
    future = np.asarray(future_hand, dtype="float32")
    reference = np.asarray(reference_q, dtype="float32")
    if (future.ndim != 4 or future.shape[2:] != (11, 3)
            or reference.shape != (future.shape[0], 18)):
        raise ValueError("future hand/reference batch shape mismatch")
    result = np.empty((future.shape[0], future.shape[1], 18), dtype="float32")
    prior = reference.copy()
    for offset in range(future.shape[1]):
        prior = recover_wrist(future[:, offset], template, prior)
        result[:, offset] = prior
    if not np.isfinite(result).all():
        raise ValueError("nonfinite recovered wrist sequence")
    return result


def load_pd_statistics(path):
    """Load and validate frozen train-only one-step wrist coefficients."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema") != "ref2dex.pd-step-inverse.v1":
        raise ValueError("unexpected wrist PD statistics schema")
    if value.get("engineering_only") is not True:
        raise ValueError("wrist PD statistics must be engineering-only")
    if value.get("statistics_scope") != "frozen_ref7_train_launches_only":
        raise ValueError("wrist statistics must use the frozen train-launch scope")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("wrist statistics must retain the three frozen sources")
    for source in sources:
        if (not isinstance(source, dict) or len(source.get("sha256", "")) != 64
                or source.get("ticks") != [0, 39]):
            raise ValueError("wrist statistics source provenance is incomplete")
    if tuple(value.get("independent_dofs", ()))[:WRIST_DOF] != EXPECTED_INDEPENDENT_DOFS:
        raise ValueError("wrist coefficients must lead with DOFs 0..5")
    coefficients = np.asarray(value.get("coefficients"), dtype="float32")
    if coefficients.shape[0] < WRIST_DOF or coefficients.shape[1:] != (2,):
        raise ValueError("wrist coefficients must have shape [>=6,2]")
    coefficients = coefficients[:WRIST_DOF]
    if not np.isfinite(coefficients).all() or np.any(coefficients[:, 0] <= 0):
        raise ValueError("invalid wrist PD coefficients")
    return coefficients, value


def pd_inverse_wrist_action(current_q, current_dq, desired_q, coefficients):
    """Return normalized six-DOF wrist commands from a one-step PD inverse."""
    current = np.asarray(current_q, dtype="float32")
    velocity = np.asarray(current_dq, dtype="float32")
    desired = np.asarray(desired_q, dtype="float32")
    coeff = np.asarray(coefficients, dtype="float32")
    if (current.ndim != 2 or current.shape[1] != 18 or velocity.shape != current.shape
            or desired.shape != current.shape or coeff.shape != (WRIST_DOF, 2)):
        raise ValueError("wrist PD inverse shape mismatch")
    target = current[:, :WRIST_DOF].copy()
    target_rot = closest_euler(wrist_rotation(desired), current[:, 3:6])
    target[:, :3] += coeff[:3, 0] * (desired[:, :3] - current[:, :3])
    target[:, :3] += coeff[:3, 1] * velocity[:, :3]
    target[:, 3:6] += coeff[3:6, 0] * (target_rot - current[:, 3:6])
    target[:, 3:6] += coeff[3:6, 1] * velocity[:, 3:6]
    action = np.zeros_like(current, dtype="float32")
    action[:, :3] = target[:, :3] - current[:, :3]
    action[:, 3:6] = (target[:, 3:6] - current[:, 3:6]) / np.pi
    if not np.isfinite(action).all():
        raise ValueError("nonfinite wrist PD action")
    return action


def replace_wrist_action(model_action, current_q, current_dq, desired_q, coefficients):
    """Keep the model's finger command and replace only normalized wrist DOFs."""
    model = np.asarray(model_action, dtype="float32")
    if model.ndim != 2 or model.shape[1] != 18:
        raise ValueError("model action must be [N,18]")
    result = model.copy()
    result[:, :WRIST_DOF] = pd_inverse_wrist_action(
        current_q, current_dq, desired_q, coefficients)[:, :WRIST_DOF]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite hybrid wrist/finger action")
    return result
