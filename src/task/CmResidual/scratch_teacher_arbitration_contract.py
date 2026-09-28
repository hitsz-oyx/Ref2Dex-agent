"""CPU-only contract for the from-scratch Cm teacher-arbitration route.

The route is intentionally a contract, rather than a trainer.  It defines the
information available to a physical-effect model and the deterministic rule
that turns its calibrated predictions into a *distillation label*.  It never
loads a checkpoint, starts a simulator, or performs an update step.

Cm is therefore kept separate from the six-expert student substrate:

* the model observes pre-action state plus one candidate expert action;
* its targets are one-step object motion and short contact retention;
* its output chooses which expert action is used as the offline teacher label;
* the resulting student can be compared with a matched Cm-off student.

This is deliberately not an online action ranker, a value critic, or a
residual intervention.  The functions below fail closed when a row leaks
post-action fields or when the randomized candidate support is incomplete.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "ref2dex.cm_scratch_teacher_arbitration.v1"
MODEL_ID = "CM-SCRATCH-TA-20260928"
CANDIDATE_COUNT = 6
ASSIGNMENT_PROPENSITY = 1.0 / CANDIDATE_COUNT
CONTACT_HORIZON = 5


class ContractError(ValueError):
    """Raised when a preflight row or decision violates the route contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _finite_vector(value: Any, *, name: str, length: int | None = None) -> tuple[float, ...]:
    """Return a finite numeric vector without importing a tensor framework."""
    _require(isinstance(value, (list, tuple)), f"{name} must be a list or tuple")
    if length is not None:
        _require(len(value) == length, f"{name} must have length {length}")
    converted: list[float] = []
    for item in value:
        _require(isinstance(item, (int, float)) and not isinstance(item, bool),
                 f"{name} must contain numbers")
        converted.append(float(item))
    _require(all(math.isfinite(item) for item in converted),
             f"{name} contains a non-finite value")
    return tuple(converted)


def validate_row(row: Mapping[str, Any], *, candidate_count: int = CANDIDATE_COUNT) -> None:
    """Validate one randomized transition used by the scratch Cm model.

    The target fields are intentionally explicit.  ``next_*``, ``future_*``
    and terminal success fields may be stored in a source record, but they are
    not allowed in the model input.  The check catches accidental leakage
    before any fit is attempted.
    """
    _require(isinstance(row, Mapping), "row must be a mapping")
    required = {
        "episode_id", "step", "candidate_id", "candidate_action",
        "pre_action_observation", "object_lift_axis",
        "target_delta_object_local_1", "target_contact_retention_h5",
        "assignment_propensity", "split",
    }
    missing = sorted(required.difference(row))
    _require(not missing, f"row is missing fields: {missing}")
    forbidden_input_fragments = ("next_", "future_", "final_", "held_lift", "terminal")
    declared_inputs = row.get("model_inputs", (
        "pre_action_observation", "candidate_action", "candidate_id"))
    _require(isinstance(declared_inputs, (list, tuple)), "model_inputs must be a sequence")
    for name in declared_inputs:
        _require(isinstance(name, str), "model_inputs names must be strings")
        _require(not any(fragment in name for fragment in forbidden_input_fragments),
                 f"post-action field leaked into model inputs: {name}")
    candidate_id = row["candidate_id"]
    _require(isinstance(candidate_id, int) and not isinstance(candidate_id, bool),
             "candidate_id must be an integer")
    _require(0 <= candidate_id < candidate_count,
             f"candidate_id must be in [0, {candidate_count})")
    _finite_vector(row["candidate_action"], name="candidate_action")
    _finite_vector(row["pre_action_observation"], name="pre_action_observation")
    _finite_vector(row["object_lift_axis"], name="object_lift_axis", length=3)
    _finite_vector(row["target_delta_object_local_1"],
                   name="target_delta_object_local_1", length=3)
    retention = row["target_contact_retention_h5"]
    _require(isinstance(retention, (int, float)) and not isinstance(retention, bool),
             "target_contact_retention_h5 must be numeric")
    _require(math.isfinite(float(retention)) and 0.0 <= float(retention) <= 1.0,
             "target_contact_retention_h5 must be in [0, 1]")
    propensity = row["assignment_propensity"]
    _require(isinstance(propensity, (int, float)) and not isinstance(propensity, bool),
             "assignment_propensity must be numeric")
    _require(math.isclose(float(propensity), ASSIGNMENT_PROPENSITY, rel_tol=0.0, abs_tol=1e-9),
             "candidate assignment propensity must be exactly 1/6")
    _require(row["split"] in {"fit", "holdout"}, "split must be fit or holdout")


def validate_rows(rows: Iterable[Mapping[str, Any]], *, candidate_count: int = CANDIDATE_COUNT) -> dict[str, Any]:
    """Validate row schema, six-arm support, and episode-disjoint splits."""
    materialized = list(rows)
    _require(materialized, "at least one row is required")
    for row in materialized:
        validate_row(row, candidate_count=candidate_count)
    by_split: dict[str, set[Any]] = {"fit": set(), "holdout": set()}
    arm_counts: dict[str, list[int]] = {
        "fit": [0] * candidate_count,
        "holdout": [0] * candidate_count,
    }
    for row in materialized:
        split = str(row["split"])
        by_split[split].add(row["episode_id"])
        arm_counts[split][int(row["candidate_id"])] += 1
    _require(by_split["fit"].isdisjoint(by_split["holdout"]),
             "fit and holdout episodes must be disjoint")
    for split, counts in arm_counts.items():
        _require(all(count > 0 for count in counts),
                 f"{split} must contain all {candidate_count} candidate arms")
    return {
        "schema": SCHEMA,
        "rows": len(materialized),
        "fit_episodes": len(by_split["fit"]),
        "holdout_episodes": len(by_split["holdout"]),
        "arm_counts": arm_counts,
    }


def select_teacher_candidate(
        candidates: Sequence[Mapping[str, Any]],
        *,
        contact_lcb_threshold: float,
        fallback_candidate: int | None = None) -> int:
    """Apply the pre-declared teacher-label arbitration rule.

    ``contact_lcb`` is the calibrated lower bound from the scratch Cm model,
    ``delta_mean`` is the predicted local object displacement, and
    ``uncertainty_radius`` is a non-negative interval radius.  The policy
    decision is a teacher-label choice made before student distillation; it is
    not an executed online intervention.  ``fallback_candidate`` must be the
    frozen C1 observation-router label for this row.  There is deliberately no
    static source-expert fallback because that would confound Cm arbitration
    with the matched Cm-off student.
    """
    _require(candidates, "candidate set must not be empty")
    _require(0.0 <= contact_lcb_threshold <= 1.0,
             "contact_lcb_threshold must be in [0, 1]")
    valid: list[tuple[float, float, int]] = []
    seen: set[int] = set()
    for candidate in candidates:
        _require(isinstance(candidate, Mapping), "candidate must be a mapping")
        candidate_id = candidate.get("candidate_id")
        _require(isinstance(candidate_id, int) and not isinstance(candidate_id, bool),
                 "candidate_id must be an integer")
        _require(candidate_id not in seen, "candidate_id must be unique")
        seen.add(candidate_id)
        _finite_vector(candidate.get("delta_mean"), name="delta_mean", length=3)
        axis = _finite_vector(candidate.get("lift_axis"), name="lift_axis", length=3)
        _require(abs(sum(value * value for value in axis) - 1.0) <= 1e-4,
                 "lift_axis must be unit length")
        contact_lcb = candidate.get("contact_lcb")
        radius = candidate.get("uncertainty_radius")
        _require(isinstance(contact_lcb, (int, float)) and
                 isinstance(radius, (int, float)),
                 "contact_lcb and uncertainty_radius must be numeric")
        _require(math.isfinite(float(contact_lcb)) and math.isfinite(float(radius)),
                 "candidate uncertainty values must be finite")
        _require(float(radius) >= 0.0, "uncertainty_radius must be non-negative")
        if float(contact_lcb) < contact_lcb_threshold:
            continue
        delta = _finite_vector(candidate["delta_mean"], name="delta_mean", length=3)
        progress = sum(delta[index] * axis[index] for index in range(3))
        # Highest certified progress wins; uncertainty and id are deterministic
        # tie-breaks so repeated distillation manifests remain reproducible.
        valid.append((progress, -float(radius), -candidate_id))
    if not valid:
        _require(isinstance(fallback_candidate, int),
                 "router teacher label is required as fallback")
        _require(0 <= fallback_candidate < CANDIDATE_COUNT,
                 f"fallback_candidate must be in [0, {CANDIDATE_COUNT})")
        return fallback_candidate
    winning_key = max(valid)
    return -winning_key[2]
