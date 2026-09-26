"""Fit and audit the five prespecified HF02 temporal option models on CPU.

The collection is a six-arm randomized option dataset.  Each row contains one
observed outcome, for the assigned arm, and all six candidate actions at the
trigger.  The models below are pooled outcome regressions trained only on the
observed assigned arm and evaluated for all six candidate arms.  Policy value
is the Horvitz estimator on the held-out assignment (p=1/6).

This script intentionally has no simulator or CUDA dependency.  It validates
both payloads against the frozen contract before fitting anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, Mapping, Tuple

import torch

from src.task.CmResidual.temporal_option_contract import (
    validate_frozen_contract,
    validate_record_payload,
)


MODEL_NAMES = (
    "state_only",
    "history_only",
    "history_plus_expert_id",
    "temporal_cm",
    "action_shuffled",
)
EXPERTS = (
    "balanced_e360",
    "cup_e340",
    "duck_e340",
    "mixed12_e300",
    "source_e260",
    "train5_e320",
)
ACTION_SHUFFLE_PERMUTATION = (1, 2, 3, 4, 5, 0)
RIDGE_LAMBDA = 1.0
PROPENSITY = 1.0 / 6.0
SUPPORTED_UNIT = "mm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def _history_features(records: Mapping[str, object]) -> torch.Tensor:
    n = int(records["assignment"].numel())
    return torch.cat((
        records["history_state"].reshape(n, -1).float(),
        records["history_action"].reshape(n, -1).float(),
        records["history_contact"].reshape(n, -1).float(),
    ), dim=1)


def _features(records: Mapping[str, object], model: str,
              *, training: bool) -> torch.Tensor:
    """Return [N,D] training or [N,6,D] evaluation features."""
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model}")
    n = int(records["assignment"].numel())
    assignment = records["assignment"].long()
    state = records["state"].float()
    actions = records["candidate_actions"].float()
    history = _history_features(records)
    if actions.shape != (n, 6, 18):
        raise ValueError(f"candidate action shape differs: {tuple(actions.shape)}")

    if model == "state_only":
        if training:
            return torch.cat((state, actions[torch.arange(n), assignment]), dim=1)
        return torch.cat((state[:, None, :].expand(-1, 6, -1), actions), dim=2)

    if model == "history_only":
        if training:
            return history
        return history[:, None, :].expand(-1, 6, -1)

    if model == "history_plus_expert_id":
        one_hot = torch.eye(6, dtype=torch.float32)
        if training:
            return torch.cat((history, one_hot[assignment]), dim=1)
        return torch.cat((
            history[:, None, :].expand(-1, 6, -1),
            one_hot[None, :, :].expand(n, -1, -1),
        ), dim=2)

    if model == "action_shuffled":
        actions = actions[:, ACTION_SHUFFLE_PERMUTATION, :]
    if training:
        return torch.cat((history, actions[torch.arange(n), assignment]), dim=1)
    return torch.cat((history[:, None, :].expand(-1, 6, -1), actions), dim=2)


def _target(records: Mapping[str, object], outcome: str) -> torch.Tensor:
    if outcome == "held_lift":
        return records["final_lift_success"].float()
    if outcome == "supported_lift_mm":
        # The card's secondary outcome is the raw mean over all 20 future
        # slots.  Slots without contact are already zero in the contract.
        return records["future_contact_supported_lift_m"].float().mean(dim=1) * 1000.0
    raise ValueError(f"unknown outcome: {outcome}")


def _fit_ridge(features: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Fit standardized ridge least squares with an unregularized intercept."""
    x = features.double()
    y = target.double()
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - mean) / scale
    z = torch.cat((z, torch.ones((z.shape[0], 1), dtype=z.dtype)), dim=1)
    penalty = torch.eye(z.shape[1], dtype=z.dtype) * RIDGE_LAMBDA
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def _predict(model: Mapping[str, torch.Tensor], features: torch.Tensor,
             *, held_lift: bool) -> torch.Tensor:
    shape = features.shape
    flat = features.reshape(-1, shape[-1]).double()
    z = (flat - model["mean"]) / model["scale"]
    z = torch.cat((z, torch.ones((z.shape[0], 1), dtype=z.dtype)), dim=1)
    prediction = (z @ model["weights"]).reshape(shape[:-1])
    return prediction.sigmoid() if held_lift else prediction


def _policy_value(prediction: torch.Tensor, records: Mapping[str, object],
                  outcome: str) -> Dict[str, object]:
    policy_arm = prediction.argmax(dim=1)
    observed_arm = records["assignment"].long()
    observed = _target(records, outcome).double()
    matched = policy_arm.eq(observed_arm)
    # Horvitz/IPW estimator: N^-1 sum I[pi(X)=A] Y / p(A|X).
    value = (matched.double() * observed / PROPENSITY).mean()
    matched_mean = observed[matched].mean() if bool(matched.any()) else torch.tensor(float("nan"))
    return {
        "ipw_policy_value": float(value),
        "matched_observed_mean": float(matched_mean),
        "matched_rows": int(matched.sum()),
        "policy_arm_counts": torch.bincount(policy_arm, minlength=6).tolist(),
        "finite": bool(torch.isfinite(prediction).all() and torch.isfinite(value)),
    }


def _row_counts(records: Mapping[str, object]) -> Dict[str, int]:
    assignment = records["assignment"].long()
    return {EXPERTS[i]: int((assignment == i).sum()) for i in range(6)}


def fit_probe(fit_records: Mapping[str, object], holdout_records: Mapping[str, object],
              *, model_artifact: Path) -> Dict[str, object]:
    models: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    metrics: Dict[str, Dict[str, Dict[str, object]]] = {}
    for name in MODEL_NAMES:
        train_features = _features(fit_records, name, training=True)
        eval_features = _features(holdout_records, name, training=False)
        models[name] = {}
        metrics[name] = {}
        for outcome in ("held_lift", "supported_lift_mm"):
            target = _target(fit_records, outcome)
            fitted = _fit_ridge(train_features, target)
            models[name][outcome] = fitted
            prediction = _predict(
                fitted, eval_features, held_lift=(outcome == "held_lift"))
            metrics[name][outcome] = _policy_value(
                prediction, holdout_records, outcome)

    # Save CPU coefficients and normalization so a successful head would be
    # reproducible without refitting from an opaque report.
    serializable = {
        "schema": "ref2dex.hf02_temporal_option_cpu_models.v1",
        "ridge_lambda": RIDGE_LAMBDA,
        "action_shuffle_permutation": ACTION_SHUFFLE_PERMUTATION,
        "models": models,
    }
    model_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serializable, model_artifact)

    temporal = metrics["temporal_cm"]
    history = metrics["history_only"]
    shuffled = metrics["action_shuffled"]
    held_margin_history_pp = 100.0 * (
        temporal["held_lift"]["ipw_policy_value"] -
        history["held_lift"]["ipw_policy_value"])
    held_margin_shuffled_pp = 100.0 * (
        temporal["held_lift"]["ipw_policy_value"] -
        shuffled["held_lift"]["ipw_policy_value"])
    supported_delta_history_mm = (
        temporal["supported_lift_mm"]["ipw_policy_value"] -
        history["supported_lift_mm"]["ipw_policy_value"])
    supported_delta_shuffled_mm = (
        temporal["supported_lift_mm"]["ipw_policy_value"] -
        shuffled["supported_lift_mm"]["ipw_policy_value"])
    finite = all(
        bool(metrics[model][outcome]["finite"])
        for model in MODEL_NAMES
        for outcome in ("held_lift", "supported_lift_mm")
    )
    fit_counts = _row_counts(fit_records)
    holdout_counts = _row_counts(holdout_records)
    row_gate = min(fit_counts.values()) >= 30 and min(holdout_counts.values()) >= 20
    gate = {
        "fit_min_rows_per_arm": min(fit_counts.values()),
        "holdout_min_rows_per_arm": min(holdout_counts.values()),
        "rows_pass": row_gate,
        "finite_pass": finite,
        "held_margin_vs_history_only_pp": held_margin_history_pp,
        "held_margin_vs_action_shuffled_pp": held_margin_shuffled_pp,
        "held_margin_gate_pass": (held_margin_history_pp >= 5.0 and
                                   held_margin_shuffled_pp >= 5.0),
        "supported_delta_vs_history_only_mm": supported_delta_history_mm,
        "supported_delta_vs_action_shuffled_mm": supported_delta_shuffled_mm,
        "supported_nonregression_pass": (supported_delta_history_mm >= 0.0 and
                                          supported_delta_shuffled_mm >= 0.0),
        "ranking_direction_pass": (held_margin_history_pp >= 0.0 and
                                    held_margin_shuffled_pp >= 0.0 and
                                    supported_delta_history_mm >= 0.0 and
                                    supported_delta_shuffled_mm >= 0.0),
    }
    gate["all_conditions_pass"] = all((
        gate["rows_pass"], gate["finite_pass"], gate["held_margin_gate_pass"],
        gate["supported_nonregression_pass"], gate["ranking_direction_pass"],
    ))
    decision = "PROMISING" if gate["all_conditions_pass"] else (
        "UNPROMISING" if gate["rows_pass"] and gate["finite_pass"]
        else "UNCLEAR")
    return {
        "schema": "ref2dex.hf02_temporal_expert_credit_cpu_fit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_rows": int(fit_records["assignment"].numel()),
        "holdout_rows": int(holdout_records["assignment"].numel()),
        "fit_arm_counts": fit_counts,
        "holdout_arm_counts": holdout_counts,
        "models": metrics,
        "fit_spec": {
            "normalization": "fit_split_mean_and_population_std_only",
            "ridge_lambda": RIDGE_LAMBDA,
            "held_lift_link": "sigmoid_after_ridge",
            "supported_lift_target": "mean(raw future_contact_supported_lift_m[20]) * 1000",
            "supported_lift_unit": SUPPORTED_UNIT,
            "action_shuffle_permutation": ACTION_SHUFFLE_PERMUTATION,
            "ipw_propensity": "1/6",
            "ipw_formula": "mean(I[policy(X)=A] * Y / p(A|X))",
            "argmax_tie_break": "lowest expert index",
        },
        "gate": gate,
        "decision": decision,
        "model_artifact": str(model_artifact),
        "model_artifact_sha256": _sha256(model_artifact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-records", type=Path, required=True)
    parser.add_argument("--holdout-records", type=Path, required=True)
    parser.add_argument("--collector-config", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU fit must run with CUDA unavailable")
    root = Path(__file__).resolve().parents[4]
    provenance = validate_frozen_contract(
        root, args.collector_config.resolve(), args.route_config.resolve(),
        verify_artifacts=True)
    fit_payload = torch.load(args.fit_records, map_location="cpu", weights_only=False)
    holdout_payload = torch.load(args.holdout_records, map_location="cpu", weights_only=False)
    fit_summary = validate_record_payload(
        fit_payload, provenance["collector"], provenance)
    holdout_summary = validate_record_payload(
        holdout_payload, provenance["collector"], provenance)
    report = fit_probe(
        fit_payload["records"], holdout_payload["records"],
        model_artifact=args.output.with_name("cpu_fit_models.pt"))
    report.update({
        "git_commit": _git_commit(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(args.evaluator.resolve()),
        "evaluator_git_blob_sha1": subprocess.check_output(
            ["git", "hash-object", str(args.evaluator.resolve())], text=True).strip(),
        "route_config_sha256": provenance["route_config_sha256"],
        "collector_config_sha256": provenance["collector_config_sha256"],
        "fit_records_sha256": _sha256(args.fit_records),
        "holdout_records_sha256": _sha256(args.holdout_records),
        "fit_contract_summary": fit_summary,
        "holdout_contract_summary": holdout_summary,
        "cuda_available": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "gate": report["gate"],
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
