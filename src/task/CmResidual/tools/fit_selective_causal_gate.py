"""CPU-only screen for a conservative, selective Cm intervention policy.

HF05 asks a different question from the frozen expert-selection and credit
families: can a causal model use randomized contact-stage interventions only
when its lower confidence bound predicts a safe supported-lift gain?  The
policy may abstain and keep the source action (arm 0).  All models are fit on
seeds 250/251; seeds 252/253 are opened once for held-out off-policy scoring.

The script validates the original collector contracts and never imports Isaac
Gym or starts CUDA collection.  Bootstrap resamples are fit-only and are used
only to construct the predeclared conservative gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Dict, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[4]
N_ENV = 64
FOLLOWUP_HORIZON = 5
ARMS = (-1, 0, 1)
PROPENSITY = 1.0 / 3.0
RIDGE_LAMBDA = 25.0
N_MODEL_BOOTSTRAPS = 128
N_CLUSTER_BOOTSTRAPS = 500
BOOTSTRAP_SEED = 2026092606
SHUFFLE_SEED = 2026092605
SUPPORTED_GAIN_LCB_MM = 5.0
CONTACT_LOSS_UCB = 0.02
MIN_INTERVENTION_FRACTION = 0.05
MAX_INTERVENTION_FRACTION = 0.50

CHECKPOINT_SHA256 = (
    "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
)
MOTION_MANIFEST_SHA256 = (
    "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
)

RUNS = {
    250: {
        "split": "fit", "assignment_seed": 20260925250,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_threearm_s250_n64_v1/transitions.pt",
        "transition_sha256": "7437ed71189fbfcbf1ef25f76d9bbc55d97ae67ba356a2bccdb4657346c1ca9b",
        "manifest_sha256": "f1b2f760c030e6d635e2f46edac70ebb42c7373e5732b2367d8fda09f1d6ff5a",
    },
    251: {
        "split": "fit", "assignment_seed": 20260925251,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_threearm_s251_n64_v1/transitions.pt",
        "transition_sha256": "c52d9fa18fd4e7bcf010ef58324229be1d65820e1565359b09cfa9e96196e774",
        "manifest_sha256": "fab08f1e249c6cf230e1422b4fb0df2ec3d8f62e775cfa990eb27b9e9e528f0b",
    },
    252: {
        "split": "holdout", "assignment_seed": 20260925252,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_threearm_s252_n64_v1/transitions.pt",
        "transition_sha256": "f9928645f7250d7383922772a216e1a9faa0c41c81c388f2768c87fdbb2a0f38",
        "manifest_sha256": "35da344c89b87933ae4848b9680a2bf41c7476ad6f1079ae6f614d1f61f140a3",
    },
    253: {
        "split": "holdout", "assignment_seed": 20260925253,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_threearm_s253_n64_v2/transitions.pt",
        "transition_sha256": "724bfd3709da60bc665e9873df91ea4785490e40c07d3350904fb60092c96330",
        "manifest_sha256": "204cb2ce59943de4617c3c472195cdc3edc54b424fa38a3eddba639b8eb08139",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all())


def _shape(records: Mapping[str, torch.Tensor], name: str,
           shape: tuple[int, ...]) -> None:
    if tuple(records[name].shape) != shape:
        raise ValueError(f"{name} shape {tuple(records[name].shape)} != {shape}")


def _validate_manifest(manifest: Mapping[str, object], seed: int,
                       spec: Mapping[str, object]) -> None:
    expected = {
        "run_status": "COMPLETED",
        "schema": "ref2dex.randomized_action_followup.v1",
        "seed": seed,
        "assignment_seed": spec["assignment_seed"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
        "followup_horizon": FOLLOWUP_HORIZON,
        "record_final_outcome": True,
        "source_actor_role": "self_trained",
        "three_arm_randomized": True,
        "policy_mode": "random",
    }
    for key, wanted in expected.items():
        if manifest.get(key) != wanted:
            raise ValueError(
                f"manifest {seed} {key}={manifest.get(key)!r}, expected {wanted!r}"
            )
    if manifest.get("source_objects") != ["airplane"]:
        raise ValueError(f"manifest {seed} object drift")
    if manifest.get("intervention_axes") != [2]:
        raise ValueError(f"manifest {seed} action-axis drift")


def load_rows(seed: int) -> Dict[str, object]:
    """Load one pinned randomized run and retain valid first-episode rows."""
    spec = RUNS[seed]
    path = Path(spec["path"])
    if sha256(path) != spec["transition_sha256"]:
        raise ValueError(f"transition SHA drift for seed {seed}")
    manifest_path = path.parent / "run_manifest.json"
    if sha256(manifest_path) != spec["manifest_sha256"]:
        raise ValueError(f"manifest SHA drift for seed {seed}")
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, seed, spec)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.randomized_action_followup.v1":
        raise ValueError(f"payload schema drift for seed {seed}")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"seed {seed} missing records")
    required = (
        "q", "dof_vel", "object_state", "base_action", "executed_action",
        "assignment", "intervention_valid", "pre_contact", "progress",
        "next_q", "next_object_state", "next_contact", "followup_progress",
        "followup_reset", "final_lift_success", "final_max_contact_lift_m",
        "final_contact_fraction", "final_episode_steps", "env_id", "motion_id",
        "start_frame",
    )
    missing = [name for name in required if name not in records]
    if missing:
        raise ValueError(f"seed {seed} missing fields {missing}")
    for name in required:
        if not isinstance(records[name], torch.Tensor) or not _finite(records[name]):
            raise ValueError(f"seed {seed} invalid/nonfinite field {name}")
    for name in ("q", "dof_vel", "base_action", "executed_action", "next_q"):
        _shape(records, name, (N_ENV, 18))
    for name in ("object_state", "next_object_state"):
        _shape(records, name, (N_ENV, 13))
    for name in (
        "assignment", "intervention_valid", "pre_contact", "progress",
        "next_contact", "followup_progress", "followup_reset", "final_lift_success",
        "final_max_contact_lift_m", "final_contact_fraction", "final_episode_steps",
        "env_id", "motion_id", "start_frame",
    ):
        _shape(records, name, (N_ENV,))

    assignment = records["assignment"].to(torch.int64)
    if set(assignment.unique().tolist()) - set(ARMS):
        raise ValueError(f"seed {seed} assignment is not three-arm")
    valid = records["pre_contact"].bool() & records["intervention_valid"].bool()
    valid &= records["followup_reset"].eq(0)
    valid &= records["followup_progress"].eq(records["progress"] + FOLLOWUP_HORIZON)
    valid &= records["final_episode_steps"].gt(records["followup_progress"])
    if int(valid.sum()) < 30:
        raise ValueError(f"seed {seed} too few valid rows")

    delta = records["executed_action"] - records["base_action"]
    expected = assignment.to(delta.dtype) * 0.1
    if not torch.allclose(delta[valid, 2], expected[valid], atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} wrist-z dose mismatch")
    if not torch.allclose(delta[valid, :2], torch.zeros_like(delta[valid, :2]), atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} non-wrist action change")
    if not torch.allclose(delta[valid, 3:], torch.zeros_like(delta[valid, 3:]), atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} non-wrist action change")
    if records["motion_id"].unique().tolist() != [0]:
        raise ValueError(f"seed {seed} motion drift")

    # The causal model sees only pre-action information.  The supported-lift
    # target is a fixed physical composite: maximum contact-supported lift
    # multiplied by first-episode contact fraction.
    features = torch.cat((
        records["q"], records["dof_vel"], records["object_state"], records["base_action"]
    ), dim=1)[valid].float()
    output: Dict[str, object] = {
        "features": features,
        "assignment": assignment[valid],
        "held_lift": records["final_lift_success"][valid].float(),
        "supported_lift_mm": (
            records["final_max_contact_lift_m"][valid].float() * 1000.0
            * records["final_contact_fraction"][valid].float()
        ),
        "max_contact_lift_mm": records["final_max_contact_lift_m"][valid].float() * 1000.0,
        "contact_fraction": records["final_contact_fraction"][valid].float(),
        "env_id": records["env_id"][valid].long(),
        "seed": seed,
        "split": spec["split"],
        "valid_rows": int(valid.sum()),
        "manifest_sha256": spec["manifest_sha256"],
        "transition_sha256": spec["transition_sha256"],
    }
    return output


def concatenate(rows: Sequence[Mapping[str, object]]) -> Dict[str, torch.Tensor]:
    keys = (
        "features", "assignment", "held_lift", "supported_lift_mm",
        "max_contact_lift_mm", "contact_fraction", "env_id",
    )
    return {key: torch.cat([row[key] for row in rows]) for key in keys}  # type: ignore[arg-type]


def _design(features: torch.Tensor, mean: torch.Tensor,
            scale: torch.Tensor) -> torch.Tensor:
    z = (features.double() - mean) / scale
    return torch.cat((z, torch.ones((len(z), 1), dtype=z.dtype)), dim=1)


def _fit_linear(x: torch.Tensor, y: torch.Tensor, ridge: float = RIDGE_LAMBDA) -> torch.Tensor:
    penalty = torch.eye(x.shape[1], dtype=torch.float64)
    penalty[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + ridge * penalty, x.T @ y.double())


def _fit_arm_bootstraps(
    fit_rows: Mapping[str, torch.Tensor], target_name: str,
    *, labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean/scale, point predictions weights, bootstrap weights."""
    features = fit_rows["features"].double()
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-6)
    x = _design(fit_rows["features"], mean, scale)
    arms = fit_rows["assignment"] if labels is None else labels
    target = fit_rows[target_name].double()
    point_weights = []
    bootstrap_weights = []
    target_offsets = {"supported_lift_mm": 11, "contact_fraction": 23, "held_lift": 37}
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED + target_offsets[target_name])
    for arm in ARMS:
        indices = torch.nonzero(arms.eq(arm), as_tuple=False).flatten()
        if len(indices) < 10:
            raise ValueError(f"arm {arm} has too few fit rows")
        point_weights.append(_fit_linear(x[indices], target[indices]))
        samples = []
        for _ in range(N_MODEL_BOOTSTRAPS):
            draw = indices[torch.randint(len(indices), (len(indices),), generator=generator)]
            samples.append(_fit_linear(x[draw], target[draw]))
        bootstrap_weights.append(torch.stack(samples))
    return mean, scale, torch.stack(point_weights), torch.stack(bootstrap_weights)


def _predict_arm_models(
    holdout_rows: Mapping[str, torch.Tensor], mean: torch.Tensor,
    scale: torch.Tensor, point_weights: torch.Tensor,
    bootstrap_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _design(holdout_rows["features"], mean, scale)
    point = torch.einsum("nd,ad->na", x, point_weights)
    boot = torch.einsum("nd,abd->nab", x, bootstrap_weights)
    return point, boot


def _policy_from_predictions(
    supported_point: torch.Tensor, supported_boot: torch.Tensor,
    contact_point: torch.Tensor, contact_boot: torch.Tensor,
    *, use_uncertainty: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build one fixed abstaining policy from arm predictions."""
    # Arm order is [-1, 0, +1].  Lower/upper quantiles are fit-only
    # bootstrap summaries and never use held-out labels.
    if use_uncertainty:
        supported_delta = supported_boot[:, :, [0, 2]] - supported_boot[:, :, 1:2]
        contact_delta = contact_boot[:, :, [0, 2]] - contact_boot[:, :, 1:2]
        lcb = torch.quantile(supported_delta, 0.10, dim=1)
        ucb_contact = torch.quantile(contact_delta, 0.90, dim=1)
    else:
        supported_delta = supported_point[:, [0, 2]] - supported_point[:, 1:2]
        contact_delta = contact_point[:, [0, 2]] - contact_point[:, 1:2]
        lcb = supported_delta
        ucb_contact = contact_delta
    eligible = (lcb >= SUPPORTED_GAIN_LCB_MM) & (ucb_contact <= CONTACT_LOSS_UCB)
    best = torch.argmax(torch.where(eligible, lcb, torch.full_like(lcb, -float("inf"))), dim=1)
    has_candidate = eligible.any(dim=1)
    policy = torch.zeros((len(best),), dtype=torch.int64)
    # candidate index 0 -> arm -1; index 1 -> arm +1
    policy[has_candidate] = torch.where(best[has_candidate].eq(0), -1, 1)
    diagnostics = {
        "lcb_supported_delta_mm": lcb,
        "ucb_contact_delta": ucb_contact,
        "eligible": eligible,
        "has_candidate": has_candidate,
    }
    return policy, diagnostics


def _ipw_value(rows: Mapping[str, torch.Tensor], policy: torch.Tensor,
               target_name: str) -> tuple[float, int]:
    assignment = rows["assignment"]
    target = rows[target_name].double()
    match = assignment.eq(policy)
    denominator = (match.double() / PROPENSITY).sum()
    if denominator <= 0:
        return float("nan"), 0
    value = (match.double() * target / PROPENSITY).sum() / denominator
    return float(value), int(match.sum())


def _cluster_ci(rows: Mapping[str, torch.Tensor], policy: torch.Tensor,
                target_name: str) -> tuple[float, float]:
    # Include the simulator seed in the cluster key; env IDs repeat across
    # seeds but are independent runs.
    keys = rows["cluster_key"]
    unique = torch.unique(keys)
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED + 71)
    values = []
    for _ in range(N_CLUSTER_BOOTSTRAPS):
        draw = unique[torch.randint(len(unique), (len(unique),), generator=generator)]
        keep = torch.zeros_like(keys, dtype=torch.bool)
        for key in draw:
            keep |= keys.eq(key)
        subset = {name: value[keep] for name, value in rows.items() if isinstance(value, torch.Tensor)}
        value, matched = _ipw_value(subset, policy[keep], target_name)
        if matched:
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, 0.025)), float(torch.quantile(tensor, 0.975))


def _evaluate_policies(rows: Mapping[str, torch.Tensor], policies: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    report = {}
    for name, policy in policies.items():
        arm_counts = {str(arm): int(policy.eq(arm).sum()) for arm in ARMS}
        intervention_fraction = float(policy.ne(0).double().mean())
        targets = {}
        for target in ("held_lift", "supported_lift_mm", "max_contact_lift_mm", "contact_fraction"):
            value, matched = _ipw_value(rows, policy, target)
            lo, hi = _cluster_ci(rows, policy, target)
            targets[target] = {"value": value, "matched_rows": matched, "ci95": [lo, hi]}
        report[name] = {
            "arm_counts": arm_counts,
            "intervention_fraction": intervention_fraction,
            "targets": targets,
        }
    return report


def run(output_dir: Path) -> Dict[str, object]:
    fit_parts = [load_rows(seed) for seed in (250, 251)]
    holdout_parts = [load_rows(seed) for seed in (252, 253)]
    fit = concatenate(fit_parts)
    holdout = concatenate(holdout_parts)
    fit["cluster_key"] = torch.cat([
        rows["env_id"] + int(rows["seed"]) * 1000 for rows in fit_parts
    ])
    holdout["cluster_key"] = torch.cat([
        rows["env_id"] + int(rows["seed"]) * 1000 for rows in holdout_parts
    ])

    fit_counts = {str(arm): int(fit["assignment"].eq(arm).sum()) for arm in ARMS}
    holdout_counts = {str(arm): int(holdout["assignment"].eq(arm).sum()) for arm in ARMS}
    if min(*fit_counts.values(), *holdout_counts.values()) < 15:
        raise ValueError("minimum arm-count gate failed")
    if not all(_finite(value) for value in fit.values() if isinstance(value, torch.Tensor)):
        raise FloatingPointError("nonfinite fit values")
    if not all(_finite(value) for value in holdout.values() if isinstance(value, torch.Tensor)):
        raise FloatingPointError("nonfinite holdout values")

    fitted = {}
    predictions = {}
    for label, labels in (("cm", None), ("shuffled", None)):
        assigned = fit["assignment"]
        if label == "shuffled":
            generator = torch.Generator().manual_seed(SHUFFLE_SEED)
            assigned = assigned[torch.randperm(len(assigned), generator=generator)]
        models = {}
        preds = {}
        for target in ("supported_lift_mm", "contact_fraction", "held_lift"):
            mean, scale, point_w, boot_w = _fit_arm_bootstraps(
                fit, target, labels=assigned if label == "shuffled" else None
            )
            point, boot = _predict_arm_models(holdout, mean, scale, point_w, boot_w)
            models[target] = {"mean": mean, "scale": scale, "point": point_w, "bootstrap": boot_w}
            preds[target] = (point, boot)
        fitted[label] = models
        predictions[label] = preds

    cm_supported_point, cm_supported_boot = predictions["cm"]["supported_lift_mm"]
    cm_contact_point, cm_contact_boot = predictions["cm"]["contact_fraction"]
    sh_supported_point, sh_supported_boot = predictions["shuffled"]["supported_lift_mm"]
    sh_contact_point, sh_contact_boot = predictions["shuffled"]["contact_fraction"]
    cm_policy, cm_diag = _policy_from_predictions(
        cm_supported_point, cm_supported_boot, cm_contact_point, cm_contact_boot,
        use_uncertainty=True,
    )
    point_policy, point_diag = _policy_from_predictions(
        cm_supported_point, cm_supported_boot, cm_contact_point, cm_contact_boot,
        use_uncertainty=False,
    )
    shuffled_policy, shuffled_diag = _policy_from_predictions(
        sh_supported_point, sh_supported_boot, sh_contact_point, sh_contact_boot,
        use_uncertainty=True,
    )
    base_policy = torch.zeros_like(cm_policy)
    policies = {
        "always_base": base_policy,
        "cm_selective": cm_policy,
        "cm_point_gate": point_policy,
        "shuffled_selective": shuffled_policy,
    }
    policy_report = _evaluate_policies(holdout, policies)

    cm_report = policy_report["cm_selective"]
    base_report = policy_report["always_base"]
    shuffled_report = policy_report["shuffled_selective"]
    cm_intervention = cm_report["intervention_fraction"]
    value_gain = cm_report["targets"]["held_lift"]["value"] - base_report["targets"]["held_lift"]["value"]
    shuffle_gain = cm_report["targets"]["held_lift"]["value"] - shuffled_report["targets"]["held_lift"]["value"]
    supported_gain = cm_report["targets"]["supported_lift_mm"]["value"] - base_report["targets"]["supported_lift_mm"]["value"]
    contact_delta = cm_report["targets"]["contact_fraction"]["value"] - base_report["targets"]["contact_fraction"]["value"]
    rows_pass = min(*fit_counts.values(), *holdout_counts.values()) >= 15
    finite_pass = True
    nontrivial_pass = MIN_INTERVENTION_FRACTION <= cm_intervention <= MAX_INTERVENTION_FRACTION
    policy_gain_pass = value_gain >= 0.05 and shuffle_gain >= 0.05
    safety_pass = supported_gain >= 0.0 and contact_delta >= -CONTACT_LOSS_UCB
    gate = {
        "minimum_rows_per_arm": 15,
        "relative_policy_margin_pp": 5.0,
        "supported_gain_min_mm": 0.0,
        "contact_loss_max_fraction": CONTACT_LOSS_UCB,
        "intervention_fraction_range": [MIN_INTERVENTION_FRACTION, MAX_INTERVENTION_FRACTION],
        "rows_pass": rows_pass,
        "finite_pass": finite_pass,
        "nontrivial_pass": nontrivial_pass,
        "policy_gain_pass": policy_gain_pass,
        "safety_pass": safety_pass,
        "all_conditions_pass": rows_pass and finite_pass and nontrivial_pass and policy_gain_pass and safety_pass,
    }
    status = "PROMISING" if gate["all_conditions_pass"] else "UNPROMISING"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "causal_gate_models.pt"
    torch.save({
        "schema": "ref2dex.hf05_selective_causal_gate_models.v1",
        "ridge_lambda": RIDGE_LAMBDA,
        "bootstrap_count": N_MODEL_BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "fitted": fitted,
    }, model_path)
    script_path = Path(__file__).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "schema": "ref2dex.hf05_selective_causal_gate_cpu_audit.v1",
        "probe_id": "P-20260926-selective-causal-gate",
        "status": status,
        "git_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_available": bool(torch.cuda.is_available()),
        "substrate": {
            "object": "airplane",
            "motion_ids": [0],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
        },
        "splits": {
            "fit": {"seeds": [250, 251], "rows": int(len(fit["assignment"])), "arm_counts": fit_counts},
            "holdout": {"seeds": [252, 253], "rows": int(len(holdout["assignment"])), "arm_counts": holdout_counts},
        },
        "model": {
            "features": "pre-action q+dof_vel+object_state+base_action (67D)",
            "arm_models": "separate ridge outcome models with fit-only normalization",
            "ridge_lambda": RIDGE_LAMBDA,
            "bootstrap_count": N_MODEL_BOOTSTRAPS,
            "supported_lift_target": "final_max_contact_lift_mm * final_contact_fraction",
            "gate": "10th-percentile supported-lift gain >=5mm and 90th-percentile contact delta <=2pp",
        },
        "policies": policy_report,
        "diagnostics": {
            "cm_selective_candidate_fraction": float(cm_diag["has_candidate"].double().mean()),
            "cm_point_gate_candidate_fraction": float(point_diag["has_candidate"].double().mean()),
            "shuffled_selective_candidate_fraction": float(shuffled_diag["has_candidate"].double().mean()),
        },
        "relative": {
            "cm_vs_base_held_lift_gain": value_gain,
            "cm_vs_shuffled_held_lift_gain": shuffle_gain,
            "cm_vs_base_supported_lift_gain_mm": supported_gain,
            "cm_vs_base_contact_delta": contact_delta,
        },
        "gate": gate,
        "inputs": {
            str(seed): {
                "transition_sha256": RUNS[seed]["transition_sha256"],
                "manifest_sha256": RUNS[seed]["manifest_sha256"],
                "assignment_seed": RUNS[seed]["assignment_seed"],
                "valid_rows": int(row["valid_rows"]),
            }
            for seed, row in zip((250, 251, 252, 253), fit_parts + holdout_parts)
        },
        "artifacts": {
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256(model_path),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": sha256(script_path),
        },
        "decision": {
            "status": status,
            "gpu_started": False,
            "collector_started": False,
            "ppo_started": False,
            "online_probe_started": False,
            "reason": "selective causal gate did not meet its predeclared policy and safety conditions" if status != "PROMISING" else "all predeclared policy and safety conditions passed",
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"status": report["status"], "gate": report["gate"], "relative": report["relative"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
