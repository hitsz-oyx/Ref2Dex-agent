"""Audit post-action handflow credit for future held-lift on CPU.

This is the first probe in the HF03 contact-supported-credit family.  It uses
four already completed randomized airplane runs (seeds 246--249); it does not
start a simulator or require CUDA.  The treatment is a one-step +/-0.1 wrist-z
perturbation at a contacting state.  The question is deliberately narrower
than policy utility: does the immediately observed post-action handflow add
predictive information about the later first-episode held-lift outcome over a
pre-action state/action model?

The fit split is seeds 246/247 and the holdout split is 248/249.  All
normalization, ridge fitting, and the placebo permutations are deterministic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, Mapping

import torch

ROOT = Path(__file__).resolve().parents[4]
N_ENV = 64
FOLLOWUP_HORIZON = 5
RIDGE_LAMBDA = 10.0
SHUFFLE_SEED = 2026092603

CHECKPOINT_SHA256 = (
    "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
)
MOTION_MANIFEST_SHA256 = (
    "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"
)

RUNS = {
    246: {
        "split": "fit",
        "assignment_seed": 20260925246,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_value_pilot_s246_n64_v1/transitions.pt",
        "transition_sha256": "55eae45b7115cde435f0ef051cdf12637c8870d36fe82351e7eb7810c100edd8",
        "manifest_sha256": "7895d522de4fff2496ab7456d130daf6b06c3b121d14d3d500674a6a3f6f1441",
    },
    247: {
        "split": "fit",
        "assignment_seed": 20260925247,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_value_pilot_s247_n64_v1/transitions.pt",
        "transition_sha256": "860af224e736a2cb7cd773f759fdb08ffc339cffc2d14f35a599b8f5c3953a1b",
        "manifest_sha256": "7127314f398fc79ad1d4b6d69153098d97e8c4652c7295e5cefb5802c30f6b56",
    },
    248: {
        "split": "holdout",
        "assignment_seed": 20260925248,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_value_pilot_s248_n64_v1/transitions.pt",
        "transition_sha256": "208342be31702732275df25803a31f48c6e8a7847818f046464eb2c2a6fd6ec1",
        "manifest_sha256": "b98c9a7be67ba59a39e115f9297a283d3dcd7c313a246a829b0ba0eca8f5aae3",
    },
    249: {
        "split": "holdout",
        "assignment_seed": 20260925249,
        "path": ROOT / "outputs/CmResidual/agent_postcontact_value_pilot_s249_n64_v1/transitions.pt",
        "transition_sha256": "52a166812df126b1bc99fe717a91c6ace829c01b865dfa2d91a853006dd44d16",
        "manifest_sha256": "fc3685ef5705d5c3e7f737446f74c92b9903c34989834adee7871d730cf72418",
    },
}

MODEL_NAMES = (
    "state_only",
    "action_aware",
    "post_handflow",
    "action_shuffled",
    "post_handflow_shuffled",
)
TARGET_NAMES = ("held_lift", "max_contact_lift_mm", "contact_fraction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all())


def _require_shape(records: Mapping[str, torch.Tensor], name: str,
                   shape: tuple[int, ...]) -> None:
    value = records[name]
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")


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
    }
    for key, wanted in expected.items():
        if manifest.get(key) != wanted:
            raise ValueError(f"manifest {seed} {key}={manifest.get(key)!r}, expected {wanted!r}")
    if manifest.get("source_objects") != ["airplane"]:
        raise ValueError(f"manifest {seed} has unexpected source_objects")
    if manifest.get("intervention_axes") != [2]:
        raise ValueError(f"manifest {seed} is not wrist-z only")


def load_rows(seed: int) -> Dict[str, torch.Tensor | object]:
    """Load and validate one existing collector payload, selecting valid rows."""
    if seed not in RUNS:
        raise ValueError(f"unknown seed {seed}")
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
    if payload.get("run_status") != "COMPLETED" or payload.get("followup_horizon") != FOLLOWUP_HORIZON:
        raise ValueError(f"payload status/horizon drift for seed {seed}")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"missing records for seed {seed}")
    required = (
        "q", "dof_vel", "object_state", "base_action", "executed_action",
        "assignment", "pre_contact", "progress", "next_q",
        "next_object_state", "next_contact", "followup_progress",
        "followup_reset", "final_lift_success", "final_max_contact_lift_m",
        "final_contact_fraction", "final_episode_steps", "env_id", "motion_id",
        "start_frame",
    )
    if any(name not in records for name in required):
        missing = [name for name in required if name not in records]
        raise ValueError(f"seed {seed} missing fields {missing}")
    n = N_ENV
    for name in required:
        value = records[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"seed {seed} {name} is not a tensor")
        if not _finite(value):
            raise FloatingPointError(f"seed {seed} non-finite {name}")
    for name in ("q", "dof_vel", "base_action", "executed_action", "next_q"):
        _require_shape(records, name, (n, 18))
    for name in ("object_state", "next_object_state"):
        _require_shape(records, name, (n, 13))
    for name in (
        "assignment", "pre_contact", "progress", "next_contact",
        "followup_progress", "followup_reset", "final_lift_success",
        "final_max_contact_lift_m", "final_contact_fraction", "final_episode_steps",
        "env_id", "motion_id", "start_frame",
    ):
        _require_shape(records, name, (n,))
    assignment = records["assignment"].to(torch.int64)
    valid = records["pre_contact"].bool() & assignment.ne(0)
    valid &= records["followup_reset"].eq(0)
    valid &= records["followup_progress"].eq(records["progress"] + FOLLOWUP_HORIZON)
    valid &= records["final_episode_steps"].gt(records["followup_progress"])
    if int(valid.sum()) == 0:
        raise ValueError(f"seed {seed} has no valid rows")
    delta = records["executed_action"] - records["base_action"]
    expected_delta = assignment.to(delta.dtype) * 0.1
    if not torch.allclose(delta[valid, 2], expected_delta[valid], atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} wrist-z dose does not match assignment")
    if not torch.allclose(delta[valid, :2], torch.zeros_like(delta[valid, :2]), atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} non-wrist action delta")
    if not torch.allclose(delta[valid, 3:], torch.zeros_like(delta[valid, 3:]), atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} non-wrist action delta")
    if records["motion_id"].unique().tolist() != [0]:
        raise ValueError(f"seed {seed} motion_id is not the pinned recorded motion")
    # Keep only the valid first-episode contact rows.  The post block is
    # observed immediately after the randomized action and contains no future
    # follow-up/contact-count fields.
    take = valid
    state = torch.cat((records["q"], records["dof_vel"], records["object_state"]), dim=1)[take].float()
    base = records["base_action"][take].float()
    treatment = delta[take].float()
    post = torch.cat((
        records["next_q"][take] - records["q"][take],
        records["next_object_state"][take] - records["object_state"][take],
        records["next_contact"][take].float().unsqueeze(1),
    ), dim=1)
    output: Dict[str, torch.Tensor | object] = {
        "state": state,
        "base_action": base,
        "treatment_delta": treatment,
        "post_handflow": post,
        "held_lift": records["final_lift_success"][take].float(),
        "max_contact_lift_mm": records["final_max_contact_lift_m"][take].float() * 1000.0,
        "contact_fraction": records["final_contact_fraction"][take].float(),
        "assignment": assignment[take],
        "env_id": records["env_id"][take].long(),
        "motion_id": records["motion_id"][take].long(),
        "start_frame": records["start_frame"][take].long(),
        "seed": seed,
        "split": spec["split"],
        "valid_rows": int(take.sum()),
        "manifest_sha256": spec["manifest_sha256"],
        "transition_sha256": spec["transition_sha256"],
    }
    return output


def concatenate(rows: list[Mapping[str, object]]) -> Dict[str, torch.Tensor]:
    keys = ("state", "base_action", "treatment_delta", "post_handflow",
            "held_lift", "max_contact_lift_mm", "contact_fraction", "assignment",
            "env_id", "motion_id", "start_frame")
    return {key: torch.cat([row[key] for row in rows]) for key in keys}  # type: ignore[arg-type]


def feature_blocks(rows: Mapping[str, torch.Tensor], model: str,
                   *, training: bool, permutation: torch.Tensor | None = None) -> torch.Tensor:
    state = rows["state"].float()
    base = rows["base_action"].float()
    treatment = rows["treatment_delta"].float()
    post = rows["post_handflow"].float()
    if model == "state_only":
        return state
    if model in ("action_aware", "action_shuffled"):
        if model == "action_shuffled" and training:
            if permutation is None:
                raise ValueError("action shuffle permutation required")
            treatment = treatment[permutation]
        return torch.cat((state, base, treatment), dim=1)
    if model in ("post_handflow", "post_handflow_shuffled"):
        if model == "post_handflow_shuffled" and training:
            if permutation is None:
                raise ValueError("post shuffle permutation required")
            treatment = treatment[permutation]
            post = post[permutation]
        return torch.cat((state, base, treatment, post), dim=1)
    raise ValueError(f"unknown model {model}")


def fit_ridge(features: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    x = features.double()
    y = target.double()
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - mean) / scale
    z = torch.cat((z, torch.ones((len(z), 1), dtype=z.dtype)), dim=1)
    penalty = torch.eye(z.shape[1], dtype=z.dtype) * RIDGE_LAMBDA
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict(fitted: Mapping[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    x = features.double()
    z = (x - fitted["mean"]) / fitted["scale"]
    z = torch.cat((z, torch.ones((len(z), 1), dtype=z.dtype)), dim=1)
    return z @ fitted["weights"]


def _rank_auc(score: torch.Tensor, target: torch.Tensor) -> float:
    target = target.to(torch.int64)
    positives = target.eq(1)
    negatives = target.eq(0)
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(prediction: torch.Tensor, target: torch.Tensor,
            *, binary: bool) -> Dict[str, float]:
    target = target.double()
    if binary:
        probability = prediction.sigmoid()
        return {
            "brier": float(((probability - target) ** 2).mean()),
            "rmse": float(((probability - target) ** 2).mean().sqrt()),
            "auroc": _rank_auc(probability, target),
            "mean_prediction": float(probability.mean()),
            "mean_target": float(target.mean()),
        }
    return {
        "rmse": float(((prediction - target) ** 2).mean().sqrt()),
        "mae": float((prediction - target).abs().mean()),
        "correlation": float(torch.corrcoef(torch.stack((prediction, target)))[0, 1]),
        "mean_prediction": float(prediction.mean()),
        "mean_target": float(target.mean()),
    }


def fit_probe(fit_rows: Mapping[str, torch.Tensor], holdout_rows: Mapping[str, torch.Tensor],
              *, model_artifact: Path) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(SHUFFLE_SEED)
    permutation = torch.randperm(len(fit_rows["held_lift"]), generator=generator)
    fitted: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    report_models: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in MODEL_NAMES:
        fitted[name] = {}
        report_models[name] = {}
        train_x = feature_blocks(fit_rows, name, training=True, permutation=permutation)
        test_x = feature_blocks(holdout_rows, name, training=False)
        for target_name in TARGET_NAMES:
            train_y = fit_rows[target_name]
            model = fit_ridge(train_x, train_y)
            test_prediction = predict(model, test_x)
            fitted[name][target_name] = model
            report_models[name][target_name] = metrics(
                test_prediction, holdout_rows[target_name], binary=(target_name == "held_lift"))
    model_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "ref2dex.hf03_contact_credit_cpu_models.v1",
        "ridge_lambda": RIDGE_LAMBDA,
        "shuffle_seed": SHUFFLE_SEED,
        "models": fitted,
    }, model_artifact)

    post = report_models["post_handflow"]
    controls = {name: report_models[name] for name in ("state_only", "action_aware", "action_shuffled", "post_handflow_shuffled")}
    def relative_improvement(control: str, target: str, metric: str) -> float:
        base = controls[control][target][metric]
        value = post[target][metric]
        return float((base - value) / max(abs(base), 1e-12))

    improvements = {
        control: {
            target: {
                metric: relative_improvement(control, target, metric)
                for metric in (("brier", "rmse") if target == "held_lift" else ("rmse",))
            }
            for target in ("held_lift", "max_contact_lift_mm")
        }
        for control in controls
    }
    fit_counts = {
        "minus": int((fit_rows["assignment"] == -1).sum()),
        "plus": int((fit_rows["assignment"] == 1).sum()),
    }
    holdout_counts = {
        "minus": int((holdout_rows["assignment"] == -1).sum()),
        "plus": int((holdout_rows["assignment"] == 1).sum()),
    }
    rows_pass = min(*fit_counts.values(), *holdout_counts.values()) >= 25
    finite_pass = all(
        all(bool(torch.isfinite(value).all()) for value in fitted[name][target].values())
        for name in MODEL_NAMES for target in TARGET_NAMES
    ) and all(
        all(math.isfinite(float(metric))
            for metric in report_models[name][target].values())
        for name in MODEL_NAMES for target in TARGET_NAMES
    )
    # The post-action representation must beat the strongest pre-action
    # control and its permutation placebo on both primary outcomes.  A 5%
    # margin is predeclared; no seed/regularizer/target rescan is allowed.
    primary_pass = all(
        improvements[control][target][metric] >= 0.05
        for control in ("action_aware", "post_handflow_shuffled")
        for target, metric in (("held_lift", "brier"), ("max_contact_lift_mm", "rmse"))
    )
    gate = {
        "fit_rows": int(len(fit_rows["held_lift"])),
        "holdout_rows": int(len(holdout_rows["held_lift"])),
        "fit_arm_counts": fit_counts,
        "holdout_arm_counts": holdout_counts,
        "rows_pass": rows_pass,
        "finite_pass": finite_pass,
        "primary_improvement_threshold": 0.05,
        "primary_improvement_pass": primary_pass,
        "all_conditions_pass": bool(rows_pass and finite_pass and primary_pass),
    }
    decision = "PROMISING" if gate["all_conditions_pass"] else (
        "UNPROMISING" if rows_pass and finite_pass else "UNCLEAR")
    return {
        "schema": "ref2dex.hf03_contact_supported_credit_cpu_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": report_models,
        "relative_improvements": improvements,
        "gate": gate,
        "decision": decision,
        "fit_spec": {
            "fit_seeds": [246, 247],
            "holdout_seeds": [248, 249],
            "normalization": "fit split mean and population std only",
            "ridge_lambda": RIDGE_LAMBDA,
            "binary_link": "sigmoid after ridge",
            "post_handflow": "next_q-q, next_object_state-object_state, next_contact",
            "action_shuffled": "fit treatment delta permuted once with fixed seed",
            "post_handflow_shuffled": "fit treatment delta and post block jointly permuted once",
            "primary_targets": ["held_lift", "max_contact_lift_mm"],
            "gate": "post handflow improves held-lift Brier and max-contact-lift RMSE by >=5% over action-aware and post-handflow-shuffled controls",
        },
        "model_artifact": str(model_artifact),
        "model_artifact_sha256": sha256(model_artifact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if torch.cuda.is_available():
        raise RuntimeError("HF03 CPU audit must run with CUDA unavailable")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "schema": "ref2dex.hf03_contact_supported_credit_cpu_audit_manifest.v1",
        "run_id": args.output.name,
        "run_status": "STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "fit_seeds": [246, 247],
        "holdout_seeds": [248, 249],
        "input_sha256": {str(seed): RUNS[seed]["transition_sha256"] for seed in RUNS},
        "source_manifest_sha256": {str(seed): RUNS[seed]["manifest_sha256"] for seed in RUNS},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
        "resource": {"gpu_count": 0, "cpu_threads": 2, "output_budget_mb": 20},
        "stop_rule": "input/provenance drift, incomplete first episode, nonfinite fit or output",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        fit_parts = [load_rows(246), load_rows(247)]
        holdout_parts = [load_rows(248), load_rows(249)]
        fit = concatenate(fit_parts)
        holdout = concatenate(holdout_parts)
        report = fit_probe(fit, holdout, model_artifact=args.output / "cpu_models.pt")
        report.update({
            "git_commit": manifest["git_commit"],
            "script_sha256": manifest["script_sha256"],
            "input_sha256": manifest["input_sha256"],
            "manifest_sha256": manifest["source_manifest_sha256"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
            "cuda_available": False,
            "resource": manifest["resource"],
            "source_rows": [
                {
                    "seed": int(row["seed"]),
                    "split": row["split"],
                    "valid_rows": int(row["valid_rows"]),
                    "assignment_seed": RUNS[int(row["seed"])]
                    ["assignment_seed"],
                    "arm_counts": {
                        "minus": int((row["assignment"] == -1).sum()),
                        "plus": int((row["assignment"] == 1).sum()),
                    },
                    "motion_ids": row["motion_id"].unique().tolist(),
                    "transition_sha256": row["transition_sha256"],
                    "manifest_sha256": row["manifest_sha256"],
                }
                for row in fit_parts + holdout_parts
            ],
            "data_scope": "single recorded motion_id=0 on the airplane source-e260 substrate; no cross-object or multi-motion claim",
        })
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_digest = sha256(report_path)
        manifest.update({
            "run_status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "report_sha256": report_digest,
            "model_artifact_sha256": report["model_artifact_sha256"],
            "decision": report["decision"],
            "gate": report["gate"],
        })
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"decision": report["decision"], "gate": report["gate"],
                          "report": str(report_path), "report_sha256": report_digest},
                         sort_keys=True))
    except BaseException as error:
        manifest.update({
            "run_status": "FAILED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "failure": f"{type(error).__name__}: {error}",
        })
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
