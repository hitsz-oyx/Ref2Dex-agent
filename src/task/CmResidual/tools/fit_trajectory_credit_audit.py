"""CPU screen for a short trajectory-level Cm credit signal.

HF04 is a new higher-level hypothesis: after a randomized contact-stage
action has executed, a five-step contact/handflow trajectory token may improve
prediction of the later first-episode held-lift outcome.  The token is a
post-action credit signal; it is not used to choose the initial action.  This
script only consumes four existing three-arm Cm-off collections and never
imports Isaac Gym or starts CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Dict, Mapping

import torch

ROOT = Path(__file__).resolve().parents[4]
N_ENV = 64
FOLLOWUP_HORIZON = 5
RIDGE_LAMBDA = 10.0
SHUFFLE_SEED = 2026092604
CHECKPOINT_SHA256 = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
MOTION_MANIFEST_SHA256 = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"

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
MODEL_NAMES = ("pre_action", "trajectory_credit", "action_shuffled", "trajectory_shuffled")
TARGET_NAMES = ("held_lift", "max_contact_lift_mm", "contact_fraction")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.float()).all())


def _shape(records: Mapping[str, torch.Tensor], name: str, shape: tuple[int, ...]) -> None:
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
            raise ValueError(f"manifest {seed} {key}={manifest.get(key)!r}, expected {wanted!r}")
    if manifest.get("source_objects") != ["airplane"] or manifest.get("intervention_axes") != [2]:
        raise ValueError(f"manifest {seed} substrate/action drift")


def load_rows(seed: int) -> Dict[str, object]:
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
        raise ValueError(f"missing records for seed {seed}")
    required = (
        "q", "dof_vel", "object_state", "base_action", "executed_action",
        "assignment", "intervention_valid", "pre_contact", "progress",
        "next_q", "next_object_state", "next_contact", "followup_contact_count",
        "followup_object_state", "followup_contact", "followup_progress",
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
    n = N_ENV
    for name in ("q", "dof_vel", "base_action", "executed_action", "next_q"):
        _shape(records, name, (n, 18))
    for name in ("object_state", "next_object_state", "followup_object_state"):
        _shape(records, name, (n, 13))
    for name in (
        "assignment", "intervention_valid", "pre_contact", "progress", "next_contact",
        "followup_contact_count", "followup_contact", "followup_progress",
        "followup_reset", "final_lift_success", "final_max_contact_lift_m",
        "final_contact_fraction", "final_episode_steps", "env_id", "motion_id",
        "start_frame",
    ):
        _shape(records, name, (n,))
    assignment = records["assignment"].to(torch.int64)
    if set(assignment.unique().tolist()) - {-1, 0, 1}:
        raise ValueError(f"seed {seed} has non-three-arm assignment")
    valid = records["pre_contact"].bool() & records["intervention_valid"].bool()
    valid &= records["followup_reset"].eq(0)
    valid &= records["followup_progress"].eq(records["progress"] + FOLLOWUP_HORIZON)
    valid &= records["final_episode_steps"].gt(records["followup_progress"])
    if int(valid.sum()) < 30:
        raise ValueError(f"seed {seed} has too few valid trajectory rows")
    delta = records["executed_action"] - records["base_action"]
    expected = assignment.to(delta.dtype) * 0.1
    if not torch.allclose(delta[valid, 2], expected[valid], atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} wrist-z dose mismatch")
    if not torch.allclose(delta[valid, :2], torch.zeros_like(delta[valid, :2]), atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} non-wrist dose")
    if not torch.allclose(delta[valid, 3:], torch.zeros_like(delta[valid, 3:]), atol=1e-6, rtol=0):
        raise ValueError(f"seed {seed} non-wrist dose")
    if records["motion_id"].unique().tolist() != [0]:
        raise ValueError(f"seed {seed} motion-id drift")
    state = torch.cat((records["q"], records["dof_vel"], records["object_state"]), 1)[valid].float()
    base = records["base_action"][valid].float()
    treatment = delta[valid].float()
    trajectory = torch.cat((
        records["next_q"][valid] - records["q"][valid],
        records["next_object_state"][valid] - records["object_state"][valid],
        records["next_contact"][valid].float().unsqueeze(1),
        records["followup_object_state"][valid] - records["object_state"][valid],
        (records["followup_contact_count"][valid].float() / FOLLOWUP_HORIZON).unsqueeze(1),
        records["followup_contact"][valid].float().unsqueeze(1),
    ), 1)
    return {
        "state": state, "base_action": base, "treatment": treatment,
        "trajectory": trajectory,
        "held_lift": records["final_lift_success"][valid].float(),
        "max_contact_lift_mm": records["final_max_contact_lift_m"][valid].float() * 1000,
        "contact_fraction": records["final_contact_fraction"][valid].float(),
        "assignment": assignment[valid], "env_id": records["env_id"][valid].long(),
        "motion_id": records["motion_id"][valid].long(),
        "start_frame": records["start_frame"][valid].long(),
        "seed": seed, "split": spec["split"], "valid_rows": int(valid.sum()),
        "transition_sha256": spec["transition_sha256"],
        "manifest_sha256": spec["manifest_sha256"],
    }


def concatenate(rows: list[Mapping[str, object]]) -> Dict[str, torch.Tensor]:
    keys = ("state", "base_action", "treatment", "trajectory", "held_lift",
            "max_contact_lift_mm", "contact_fraction", "assignment", "env_id",
            "motion_id", "start_frame")
    return {key: torch.cat([row[key] for row in rows]) for key in keys}  # type: ignore[arg-type]


def features(rows: Mapping[str, torch.Tensor], model: str, *, training: bool,
             permutation: torch.Tensor | None = None) -> torch.Tensor:
    state = rows["state"].float()
    base = rows["base_action"].float()
    treatment = rows["treatment"].float()
    trajectory = rows["trajectory"].float()
    if model == "pre_action":
        return torch.cat((state, base, treatment), 1)
    if model == "action_shuffled":
        if training:
            if permutation is None:
                raise ValueError("missing action permutation")
            treatment = treatment[permutation]
        return torch.cat((state, base, treatment), 1)
    if model == "trajectory_credit":
        return torch.cat((state, base, treatment, trajectory), 1)
    if model == "trajectory_shuffled":
        if training:
            if permutation is None:
                raise ValueError("missing trajectory permutation")
            treatment = treatment[permutation]
            trajectory = trajectory[permutation]
        return torch.cat((state, base, treatment, trajectory), 1)
    raise ValueError(model)


def fit_ridge(x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    x, y = x.double(), y.double()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    z = torch.cat(((x - mean) / scale, torch.ones((len(x), 1), dtype=torch.float64)), 1)
    penalty = torch.eye(z.shape[1], dtype=torch.float64) * RIDGE_LAMBDA
    penalty[-1, -1] = 0
    return {"mean": mean, "scale": scale,
            "weights": torch.linalg.solve(z.T @ z + penalty, z.T @ y)}


def predict(model: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    z = (x.double() - model["mean"]) / model["scale"]
    z = torch.cat((z, torch.ones((len(z), 1), dtype=torch.float64)), 1)
    return z @ model["weights"]


def _auc(score: torch.Tensor, target: torch.Tensor) -> float:
    target = target.to(torch.int64)
    pos, neg = target.eq(1), target.eq(0)
    npos, nneg = int(pos.sum()), int(neg.sum())
    if not npos or not nneg:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metric(prediction: torch.Tensor, target: torch.Tensor, binary: bool) -> Dict[str, float]:
    target = target.double()
    if binary:
        probability = prediction.sigmoid()
        return {
            "brier": float(((probability - target) ** 2).mean()),
            "rmse": float(((probability - target) ** 2).mean().sqrt()),
            "auroc": _auc(probability, target),
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
              artifact: Path) -> Dict[str, object]:
    permutation = torch.randperm(
        len(fit_rows["held_lift"]), generator=torch.Generator().manual_seed(SHUFFLE_SEED))
    fitted: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    report_models: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in MODEL_NAMES:
        fitted[name], report_models[name] = {}, {}
        train_x = features(fit_rows, name, training=True, permutation=permutation)
        test_x = features(holdout_rows, name, training=False)
        for target in TARGET_NAMES:
            fitted[name][target] = fit_ridge(train_x, fit_rows[target])
            report_models[name][target] = metric(
                predict(fitted[name][target], test_x), holdout_rows[target], target == "held_lift")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "ref2dex.hf04_trajectory_credit_cpu_models.v1",
                "ridge_lambda": RIDGE_LAMBDA, "shuffle_seed": SHUFFLE_SEED,
                "models": fitted}, artifact)
    def improvement(target: str, metric_name: str, control: str) -> float:
        base = report_models[control][target][metric_name]
        value = report_models["trajectory_credit"][target][metric_name]
        return float((base - value) / max(abs(base), 1e-12))
    relative = {
        control: {
            "held_lift_brier": improvement("held_lift", "brier", control),
            "max_contact_lift_rmse": improvement("max_contact_lift_mm", "rmse", control),
        }
        for control in ("pre_action", "action_shuffled", "trajectory_shuffled")
    }
    fit_counts = {str(a): int((fit_rows["assignment"] == a).sum()) for a in (-1, 0, 1)}
    holdout_counts = {str(a): int((holdout_rows["assignment"] == a).sum()) for a in (-1, 0, 1)}
    rows_pass = min(*fit_counts.values(), *holdout_counts.values()) >= 15
    finite_pass = all(
        all(bool(torch.isfinite(value).all()) for value in fitted[name][target].values())
        for name in MODEL_NAMES for target in TARGET_NAMES
    ) and all(
        all(math.isfinite(float(value)) for value in report_models[name][target].values())
        for name in MODEL_NAMES for target in TARGET_NAMES
    )
    primary_pass = all(
        relative[control][metric_name] >= 0.05
        for control in ("pre_action", "trajectory_shuffled")
        for metric_name in ("held_lift_brier", "max_contact_lift_rmse")
    )
    gate = {
        "fit_rows": int(len(fit_rows["held_lift"])),
        "holdout_rows": int(len(holdout_rows["held_lift"])),
        "fit_arm_counts": fit_counts, "holdout_arm_counts": holdout_counts,
        "minimum_rows_per_arm": 15, "relative_margin": 0.05,
        "rows_pass": rows_pass, "finite_pass": finite_pass,
        "primary_improvement_pass": primary_pass,
        "all_conditions_pass": bool(rows_pass and finite_pass and primary_pass),
    }
    decision = "PROMISING" if gate["all_conditions_pass"] else (
        "UNPROMISING" if rows_pass and finite_pass else "UNCLEAR")
    return {
        "schema": "ref2dex.hf04_trajectory_credit_cpu_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": report_models, "relative_improvements": relative,
        "gate": gate, "decision": decision,
        "fit_spec": {
            "fit_seeds": [250, 251], "holdout_seeds": [252, 253],
            "assignment_propensity": "1/3 for each arm -1,0,+1",
            "post_option_token": "next state delta + five-step followup object delta/contact count/contact",
            "normalization": "fit split mean and population std only",
            "ridge_lambda": RIDGE_LAMBDA, "binary_link": "sigmoid after ridge",
            "gate": "trajectory_credit improves held-lift Brier and max-contact-lift RMSE by >=5% over pre_action and trajectory_shuffled",
            "scope": "post-action credit prediction only; not an initial-action policy selector",
        },
        "model_artifact": str(artifact), "model_artifact_sha256": sha256(artifact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if torch.cuda.is_available():
        raise RuntimeError("HF04 audit must run CPU-only")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "schema": "ref2dex.hf04_trajectory_credit_cpu_audit_manifest.v1",
        "run_id": args.output.name, "run_status": "STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "fit_seeds": [250, 251], "holdout_seeds": [252, 253],
        "input_sha256": {str(seed): RUNS[seed]["transition_sha256"] for seed in RUNS},
        "source_manifest_sha256": {str(seed): RUNS[seed]["manifest_sha256"] for seed in RUNS},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
        "resource": {"gpu_count": 0, "cpu_threads": 2, "output_budget_mb": 20},
        "stop_rule": "provenance drift, incomplete first episode, nonfinite fit or output",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        fit_parts = [load_rows(250), load_rows(251)]
        holdout_parts = [load_rows(252), load_rows(253)]
        report = fit_probe(concatenate(fit_parts), concatenate(holdout_parts), args.output / "cpu_models.pt")
        report.update({
            "git_commit": manifest["git_commit"], "script_sha256": manifest["script_sha256"],
            "input_sha256": manifest["input_sha256"], "manifest_sha256": manifest["source_manifest_sha256"],
            "checkpoint_sha256": CHECKPOINT_SHA256, "motion_manifest_sha256": MOTION_MANIFEST_SHA256,
            "cuda_available": False, "resource": manifest["resource"],
            "data_scope": "single recorded motion_id=0 on airplane source-e260; no cross-object/multi-motion claim",
            "source_rows": [
                {"seed": int(row["seed"]), "split": row["split"], "valid_rows": int(row["valid_rows"]),
                 "assignment_seed": RUNS[int(row["seed"])] ["assignment_seed"],
                 "arm_counts": {str(a): int((row["assignment"] == a).sum()) for a in (-1, 0, 1)},
                 "motion_ids": row["motion_id"].unique().tolist(),
                 "transition_sha256": row["transition_sha256"], "manifest_sha256": row["manifest_sha256"]}
                for row in fit_parts + holdout_parts
            ],
        })
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_digest = sha256(report_path)
        manifest.update({"run_status": "COMPLETED", "completed_at": datetime.now(timezone.utc).isoformat(),
                         "report_sha256": report_digest, "model_artifact_sha256": report["model_artifact_sha256"],
                         "decision": report["decision"], "gate": report["gate"]})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"decision": report["decision"], "gate": report["gate"],
                          "report": str(report_path), "report_sha256": report_digest}, sort_keys=True))
    except BaseException as error:
        manifest.update({"run_status": "FAILED", "completed_at": datetime.now(timezone.utc).isoformat(),
                         "failure": f"{type(error).__name__}: {error}"})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
