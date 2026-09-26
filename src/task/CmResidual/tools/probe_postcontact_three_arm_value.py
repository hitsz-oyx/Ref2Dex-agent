"""Fit and evaluate a three-arm post-contact held-lift value probe.

The input files are produced by ``--three-arm-randomized`` and contain base,
minus-z, and plus-z outcomes from one simulator batch.  The probe is kept
separate from the two-arm analysis because the base arm has a third treatment
propensity and must be handled by a multi-arm Hájek estimator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "ref2dex.randomized_action_followup.v1"
ARMS = (-1, 0, 1)
VARIANTS = ("state_only", "cm_aware", "action_shuffled")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _design(state: np.ndarray, treatment: np.ndarray, variant: str) -> np.ndarray:
    if variant == "state_only":
        return state
    treatment = treatment.astype(np.int64)
    one_hot = np.stack([treatment == arm for arm in ARMS], axis=1).astype(np.float32)
    if variant in ("cm_aware", "action_shuffled"):
        return np.concatenate((state, one_hot,
                               np.concatenate([state * one_hot[:, i:i + 1]
                                               for i in range(len(ARMS))], axis=1)), axis=1)
    raise ValueError(f"unknown variant {variant}")


def _load(path: Path, source_id: int) -> dict[str, np.ndarray | str | int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("run_status") != "COMPLETED":
        raise ValueError(f"{path}: incomplete or unexpected schema")
    if not payload.get("record_final_outcome") or not payload.get("three_arm_randomized"):
        raise ValueError(f"{path}: file is not a complete three-arm run")
    records = payload["records"]
    required = {"q", "dof_vel", "object_state", "base_action", "executed_action",
                "assignment", "pre_contact", "intervention_valid", "env_id",
                "final_lift_success", "final_max_contact_lift_m",
                "final_contact_fraction", "progress", "start_frame"}
    missing = sorted(required.difference(records))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    lengths = {key: int(records[key].shape[0]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path}: inconsistent record lengths {lengths}")
    assignment = records["assignment"].reshape(-1).numpy().astype(np.int8)
    contact = records["pre_contact"].reshape(-1).numpy().astype(bool)
    valid = records["intervention_valid"].reshape(-1).numpy().astype(bool)
    selected = contact & valid
    if selected.sum() < 30 or not np.isin(assignment[selected], ARMS).all():
        raise ValueError(f"{path}: too few or invalid eligible rows")
    counts = {arm: int(((assignment == arm) & selected).sum()) for arm in ARMS}
    if min(counts.values()) < 5:
        raise ValueError(f"{path}: sparse arm counts {counts}")
    q = records["q"].float()
    object_state = records["object_state"].float()
    q_relative = q.clone()
    q_relative[:, :3] -= object_state[:, :3]
    state = torch.cat((q_relative, records["dof_vel"].float(), object_state,
                       records["base_action"].float(),
                       records["progress"].float().reshape(-1, 1) / 500.0,
                       records["start_frame"].float().reshape(-1, 1) / 500.0), dim=1)
    expected = assignment.astype(np.float32) * float(payload["delta_z_action"])
    actual = (records["executed_action"][:, 2] - records["base_action"][:, 2]).numpy()
    if not np.allclose(actual[selected], expected[selected], atol=1e-5, rtol=0):
        raise ValueError(f"{path}: executed action mismatch")
    result = {
        "state": state.numpy()[selected].astype(np.float32),
        "assignment": assignment[selected].astype(np.int64),
        "final_lift_success": records["final_lift_success"].reshape(-1).numpy()[selected].astype(np.float32),
        "final_max_contact_lift_m": records["final_max_contact_lift_m"].reshape(-1).numpy()[selected].astype(np.float32),
        "final_contact_fraction": records["final_contact_fraction"].reshape(-1).numpy()[selected].astype(np.float32),
        "source_id": source_id,
        "path": str(path), "sha256": sha256(path), "counts": counts,
        "propensities": {arm: counts[arm] / float(sum(counts.values())) for arm in ARMS},
    }
    if any(not np.isfinite(value).all() for value in result.values()
           if isinstance(value, np.ndarray)):
        raise FloatingPointError(f"{path}: non-finite row")
    return result


def _pool(parts):
    keys = ("state", "assignment", "final_lift_success",
            "final_max_contact_lift_m")
    result = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    result["source_id"] = np.concatenate([
        np.full(len(part["assignment"]), int(part["source_id"]), dtype=np.int64)
        for part in parts])
    result["propensities"] = [part["propensities"] for part in parts]
    return result


def _multi_arm_value(y, assignment, policy, source, propensities):
    weights = np.zeros(len(y), dtype=np.float64)
    matched = assignment == policy
    for index, source_id in enumerate(np.unique(source)):
        for arm in ARMS:
            rows = matched & (source == source_id) & (assignment == arm)
            weights[rows] = 1.0 / propensities[index][arm]
    if not matched.any():
        return float("nan"), 0
    return float((weights * y).sum() / weights.sum()), int(matched.sum())


def _bootstrap(y, assignment, policy, source, propensities, seed, draws=1000):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(source == value) for value in np.unique(source)]
    values = []
    for _ in range(draws):
        indices = np.concatenate([
            group[rng.integers(0, len(group), size=len(group))]
            for group in groups])
        value, _ = _multi_arm_value(y[indices], assignment[indices], policy[indices],
                                    source[indices], propensities)
        if np.isfinite(value):
            values.append(value)
    return np.quantile(values, [.025, .975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", action="append", required=True, type=Path)
    parser.add_argument("--test-input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.output.exists() or len(args.train_input) < 2 or len(args.test_input) < 2:
        parser.error("new output and at least two train/test files are required")
    args.output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "train_input": [str(path) for path in args.train_input],
                "test_input": [str(path) for path in args.test_input],
                "fit_seed": args.fit_seed, "wall_budget_minutes": 20}
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        train_parts = [_load(path, index) for index, path in enumerate(args.train_input)]
        test_parts = [_load(path, 100 + index)
                      for index, path in enumerate(args.test_input)]
        train, test = _pool(train_parts), _pool(test_parts)
        scaler = StandardScaler().fit(train["state"])
        pca = PCA(n_components=min(24, len(train["state"]) - 2,
                                  train["state"].shape[1]), random_state=args.fit_seed)
        x_train_state = pca.fit_transform(scaler.transform(train["state"]))
        x_test_state = pca.transform(scaler.transform(test["state"]))
        rng = np.random.default_rng(args.fit_seed + 1)
        shuffled_parts = [part["assignment"][rng.permutation(len(part["assignment"]))]
                          for part in train_parts]
        shuffled = np.concatenate(shuffled_parts)
        reports, models = {}, {}
        y_train = train["final_lift_success"]
        y_test = test["final_lift_success"]
        for variant in VARIANTS:
            fit_assignment = shuffled if variant == "action_shuffled" else train["assignment"]
            classifier = LogisticRegression(C=.1, solver="liblinear", max_iter=2000,
                                            random_state=args.fit_seed)
            classifier.fit(_design(x_train_state, fit_assignment, variant), y_train)
            factual = classifier.predict_proba(
                _design(x_test_state, test["assignment"], variant))[:, 1]
            candidates = np.stack([
                classifier.predict_proba(_design(x_test_state,
                                                 np.full(len(x_test_state), arm), variant))[:, 1]
                for arm in ARMS], axis=1)
            if variant == "state_only":
                means = {arm: float(y_train[train["assignment"] == arm].mean()) for arm in ARMS}
                best = max(ARMS, key=means.get)
                policy = np.full(len(y_test), best, dtype=np.int64)
            else:
                policy = np.asarray(ARMS)[candidates.argmax(axis=1)]
            value, matched = _multi_arm_value(
                y_test, test["assignment"], policy, test["source_id"],
                [part["propensities"] for part in test_parts])
            reports[variant] = {
                "brier": float(brier_score_loss(y_test, factual)),
                "auroc": float(roc_auc_score(y_test, factual)),
                "policy_value": value, "matched_rows": matched,
                "choice_counts": {str(arm): int((policy == arm).sum()) for arm in ARMS},
                "cluster_95ci": _bootstrap(
                    y_test, test["assignment"], policy, test["source_id"],
                    [part["propensities"] for part in test_parts],
                    args.fit_seed + 20 + list(VARIANTS).index(variant)),
            }
            models[variant] = classifier
        aware, blind, shuffled_value = (reports[name]["policy_value"]
                                        for name in ("cm_aware", "state_only",
                                                     "action_shuffled"))
        gate = {"aware_above_state_only": bool(aware > blind),
                "aware_above_shuffled": bool(aware > shuffled_value),
                "minimum_gain_5pp": bool(aware >= max(blind, shuffled_value) + .05)}
        gate["prespecified_gate_passed"] = bool(all(gate.values()))
        raw = {str(arm): {"train": float(y_train[train["assignment"] == arm].mean()),
                          "test": float(y_test[test["assignment"] == arm].mean())}
               for arm in ARMS}
        report = {"schema": "ref2dex.postcontact_three_arm_value_probe.v1",
                  "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
                  "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
                  "train_files": [{"path": part["path"], "sha256": part["sha256"],
                                   "counts": part["counts"]} for part in train_parts],
                  "test_files": [{"path": part["path"], "sha256": part["sha256"],
                                  "counts": part["counts"]} for part in test_parts],
                  "arm_means": raw, "models": reports,
                  "prespecified_gate": gate,
                  "limits": ["single airplane object and one contact time",
                             "randomized population value, not individual counterfactuals",
                             "a passing offline gate would still require frozen online Cm-on/off"],
                  "elapsed_seconds": time.monotonic() - started}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        joblib.dump({"state_scaler": scaler, "state_pca": pca, "models": models,
                     "arms": ARMS}, args.output / "models.joblib", compress=3)
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path),
                        model_sha256=sha256(args.output / "models.joblib"))
        print(json.dumps({"gate": gate, "models": reports, "arm_means": raw}, sort_keys=True))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
