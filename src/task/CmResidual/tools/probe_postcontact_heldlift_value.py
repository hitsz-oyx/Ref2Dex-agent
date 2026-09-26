"""Probe a post-contact Cm value head against complete held-lift outcomes.

The collector randomizes one wrist-z action after contact and then lets the
episode finish.  This tool deliberately keeps the randomization and the
evaluation split separate: models are fitted on complete episodes from the
training seeds and are evaluated on new simulator seeds from the same
airplane distribution.

This is an exploratory policy-utility probe.  It does not make individual
counterfactual claims.  Policy values on the randomized test rows use a
Hájek inverse-propensity estimate and are reported together with a fixed-arm
control and an action-shuffled control.
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "ref2dex.randomized_action_followup.v1"
TARGETS = ("final_lift_success", "final_max_contact_lift_m",
           "final_contact_fraction")
VARIANTS = ("state_only", "action_main", "cm_aware", "action_shuffled")
MIN_ROWS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _load_rows(path: Path, source_id: int) -> dict[str, np.ndarray | str | int]:
    """Load and audit one complete-episode randomized-action file."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {payload.get('schema')!r}")
    if payload.get("run_status") != "COMPLETED":
        raise ValueError(f"{path}: run is not completed")
    if not payload.get("record_final_outcome", False):
        raise ValueError(f"{path}: final outcome recording is disabled")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"{path}: records are missing")
    required = {
        "q", "dof_vel", "object_state", "base_action", "executed_action",
        "assignment", "pre_contact", "env_id", "progress", "start_frame",
        "final_lift_success", "final_max_contact_lift_m",
        "final_contact_fraction", "final_episode_steps",
    }
    missing = sorted(required.difference(records))
    if missing:
        raise ValueError(f"{path}: missing record fields {missing}")
    lengths = {key: int(records[key].shape[0]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path}: record lengths differ: {lengths}")
    assignment = _as_numpy(records["assignment"].reshape(-1)).astype(np.int8)
    pre_contact = _as_numpy(records["pre_contact"].reshape(-1)).astype(bool)
    env_id = _as_numpy(records["env_id"].reshape(-1)).astype(np.int64)
    if not np.isin(assignment, (-1, 0, 1)).all():
        raise ValueError(f"{path}: invalid assignment values")
    if np.any((assignment != 0) & ~pre_contact):
        raise ValueError(f"{path}: an intervention occurred before contact")
    selected = (assignment != 0) & pre_contact
    if int(selected.sum()) < MIN_ROWS:
        raise ValueError(f"{path}: only {int(selected.sum())} randomized rows")
    if len(np.unique(env_id[selected])) != int(selected.sum()):
        raise ValueError(f"{path}: expected one intervention per environment")

    q = records["q"].float()
    object_state = records["object_state"].float()
    q_relative = q.clone()
    q_relative[:, :3] -= object_state[:, :3]
    # These are all pre-action quantities.  In particular, next_* and the
    # follow-up fields are intentionally excluded to avoid outcome leakage.
    state = torch.cat((q_relative, records["dof_vel"].float(), object_state,
                       records["base_action"].float(),
                       records["progress"].float().reshape(-1, 1) / 500.0,
                       records["start_frame"].float().reshape(-1, 1) / 500.0), dim=1)
    action_delta = (records["executed_action"].float() -
                    records["base_action"].float())[:, 2]
    expected_delta = assignment.astype(np.float32) * float(payload["delta_z_action"])
    if not np.allclose(_as_numpy(action_delta)[selected], expected_delta[selected],
                       atol=1e-5, rtol=0.0):
        raise ValueError(f"{path}: executed action does not match assignment")
    arrays: dict[str, np.ndarray | str | int] = {
        "state": _as_numpy(state)[selected].astype(np.float32),
        "assignment": assignment[selected].astype(np.float32),
        "env_id": env_id[selected],
        "final_lift_success": _as_numpy(records["final_lift_success"].reshape(-1))[selected].astype(np.float32),
        "final_max_contact_lift_m": _as_numpy(records["final_max_contact_lift_m"].reshape(-1))[selected].astype(np.float32),
        "final_contact_fraction": _as_numpy(records["final_contact_fraction"].reshape(-1))[selected].astype(np.float32),
        "final_episode_steps": _as_numpy(records["final_episode_steps"].reshape(-1))[selected].astype(np.float32),
        "propensity_plus": float((assignment[selected] == 1).mean()),
        "source_id": source_id,
        "path": str(path),
        "sha256": sha256(path),
    }
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise FloatingPointError(f"{path}: non-finite {key}")
    return arrays


def _pool(parts: list[dict[str, np.ndarray | str | int]]) -> dict[str, np.ndarray]:
    keys = ("state", "assignment", "env_id", *TARGETS, "final_episode_steps")
    result = {key: np.concatenate([part[key] for part in parts], axis=0)
              for key in keys}
    result["source_id"] = np.concatenate([
        np.full(len(part["assignment"]), int(part["source_id"]), dtype=np.int64)
        for part in parts])
    result["propensity_plus"] = np.concatenate([
        np.full(len(part["assignment"]), float(part["propensity_plus"]), dtype=np.float64)
        for part in parts])
    return result


def _design(state: np.ndarray, treatment: np.ndarray, variant: str) -> np.ndarray:
    if variant == "state_only":
        return state
    z = treatment.reshape(-1, 1)
    if variant == "action_main":
        return np.concatenate((state, z), axis=1)
    if variant in ("cm_aware", "action_shuffled"):
        return np.concatenate((state, z, state * z), axis=1)
    raise ValueError(f"unknown model variant {variant}")


def _shuffle_treatment(parts: list[dict[str, np.ndarray | str | int]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = []
    for part in parts:
        values = np.asarray(part["assignment"], dtype=np.float32).copy()
        shuffled.append(values[rng.permutation(len(values))])
    return np.concatenate(shuffled)


def _factual_metrics(prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    result = {"brier": float(brier_score_loss(y, prob))}
    if len(np.unique(y)) > 1:
        result["auroc"] = float(roc_auc_score(y, prob))
    return result


def _hajek(y: np.ndarray, assignment: np.ndarray, policy: np.ndarray,
           propensity_plus: np.ndarray) -> tuple[float, int]:
    propensity = np.where(assignment == 1, propensity_plus, 1.0 - propensity_plus)
    matched = assignment == policy
    weights = matched.astype(np.float64) / np.clip(propensity, 1e-6, 1.0)
    if weights.sum() <= 0:
        return float("nan"), 0
    return float(np.sum(weights * y) / np.sum(weights)), int(matched.sum())


def _fixed_value(y: np.ndarray, assignment: np.ndarray, arm: int) -> float:
    selected = assignment == arm
    return float(y[selected].mean()) if selected.any() else float("nan")


def _bootstrap_policy(y: np.ndarray, assignment: np.ndarray, policy: np.ndarray,
                      propensity: np.ndarray, source: np.ndarray, *, seed: int,
                      draws: int = 1000) -> list[float]:
    rng = np.random.default_rng(seed)
    source_values = np.unique(source)
    groups = [np.flatnonzero(source == value) for value in source_values]
    values: list[float] = []
    for _ in range(draws):
        indices = np.concatenate([
            group[rng.integers(0, len(group), size=len(group))]
            for group in groups])
        value, _ = _hajek(y[indices], assignment[indices], policy[indices],
                          propensity[indices])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(100, int(draws * .8)):
        raise ValueError("too few valid policy bootstrap draws")
    return np.quantile(np.asarray(values), [.025, .975]).tolist()


def _rank_report(score: np.ndarray, y: np.ndarray, assignment: np.ndarray,
                 source: np.ndarray) -> dict[str, float | int | list[float]]:
    q25, q75 = np.quantile(score, [.25, .75])
    low, high = score <= q25, score >= q75

    def ate(mask: np.ndarray) -> float:
        plus, minus = mask & (assignment == 1), mask & (assignment == -1)
        if not plus.any() or not minus.any():
            return float("nan")
        return float(y[plus].mean() - y[minus].mean())

    low_ate, high_ate = ate(low), ate(high)
    rng = np.random.default_rng(20260925)
    groups = [np.flatnonzero(source == value) for value in np.unique(source)]
    draws = []
    for _ in range(1000):
        indices = np.concatenate([
            group[rng.integers(0, len(group), size=len(group))]
            for group in groups])
        lo, hi = ate(low[indices]), ate(high[indices])
        if np.isfinite(lo) and np.isfinite(hi):
            draws.append(hi - lo)
    result = {"q25": float(q25), "q75": float(q75),
              "low_count": int(low.sum()), "high_count": int(high.sum()),
              "ate_low": low_ate, "ate_high": high_ate,
              "high_minus_low": float(high_ate - low_ate)}
    if len(draws) >= 800:
        result["cluster_95ci"] = np.quantile(np.asarray(draws), [.025, .975]).tolist()
    else:
        result["cluster_95ci"] = [float("nan"), float("nan")]
    return result


def _fit_and_score(train: dict[str, np.ndarray], test: dict[str, np.ndarray],
                   train_parts: list[dict[str, np.ndarray | str | int]],
                   *, seed: int) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(train["state"])
    pca = PCA(n_components=min(24, train["state"].shape[0] - 2,
                              train["state"].shape[1]), random_state=seed)
    train_state = pca.fit_transform(scaler.transform(train["state"]))
    test_state = pca.transform(scaler.transform(test["state"]))
    shuffled = _shuffle_treatment(train_parts, seed + 1)
    models = {}
    reports = {}
    train_y = train["final_lift_success"]
    for variant in VARIANTS:
        fit_treatment = shuffled if variant == "action_shuffled" else train["assignment"]
        x_train = _design(train_state, fit_treatment, variant)
        classifier = LogisticRegression(C=0.1, solver="liblinear", max_iter=2000,
                                        random_state=seed)
        classifier.fit(x_train, train_y)
        x_test = _design(test_state, test["assignment"], variant)
        factual = classifier.predict_proba(x_test)[:, 1]
        plus = classifier.predict_proba(
            _design(test_state, np.ones(len(test_state)), variant))[:, 1]
        minus = classifier.predict_proba(
            _design(test_state, -np.ones(len(test_state)), variant))[:, 1]
        models[(variant, "binary")] = classifier
        reports[variant] = {"factual": _factual_metrics(factual,
                                                         test_y := test_y_global(test)),
                            "predicted_contrast_mean": float((plus - minus).mean())}
        reports[variant]["binary_plus_minus_spread"] = _rank_report(
            plus - minus, test_y, test["assignment"], test["source_id"])
        if variant == "state_only":
            train_arm_means = {arm: _fixed_value(train_y, train["assignment"], arm)
                               for arm in (-1, 1)}
            policy = np.full(len(test_state), 1 if train_arm_means[1] >= train_arm_means[-1]
                             else -1, dtype=np.float32)
        else:
            policy = np.where(plus >= minus, 1, -1).astype(np.float32)
        value, matched = _hajek(test_y, test["assignment"], policy,
                                test["propensity_plus"])
        reports[variant]["policy"] = {
            "value": value, "matched_rows": matched,
            "match_fraction": float(matched / len(policy)),
            "cluster_95ci": _bootstrap_policy(
                test_y, test["assignment"], policy, test["propensity_plus"],
                test["source_id"], seed=seed + 100 + list(VARIANTS).index(variant)),
            "choice_plus": int((policy == 1).sum()),
            "choice_minus": int((policy == -1).sum()),
        }
        # A small continuous head is fitted with exactly the same state/action
        # representation.  It is a secondary diagnostic, not the gate target.
        continuous = {}
        for target in ("final_max_contact_lift_m", "final_contact_fraction"):
            reg = Ridge(alpha=10.0)
            reg.fit(x_train, train[target])
            pred = reg.predict(x_test)
            pplus = reg.predict(_design(test_state, np.ones(len(test_state)), variant))
            pminus = reg.predict(_design(test_state, -np.ones(len(test_state)), variant))
            continuous[target] = {
                "factual_rmse": float(mean_squared_error(test[target], pred,
                                                          squared=False)),
                "predicted_contrast_mean": float((pplus - pminus).mean()),
                "contrast_rank": _rank_report(pplus - pminus, test[target],
                                               test["assignment"], test["source_id"]),
            }
            models[(variant, target)] = reg
        reports[variant]["continuous"] = continuous
    metadata = {"state_scaler": scaler, "state_pca": pca, "models": models,
                "pca_components": int(pca.n_components_)}
    return reports, metadata


def test_y_global(rows: dict[str, np.ndarray]) -> np.ndarray:
    """Named helper keeps the binary target choice explicit in reports."""
    return rows["final_lift_success"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", action="append", required=True,
                        type=Path, help="completed randomized-action file; repeat")
    parser.add_argument("--test-input", action="append", required=True,
                        type=Path, help="held-out completed file; repeat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    if len(args.train_input) < 2 or len(args.test_input) < 2:
        parser.error("use at least two train and two test seeds")
    args.output.mkdir(parents=True)
    manifest = {
        "run_status": "STARTED", "run_id": args.output.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "train_input": [str(path) for path in args.train_input],
        "test_input": [str(path) for path in args.test_input],
        "fit_seed": args.fit_seed, "pca_components_max": 24,
        "classifier_C": 0.1, "ridge_alpha": 10.0,
        "policy_estimator": "Hajek inverse propensity, test assignment only",
        "wall_budget_minutes": 30, "output_budget_mb": 20,
        "stop_rule": "input drift, incomplete labels, non-finite model or output",
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        train_parts = [_load_rows(path, index) for index, path in enumerate(args.train_input)]
        test_parts = [_load_rows(path, 100 + index)
                      for index, path in enumerate(args.test_input)]
        train, test = _pool(train_parts), _pool(test_parts)
        train_reports, artifacts = _fit_and_score(
            train, test, train_parts, seed=args.fit_seed)
        # Report the unmodelled randomized contrast first, as a sanity check.
        raw = {}
        for target in TARGETS:
            raw[target] = {
                "train_plus_minus": _fixed_value(train[target], train["assignment"], 1) -
                _fixed_value(train[target], train["assignment"], -1),
                "test_plus_minus": _fixed_value(test[target], test["assignment"], 1) -
                _fixed_value(test[target], test["assignment"], -1),
            }
        aware = train_reports["cm_aware"]["policy"]["value"]
        blind = train_reports["state_only"]["policy"]["value"]
        shuffled = train_reports["action_shuffled"]["policy"]["value"]
        rank_aware = train_reports["cm_aware"]["binary_plus_minus_spread"]
        rank_shuffled = train_reports["action_shuffled"]["binary_plus_minus_spread"]
        gate = {
            "aware_policy_above_state_only": bool(aware > blind),
            "aware_policy_above_shuffled": bool(aware > shuffled),
            "aware_binary_rank_above_shuffled": bool(
                rank_aware["high_minus_low"] > rank_shuffled["high_minus_low"]),
            "pooled_minimum_gain_5pp": bool(aware >= max(blind, shuffled) + 0.05),
        }
        gate["prespecified_gate_passed"] = bool(all(gate.values()))
        report = {
            "schema": "ref2dex.postcontact_heldlift_value_probe.v1",
            "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
            "train_rows": int(len(train["assignment"])),
            "test_rows": int(len(test["assignment"])),
            "train_files": [{"path": part["path"], "sha256": part["sha256"],
                             "rows": int(len(part["assignment"])),
                             "plus": int((part["assignment"] == 1).sum()),
                             "minus": int((part["assignment"] == -1).sum())}
                            for part in train_parts],
            "test_files": [{"path": part["path"], "sha256": part["sha256"],
                            "rows": int(len(part["assignment"])),
                            "plus": int((part["assignment"] == 1).sum()),
                            "minus": int((part["assignment"] == -1).sum())}
                           for part in test_parts],
            "feature_dim_before_pca": int(train["state"].shape[1]),
            "feature_dim_after_pca": artifacts["pca_components"],
            "targets": TARGETS, "raw_randomized_contrast": raw,
            "models": train_reports, "prespecified_gate": gate,
            "limits": [
                "the test policy value is an inverse-propensity estimate on randomized rows",
                "one intervention per episode identifies a population contrast, not an individual counterfactual",
                "the fixed airplane distribution and wrist-z candidate family are narrow",
                "a failed exploratory gate is not evidence that every Cm representation fails",
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        joblib.dump(artifacts, args.output / "models.joblib", compress=3)
        manifest.update(
            run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
            report_sha256=sha256(report_path),
            model_sha256=sha256(args.output / "models.joblib"),
        )
        print(json.dumps({"run_status": "COMPLETED", "gate": gate,
                          "policy": {"aware": aware, "state_only": blind,
                                     "shuffled": shuffled},
                          "raw": raw}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
