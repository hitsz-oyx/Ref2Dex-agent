#!/usr/bin/env python3
"""Audit a direct-Q plus Cm-relative residual on randomized candidate rows.

Only rows assigned to the uniform-random arm are used for fitting/evaluation.
The other arms are retained only for reporting how often a frozen score would
change the action from the Cup baseline.  This is an offline ranking Probe;
it is not a policy or counterfactual-success evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    x1 = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    eye = np.eye(x1.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(x1.T @ x1 + lam * eye, x1.T @ y)


def _predict(beta: np.ndarray, direct: np.ndarray, cm: np.ndarray,
             means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    x = np.stack([direct, cm], axis=-1)
    x = (x - means) / scales
    x1 = np.concatenate([np.ones((*x.shape[:-1], 1)), x], axis=-1)
    return x1 @ beta


def analyze(record: Path, output: Path) -> None:
    data = torch.load(record, map_location="cpu", weights_only=False)
    if data.get("schema") != "ref2dex.cm_decision_interface.v2":
        raise ValueError("decision-interface v1 is frozen; use corrected v2 records without shuffled control")
    if data.get("simulator_seed") is None:
        raise ValueError("corrected decision record must include simulator_seed")
    names = list(data["arm_names"])
    if "shuffled" in names or "random" not in names or "cup" not in names:
        raise ValueError("random and cup arms are required")
    random_arm, cup_arm = names.index("random"), names.index("cup")

    assignment = data["assignment"].numpy().astype(int)
    selected = data["selected_index"].numpy().astype(int)
    direct = data["direct_q_scores"].numpy().astype(np.float64)
    cm = data["cm_value_scores"].numpy().astype(np.float64)
    state = data["state"].numpy()
    future = data["future_state"].numpy()
    reward = data["future_reward"].numpy().sum(-1)
    height = np.maximum(future[..., 38] - state[:, None, 38], 0.0)
    height_min = height[..., -3:].min(-1) * 1000.0
    if not np.isfinite(direct).all() or not np.isfinite(cm).all():
        raise ValueError("nonfinite teacher scores")
    if not np.isfinite(height_min).all() or not np.isfinite(reward).all():
        raise ValueError("nonfinite observed utility")

    # Candidate zero is the Cup baseline.  The residuals are the only Cm
    # channel exposed to the fitted interface.
    direct_rel = direct - direct[:, :1]
    cm_rel = cm - cm[:, :1]
    row = np.arange(len(assignment))
    random = assignment == random_arm
    random_rows = np.flatnonzero(random)
    if len(random_rows) < 24:
        raise ValueError("uniform-random support below Probe contract")

    # Group split is independent of arm assignment.  This prevents repeated
    # motion/start contexts from appearing in both fitting and held-out rows.
    motion = data["motion_id"].numpy().astype(int)
    start = data["start_frame"].numpy().astype(int)
    group_names = np.asarray([f"{m}/{s}" for m, s in zip(motion, start)])
    unique = np.unique(group_names)
    group_code = {g: i for i, g in enumerate(unique)}
    fold = np.asarray([group_code[g] % 5 for g in group_names])
    fit = random & (fold != 0)
    held = random & (fold == 0)
    held_rows = np.flatnonzero(held)
    split_support_ok = bool(int(held.sum()) >= 8 and len(np.unique(group_names[held])) >= 5)
    if not split_support_ok:
        result = dict(
            schema="ref2dex.cm_relative_direct_q_audit.v2",
            run_status="COMPLETED",
            experiment_id="P-20261003-cm-relative-direct-q-corrected",
            record=str(record.resolve()),
            record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
            rows=int(len(assignment)), random_rows=int(random.sum()),
            held_random_rows=int(held.sum()), held_groups=int(len(np.unique(group_names[held]))),
            fit_random_rows=int(fit.sum()), reports={}, coefficients={},
            prospective_changed_rows=None, prospective_changed_fraction=None,
            prospective_top_counts=None,
            gates=dict(support_ok=False, ranking_ok=False, coverage_ok=False),
            probe_label="UNCLEAR",
            interpretation=("Uniform-random rows do not provide the predeclared "
                            "held-out group support; no residual fit or native action "
                            "claim is allowed"),
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    chosen_direct = direct_rel[row, selected]
    chosen_cm = cm_rel[row, selected]
    # The random arm makes selected candidates independent of score.  Fit one
    # local-utility head per target, then apply the same frozen interface to
    # every candidate in held-out states.
    reports = {}
    coefficients = {}
    for target_name, target in (("height_last3_min_mm", height_min),
                                ("local_reward_sum", reward)):
        fit_rows = np.flatnonzero(fit)
        xfit = np.stack([chosen_direct[fit_rows], chosen_cm[fit_rows]], -1)
        means = xfit.mean(0)
        scales = xfit.std(0).clip(1e-6)
        beta = _ridge((xfit - means) / scales, target[fit_rows])
        pred_direct = _predict(beta, direct_rel, np.zeros_like(cm_rel), means, scales)
        pred_cm = _predict(beta, np.zeros_like(direct_rel), cm_rel, means, scales)
        pred_blend = _predict(beta, direct_rel, cm_rel, means, scales)
        actual = target[held_rows]
        observed_direct = pred_direct[held_rows, selected[held_rows]]
        observed_cm = pred_cm[held_rows, selected[held_rows]]
        observed_blend = pred_blend[held_rows, selected[held_rows]]
        scores = {}
        for name, values in (("direct_q", observed_direct), ("cm_residual", observed_cm),
                             ("blend", observed_blend)):
            scores[name] = dict(
                held_spearman=_spearman(values, actual),
                held_mean_pred=float(values.mean()),
            )
        top = pred_blend.argmax(-1)
        top_direct = pred_direct.argmax(-1)
        top_cm = pred_cm.argmax(-1)
        scores["blend"]["held_top_match"] = int((top[held_rows] == selected[held_rows]).sum())
        scores["blend"]["held_top_match_fraction"] = float(
            (top[held_rows] == selected[held_rows]).mean())
        scores["direct_q"]["held_top_match"] = int(
            (top_direct[held_rows] == selected[held_rows]).sum())
        scores["cm_residual"]["held_top_match"] = int(
            (top_cm[held_rows] == selected[held_rows]).sum())
        coefficients[target_name] = dict(beta=beta.tolist(), means=means.tolist(), scales=scales.tolist())
        reports[target_name] = scores

    # Use the height head to report prospective action coverage.  This is a
    # frozen score audit only; utility is evaluated on random held-out rows.
    height_coef = coefficients["height_last3_min_mm"]
    beta = np.asarray(height_coef["beta"])
    means = np.asarray(height_coef["means"])
    scales = np.asarray(height_coef["scales"])
    prospective = _predict(beta, direct_rel, cm_rel, means, scales)
    prospective_top = prospective.argmax(-1)
    changed = prospective_top != 0
    random_held_top = prospective_top[held]
    # The held random rows are the only rows used for the action-ranking gate.
    blend_height = reports["height_last3_min_mm"]["blend"]["held_spearman"]
    direct_height = reports["height_last3_min_mm"]["direct_q"]["held_spearman"]
    blend_reward = reports["local_reward_sum"]["blend"]["held_spearman"]
    direct_reward = reports["local_reward_sum"]["direct_q"]["held_spearman"]
    ranking_ok = bool(
        np.isfinite(blend_height) and np.isfinite(blend_reward)
        and blend_height > 0 and blend_reward > 0
        and blend_height >= direct_height + 0.05
        and blend_reward >= direct_reward + 0.05
    )
    coverage_ok = bool(int(changed.sum()) >= 24 and int(reports["height_last3_min_mm"]["blend"]["held_top_match"]) >= 8)
    support_ok = bool(int(held.sum()) >= 8 and len(np.unique(group_names[held])) >= 5)
    label = "PROMISING" if support_ok and ranking_ok and coverage_ok else (
        "UNCLEAR" if not support_ok or not coverage_ok else "UNPROMISING")
    result = dict(
        schema="ref2dex.cm_relative_direct_q_audit.v2",
        run_status="COMPLETED",
        experiment_id="P-20261003-cm-relative-direct-q-corrected",
        record=str(record.resolve()),
        record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
        rows=int(len(assignment)), random_rows=int(random.sum()),
        held_random_rows=int(held.sum()), held_groups=int(len(np.unique(group_names[held]))),
        fit_random_rows=int(fit.sum()),
        reports=reports, coefficients=coefficients,
        prospective_changed_rows=int(changed.sum()),
        prospective_changed_fraction=float(changed.mean()),
        prospective_top_counts=np.bincount(prospective_top, minlength=direct.shape[1]).tolist(),
        gates=dict(support_ok=support_ok, ranking_ok=ranking_ok, coverage_ok=coverage_ok),
        probe_label=label,
        interpretation=("Offline randomized-row ranking only; no counterfactual policy utility, "
                        "native control, or PPO claim"),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.record, args.output)


if __name__ == "__main__":
    main()
