#!/usr/bin/env python3
"""Probe an action-relative Cm target with motion/start group isolation.

The candidate panel is randomized with p=1/8.  This audit never uses a future
state as an input.  It compares an absolute score adapter with a fit-only
leave-one-row-out, group-centered target so state-level baseline variation cannot
be mistaken for a candidate consequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

NAMES = ["expert0", "expert1", "expert2", "expert3", "expert4", "expert5", "base_hold", "fixed_cup"]


def load_records(paths):
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    if not payloads or any(p.get("schema") != "ref2dex.cm_candidate_advantage.v1" for p in payloads):
        raise ValueError("candidate advantage schema mismatch")
    checkpoint = payloads[0]["checkpoint_sha256"]
    if any(p["checkpoint_sha256"] != checkpoint for p in payloads):
        raise ValueError("checkpoint drift")
    if any(p["fixed_cup_index"] != 7 or not p["frozen_cm"] or p["optimizer_used"] for p in payloads):
        raise ValueError("frozen candidate contract failed")
    keys = ("assignment", "state", "candidate_actions", "candidate_pd_targets",
            "motion_id", "start_frame")
    arrays = {key: torch.cat([p[key] for p in payloads]).numpy() for key in keys}
    arrays["score"] = torch.cat([p["outcome"]["score_mm"] for p in payloads]).numpy()
    for key in ("retained", "contact_last3", "clearance_last3"):
        arrays[key] = torch.cat([p["outcome"][key] for p in payloads]).numpy().astype(np.float64)
    arrays["cm_score"] = torch.cat([p["cm_diagnostics"]["score_mm"] for p in payloads]).numpy()
    arrays["cm_std"] = torch.cat([p["cm_diagnostics"]["relative_std_mm"] for p in payloads]).numpy()
    arrays["cm_retention"] = torch.cat([p["cm_diagnostics"]["retention"] for p in payloads]).numpy()
    arrays["cm_release"] = torch.cat([p["cm_diagnostics"]["release"] for p in payloads]).numpy()
    arrays["groups"] = np.asarray([
        f"{int(m)}/{int(s)}" for m, s in zip(arrays["motion_id"], arrays["start_frame"])
    ])
    if not np.allclose(np.concatenate([p["propensity"].numpy() for p in payloads]), 1 / 8):
        raise ValueError("non-uniform propensity")
    if not np.isfinite(arrays["score"]).all():
        raise ValueError("non-finite score")
    return payloads, arrays, checkpoint


def split_groups(groups):
    held = np.asarray([
        int(hashlib.sha256(f"cm-value-adapter/{group.replace('/', '/') }".encode()).hexdigest()[:8], 16) % 100 >= 70
        for group in groups
    ], dtype=bool)
    # The hash above intentionally has the same namespace as the existing
    # adapter.  Convert row flags to a group flag so a group cannot straddle fit/held.
    unique = np.unique(groups)
    group_held = {group: bool(held[np.flatnonzero(groups == group)[0]]) for group in unique}
    return np.asarray([group_held[group] for group in groups], dtype=bool)


def cross_fit_folds(groups, count=5):
    unique = np.unique(groups)
    assignments = {
        group: int(hashlib.sha256(f"cm-group-relative-cv/{group}".encode()).hexdigest()[:8], 16) % count
        for group in unique
    }
    return np.asarray([assignments[group] for group in groups], dtype=np.int64)


def features(arrays, include_cm):
    state = arrays["state"].astype(np.float64)
    actions = arrays["candidate_actions"].astype(np.float64)
    pd = arrays["candidate_pd_targets"].astype(np.float64)
    n, count = actions.shape[:2]
    common = np.repeat(state[:, None, :], count, axis=1)
    identity = np.broadcast_to(np.eye(count, dtype=np.float64)[None], (n, count, count))
    base = np.concatenate((common, actions, pd, identity), axis=-1)
    if not include_cm:
        return base
    selected = np.stack((arrays["cm_score"], arrays["cm_std"], arrays["cm_retention"], arrays["cm_release"]), axis=-1)
    fixed = selected[:, 7:8]
    selected_delta = selected - fixed
    panel = selected[:, None].repeat(count, axis=1).reshape(n, count, -1)
    return np.concatenate((base, selected, selected_delta, panel), axis=-1)


def fit_predict(x_fit, target_fit, x_eval, fit_rows):
    observed = x_fit[np.arange(len(fit_rows)), fit_rows]
    mean = observed.mean(0)
    scale = observed.std(0)
    scale[scale < 1e-8] = 1.0
    model = Ridge(alpha=1.0).fit((observed - mean) / scale, target_fit)
    flat = ((x_eval.reshape(-1, x_eval.shape[-1]) - mean) / scale)
    return model.predict(flat).reshape(len(x_eval), x_eval.shape[1])


def group_centered_target(y, groups, fit):
    target = np.full(len(y), np.nan, dtype=np.float64)
    for group in np.unique(groups[fit]):
        rows = np.flatnonzero(fit & (groups == group))
        if len(rows) < 2:
            continue
        total = y[rows].sum()
        target[rows] = y[rows] - (total - y[rows]) / (len(rows) - 1)
    return target


def bootstrap_policy(y, assignment, choice, groups, draws=10000, seed=12607):
    unique, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    numerator = np.bincount(
        code, weights=np.where(assignment == choice, y * 8.0, 0.0), minlength=len(unique)
    )
    denominator = np.bincount(code, minlength=len(unique)).astype(float)
    values = np.empty(draws)
    for index in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        values[index] = (numerator * multiplicity).sum() / max((denominator * multiplicity).sum(), 1.0)
    return values


def interval(values):
    return {
        "mean": float(np.mean(values)),
        "lower90": float(np.percentile(values, 5)),
        "upper90": float(np.percentile(values, 95)),
    }


def demeaned_spearman(prediction, target, groups):
    pred = np.asarray(prediction, dtype=np.float64).copy()
    actual = np.asarray(target, dtype=np.float64).copy()
    for group in np.unique(groups):
        rows = groups == group
        pred[rows] -= pred[rows].mean()
        actual[rows] -= actual[rows].mean()
    valid = np.isfinite(pred) & np.isfinite(actual)
    statistic = spearmanr(pred[valid], actual[valid]).statistic
    return float(statistic) if np.isfinite(statistic) else None


def cross_fit_prediction(x, y, groups, assignment, relative):
    fold = cross_fit_folds(groups)
    prediction = np.full((len(y), x.shape[1]), np.nan, dtype=np.float64)
    for held_fold in range(5):
        train = fold != held_fold
        test = fold == held_fold
        target = y
        if relative:
            centered = group_centered_target(y, groups, train)
            train &= np.isfinite(centered)
            target = centered
        if not train.any() or not test.any():
            raise ValueError("empty cross-fit fold")
        prediction[test] = fit_predict(
            x[train], target[train], x[test], assignment[train].astype(np.int64)
        )
    return prediction, fold


def policy_report(prediction, y, assignment, groups, metrics, label, seed):
    choice = prediction.argmax(-1)
    fixed_choice = np.full(len(y), 7, dtype=np.int64)
    values = bootstrap_policy(y, assignment, choice, groups, seed=seed)
    fixed = bootstrap_policy(y, assignment, fixed_choice, groups, seed=seed + 1)
    observed_prediction = prediction[np.arange(len(choice)), assignment]
    report = {
        "rows": int(len(y)),
        "groups": int(np.unique(groups).size),
        "changed_from_fixed": int((choice != 7).sum()),
        "changed_fraction": float(np.mean(choice != 7)),
        "choice_counts": np.bincount(choice, minlength=8).tolist(),
        "ipw_delta_vs_fixed": interval(values - fixed),
        "held_demeaned_spearman": demeaned_spearman(observed_prediction, y, groups),
        "metric_ipw_delta_vs_fixed": {},
        "label": label,
    }
    for index, (metric_name, metric_values) in enumerate(metrics.items()):
        policy_metric = bootstrap_policy(metric_values, assignment, choice, groups, seed=seed + 2 + index * 2)
        fixed_metric = bootstrap_policy(metric_values, assignment, fixed_choice, groups, seed=seed + 3 + index * 2)
        report["metric_ipw_delta_vs_fixed"][metric_name] = interval(policy_metric - fixed_metric)
    return report


def audit(paths, output):
    payloads, arrays, checkpoint = load_records(paths)
    groups = arrays["groups"]
    held = split_groups(groups)
    fit = ~held
    if held.sum() == 0 or fit.sum() == 0 or np.unique(groups[held]).size < 5:
        raise ValueError("insufficient group split")
    x_base = features(arrays, include_cm=False)
    x_cm = features(arrays, include_cm=True)
    absolute_target = arrays["score"].astype(np.float64)
    centered = group_centered_target(absolute_target, groups, fit)
    fit_centered = fit & np.isfinite(centered)
    choices = {}
    reports = {}
    fixed_choice = np.full(held.sum(), 7, dtype=np.int64)
    held_y = absolute_target[held]
    held_assignment = arrays["assignment"][held].astype(np.int64)
    held_groups = groups[held]
    held_metrics = {
        "score_mm": held_y,
        "retained": arrays["retained"][held],
        "contact_last3": arrays["contact_last3"][held],
        "clearance_last3": arrays["clearance_last3"][held],
    }
    for name, x, target, rows in (
        ("absolute_state_action", x_base, absolute_target, fit),
        ("absolute_state_action_plus_cm", x_cm, absolute_target, fit),
        ("group_relative_state_action_plus_cm", x_cm, centered, fit_centered),
    ):
        prediction = fit_predict(x[rows], target[rows], x[held], arrays["assignment"][rows].astype(np.int64))
        choice = prediction.argmax(-1)
        choices[name] = choice
        values = bootstrap_policy(held_y, held_assignment, choice, held_groups)
        fixed = bootstrap_policy(held_y, held_assignment, fixed_choice, held_groups, seed=12608)
        observed_prediction = prediction[np.arange(len(choice)), held_assignment]
        reports[name] = {
            "rows": int(held.sum()),
            "groups": int(np.unique(held_groups).size),
            "changed_from_fixed": int((choice != 7).sum()),
            "changed_fraction": float(np.mean(choice != 7)),
            "choice_counts": np.bincount(choice, minlength=8).tolist(),
            "ipw_delta_vs_fixed": interval(values - fixed),
            "held_demeaned_spearman": demeaned_spearman(observed_prediction, held_y, held_groups),
            "metric_ipw_delta_vs_fixed": {},
        }
        for metric_name, metric_values in held_metrics.items():
            policy_metric = bootstrap_policy(metric_values, held_assignment, choice, held_groups, seed=12611)
            fixed_metric = bootstrap_policy(metric_values, held_assignment, fixed_choice, held_groups, seed=12612)
            reports[name]["metric_ipw_delta_vs_fixed"][metric_name] = interval(policy_metric - fixed_metric)
    for name, choice in (
        ("cm_raw_top", arrays["cm_score"][held].argmax(-1)),
        ("fixed_cup", fixed_choice),
    ):
        values = bootstrap_policy(held_y, held_assignment, choice, held_groups, seed=12609)
        fixed = bootstrap_policy(held_y, held_assignment, fixed_choice, held_groups, seed=12610)
        reports[name] = {
            "rows": int(held.sum()),
            "groups": int(np.unique(held_groups).size),
            "changed_from_fixed": int((choice != 7).sum()),
            "changed_fraction": float(np.mean(choice != 7)),
            "choice_counts": np.bincount(choice, minlength=8).tolist(),
            "ipw_delta_vs_fixed": interval(values - fixed),
            "metric_ipw_delta_vs_fixed": {},
        }
        for metric_name, metric_values in held_metrics.items():
            policy_metric = bootstrap_policy(metric_values, held_assignment, choice, held_groups, seed=12613)
            fixed_metric = bootstrap_policy(metric_values, held_assignment, fixed_choice, held_groups, seed=12614)
            reports[name]["metric_ipw_delta_vs_fixed"][metric_name] = interval(policy_metric - fixed_metric)

    all_metrics = {
        "score_mm": absolute_target,
        "retained": arrays["retained"],
        "contact_last3": arrays["contact_last3"],
        "clearance_last3": arrays["clearance_last3"],
    }
    all_assignment = arrays["assignment"].astype(np.int64)
    for name, x, relative in (
        ("cross_fit_absolute_state_action_plus_cm", x_cm, False),
        ("cross_fit_group_relative_state_action_plus_cm", x_cm, True),
    ):
        prediction, fold = cross_fit_prediction(x, absolute_target, groups, all_assignment, relative)
        reports[name] = policy_report(
            prediction, absolute_target, all_assignment, groups, all_metrics, name, 12620
        )
        reports[name]["folds"] = {
            "count": 5,
            "group_counts": [int(np.unique(groups[fold == index]).size) for index in range(5)],
            "row_counts": [int((fold == index).sum()) for index in range(5)],
        }
    result = {
        "schema": "ref2dex.cm_group_relative_target_audit.v1",
        "run_status": "COMPLETED",
        "records": [str(path.resolve()) for path in paths],
        "rows": int(len(absolute_target)),
        "groups": int(np.unique(groups).size),
        "split": {
            "fit_rows": int(fit.sum()),
            "held_rows": int(held.sum()),
            "fit_groups": int(np.unique(groups[fit]).size),
            "held_groups": int(np.unique(groups[held]).size),
            "centered_fit_rows": int(fit_centered.sum()),
        },
        "checkpoint_sha256": checkpoint,
        "candidate_names": NAMES,
        "support": [int((arrays["assignment"] == k).sum()) for k in range(8)],
        "reports": reports,
        "target_contract": "fit-only leave-one-row-out motion/start group-centered realized H10 score; no future state input",
        "interpretation": "Offline action-relative value-equivalence Probe; no actor, policy, or success claim",
        "decision_rule": "A new route is promising only if group-relative held IPW lower90 is positive and held demeaned ranking improves over the absolute Cm adapter.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for root in args.root:
        # Complete r1/r3/r4 runs use different directory prefixes; incomplete
        # launches have manifests but no records.pt and are excluded here.
        paths.extend(sorted(root.glob("*/records.pt")))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("no complete candidate records")
    audit(paths, args.output)


if __name__ == "__main__":
    main()
