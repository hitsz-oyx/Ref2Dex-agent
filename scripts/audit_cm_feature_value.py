#!/usr/bin/env python3
"""Audit whether frozen Cm physical features add local task-value information.

The target is the measured one-candidate H10 physical score.  This is a grouped
offline sufficiency probe, not a success classifier and not a policy result.
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
from sklearn.model_selection import GroupKFold


def held_group(motion: np.ndarray, start: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(hashlib.sha256(f"cm-feature-audit/{int(m)}/{int(s)}".encode()).hexdigest()[:8], 16) % 100 >= 70
        for m, s in zip(motion, start)
    ], dtype=bool)


def features(payloads):
    assignment = torch.cat([p["assignment"] for p in payloads]).numpy().astype(int)
    state = torch.cat([p["state"] for p in payloads]).numpy().astype(float)
    pd = torch.cat([p["candidate_pd_targets"] for p in payloads]).numpy().astype(float)
    cm = {key: torch.cat([p["cm_diagnostics"][key] for p in payloads]).numpy().astype(float)
          for key in ("score_mm", "relative_std_mm", "retention", "release")}
    n = len(assignment)
    rows = np.arange(n)
    chosen_pd = pd[rows, assignment]
    chosen = np.stack([cm[key][rows, assignment] for key in cm], axis=1)
    fixed = np.stack([cm[key][:, 7] for key in cm], axis=1)
    # The baseline sees the physical state, actual candidate command, and ID.
    baseline = np.concatenate((state, chosen_pd, np.eye(8)[assignment]), axis=1)
    # Cm remains a physical consequence representation: selected prediction,
    # relative-to-fixed prediction, and full candidate-panel context.
    panel = np.concatenate([cm[key] for key in cm], axis=1)
    extra = np.concatenate((chosen, chosen - fixed, panel), axis=1)
    return baseline, np.concatenate((baseline, extra), axis=1), assignment


def fit_predict(x_fit, y_fit, x_eval):
    mean = x_fit.mean(0)
    scale = x_fit.std(0)
    scale[scale < 1e-8] = 1.0
    model = Ridge(alpha=1.0).fit((x_fit - mean) / scale, y_fit)
    return model.predict((x_eval - mean) / scale)


def metrics(y, pred):
    return dict(rmse_mm=float(np.sqrt(np.mean((y - pred) ** 2))),
                mae_mm=float(np.mean(np.abs(y - pred))),
                spearman=float(spearmanr(y, pred).statistic))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/records.pt"))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    if not payloads or any(p.get("schema") != "ref2dex.cm_candidate_advantage.v1" for p in payloads):
        raise ValueError("candidate advantage schema mismatch")
    checkpoint = payloads[0]["checkpoint_sha256"]
    if any(p["checkpoint_sha256"] != checkpoint for p in payloads):
        raise ValueError("checkpoint drift")
    if any(not p.get("frozen_cm") or p.get("optimizer_used") or p.get("fixed_cup_index") != 7 for p in payloads):
        raise ValueError("frozen physical-model contract failed")
    x_base, x_cm, assignment = features(payloads)
    y = torch.cat([p["outcome"]["score_mm"] for p in payloads]).numpy().astype(float)
    motion = torch.cat([p["motion_id"] for p in payloads]).numpy().astype(int)
    start = torch.cat([p["start_frame"] for p in payloads]).numpy().astype(int)
    groups = np.asarray([f"{m}:{s}" for m, s in zip(motion, start)])
    held = held_group(motion, start)
    if held.sum() == 0 or (~held).sum() == 0:
        raise ValueError("empty fixed split")
    result = dict(schema="ref2dex.cm_feature_value_audit.v1", run_status="COMPLETED",
                  records=[str(p.resolve()) for p in paths], rows=len(y), groups=int(np.unique(groups).size),
                  checkpoint_sha256=checkpoint, target="one-candidate H10 physical score_mm",
                  assignment_counts=np.bincount(assignment, minlength=8).tolist(),
                  fixed_split=dict(fit_rows=int((~held).sum()), held_rows=int(held.sum()),
                                   fit_groups=int(np.unique(groups[~held]).size), held_groups=int(np.unique(groups[held]).size)))
    result["fixed_holdout"] = {}
    for name, x in (("state_action", x_base), ("state_action_plus_cm", x_cm)):
        pred = fit_predict(x[~held], y[~held], x[held])
        result["fixed_holdout"][name] = metrics(y[held], pred)
    # Group-disjoint out-of-fold comparison tests whether the result is tied to
    # one arbitrary held-out group split.
    result["group_oof"] = {"folds": 5, "models": {"state_action": [], "state_action_plus_cm": []}}
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(x_base, y, groups):
        for name, x in (("state_action", x_base), ("state_action_plus_cm", x_cm)):
            result["group_oof"]["models"][name].append(metrics(y[test], fit_predict(x[train], y[train], x[test])))
    # A simple incremental screen: fit each prefix of the disjoint groups in a
    # fixed order and evaluate the same held groups.  This distinguishes a
    # feature failure from a clear lack of data without a hyperparameter sweep.
    fit_groups = np.unique(groups[~held])
    curve = []
    for count in (max(5, len(fit_groups) // 3), max(5, 2 * len(fit_groups) // 3), len(fit_groups)):
        chosen_groups = set(fit_groups[:count])
        train = np.asarray([g in chosen_groups for g in groups])
        curve.append(dict(fit_groups=count, state_action=metrics(y[held], fit_predict(x_base[train], y[train], x_base[held])),
                          state_action_plus_cm=metrics(y[held], fit_predict(x_cm[train], y[train], x_cm[held]))))
    result["fixed_holdout_learning_curve"] = curve
    result["interpretation"] = "Cm physical feature sufficiency for measured local consequence; no policy or success claim"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
