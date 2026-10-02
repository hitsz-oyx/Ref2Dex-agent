#!/usr/bin/env python3
"""Offline contextual-bandit audit for a Cm physical value adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge


NAMES = ["expert0", "expert1", "expert2", "expert3", "expert4", "expert5", "base_hold", "fixed_cup"]


def split_group(motion, start):
    return np.asarray([
        int(hashlib.sha256(f"cm-value-adapter/{int(m)}/{int(s)}".encode()).hexdigest()[:8], 16) % 100 >= 70
        for m, s in zip(motion, start)
    ], dtype=bool)


def load_records(root):
    paths = sorted(root.glob("*/records.pt"))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    if not payloads or any(p.get("schema") != "ref2dex.cm_candidate_advantage.v1" for p in payloads):
        raise ValueError("candidate advantage schema mismatch")
    checkpoint = payloads[0]["checkpoint_sha256"]
    if any(p["checkpoint_sha256"] != checkpoint for p in payloads):
        raise ValueError("checkpoint drift")
    if any(not p.get("frozen_cm") or p.get("optimizer_used") or p.get("fixed_cup_index") != 7 for p in payloads):
        raise ValueError("frozen Cm contract failed")
    assignment = torch.cat([p["assignment"] for p in payloads]).numpy().astype(int)
    state = torch.cat([p["state"] for p in payloads]).numpy().astype(float)
    pd = torch.cat([p["candidate_pd_targets"] for p in payloads]).numpy().astype(float)
    diag = {key: torch.cat([p["cm_diagnostics"][key] for p in payloads]).numpy().astype(float)
            for key in ("score_mm", "relative_std_mm", "retention", "release")}
    y = torch.cat([p["outcome"]["score_mm"] for p in payloads]).numpy().astype(float)
    motion = torch.cat([p["motion_id"] for p in payloads]).numpy().astype(int)
    start = torch.cat([p["start_frame"] for p in payloads]).numpy().astype(int)
    return paths, payloads, checkpoint, assignment, state, pd, diag, y, motion, start


def candidate_features(state, pd, diag, include_cm):
    n = len(state)
    k = 8
    common = state[:, None, :].repeat(k, axis=1)
    ids = np.broadcast_to(np.eye(k)[None], (n, k, k))
    base = np.concatenate((common, pd, ids), axis=-1)
    if not include_cm:
        return base
    panel = np.concatenate([diag[key] for key in ("score_mm", "relative_std_mm", "retention", "release")], axis=-1)
    panel = panel[:, None, :].repeat(k, axis=1)
    selected = np.stack([diag[key] for key in ("score_mm", "relative_std_mm", "retention", "release")], axis=-1)
    fixed = np.repeat(selected[:, 7:8, :], k, axis=1)
    # candidate-conditioned physical consequence features remain explicit;
    # the adapter cannot see any future state or post-action label.
    return np.concatenate((base, selected, selected - fixed, panel), axis=-1)


def fit_predict_all(x_train, y_train, x_eval):
    n, k, d = x_train.shape
    x_train = x_train.reshape(n * k, d)
    # The observed action is selected by the caller; only one target per row is
    # supplied, so the same row features are used with its assigned candidate.
    model = Ridge(alpha=1.0)
    mean = x_train.mean(0)
    scale = x_train.std(0)
    scale[scale < 1e-8] = 1.0
    # Train only on the observed assigned candidate rows.
    return model, mean, scale


def train_predict(x_train, y_train, assignment_train, x_eval):
    rows = np.arange(len(y_train))
    observed = x_train[rows, assignment_train]
    mean = observed.mean(0)
    scale = observed.std(0)
    scale[scale < 1e-8] = 1.0
    model = Ridge(alpha=1.0).fit((observed - mean) / scale, y_train)
    n, k, d = x_eval.shape
    return model.predict(((x_eval.reshape(n * k, d) - mean) / scale)).reshape(n, k)


def bootstrap_delta(y, observed, proposed, groups, draws=10000, seed=7001):
    unique, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    advantage = np.where(observed == proposed, y * 8.0, 0.0)
    fixed = np.where(observed == 7, y * 8.0, 0.0)
    for i in range(draws):
        mult = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        denom = max(float(mult[code].sum()), 1.0)
        values[i] = ((advantage - fixed) * mult[code]).sum() / denom
    return dict(mean=float(values.mean()), lower90=float(np.percentile(values, 5)),
                upper90=float(np.percentile(values, 95)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths, payloads, checkpoint, assignment, state, pd, diag, y, motion, start = load_records(args.root)
    groups = np.asarray([f"{m}:{s}" for m, s in zip(motion, start)])
    held = split_group(motion, start)
    if held.sum() == 0 or (~held).sum() == 0:
        raise ValueError("empty split")
    x_base = candidate_features(state, pd, diag, False)
    x_cm = candidate_features(state, pd, diag, True)
    reports = {}
    choices = {}
    for name, x in (("state_action", x_base), ("state_action_plus_cm", x_cm)):
        prediction = train_predict(x[~held], y[~held], assignment[~held], x[held])
        choice = prediction.argmax(-1)
        choices[name] = choice
        reports[name] = dict(
            rows=int(held.sum()), groups=int(np.unique(groups[held]).size),
            choice_counts=np.bincount(choice, minlength=8).tolist(),
            changed_from_fixed=int((choice != 7).sum()),
            changed_fraction=float(np.mean(choice != 7)),
            ipw_delta_vs_fixed=bootstrap_delta(y[held], assignment[held], choice, groups[held]),
            predicted_score_gap=float(np.mean(prediction.max(-1) - prediction[:, 7])),
        )
    result = dict(
        schema="ref2dex.cm_value_adapter_audit.v1", run_status="COMPLETED",
        records=[str(path.resolve()) for path in paths], rows=len(y), groups=int(np.unique(groups).size),
        checkpoint_sha256=checkpoint, target="one-candidate H10 physical score_mm",
        assignment_counts=np.bincount(assignment, minlength=8).tolist(),
        split=dict(fit_rows=int((~held).sum()), held_rows=int(held.sum()),
                   fit_groups=int(np.unique(groups[~held]).size), held_groups=int(np.unique(groups[held]).size)),
        reports=reports,
        interpretation="Frozen Cm physical transition features feed a separate local-score adapter; no success or trained-policy claim",
    )
    result["cm_increment"] = dict(
        score_delta_mean=reports["state_action_plus_cm"]["ipw_delta_vs_fixed"]["mean"] - reports["state_action"]["ipw_delta_vs_fixed"]["mean"],
        choice_agreement=float(np.mean(choices["state_action_plus_cm"] == choices["state_action"])),
    )
    result["probe_label"] = "PROMISING_OFFLINE" if reports["state_action_plus_cm"]["ipw_delta_vs_fixed"]["lower90"] > 0 else "UNPROMISING_OFFLINE"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
