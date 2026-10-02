#!/usr/bin/env python3
"""Analyze exact same-state Cm action-panel records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def bootstrap(values, groups, draws=10000, seed=91003):
    groups = np.asarray(groups)
    unique, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        mult = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        out[i] = (values * mult[code]).sum() / max(float(mult[code].sum()), 1.0)
    return dict(mean=float(out.mean()), lower90=float(np.percentile(out, 10)),
                upper90=float(np.percentile(out, 90)))


def analyze(record: Path, output: Path):
    payload = torch.load(record, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cm_exact_paired_panel.v1":
        raise ValueError("exact paired panel schema mismatch")
    rows = payload["rows"]
    n = len(rows)
    if n < 1:
        raise ValueError("empty paired panel")
    fixed = payload["fixed_index"]
    scores = np.stack([r["cm_diagnostics"]["score_mm"].numpy() for r in rows])
    std = np.stack([r["cm_diagnostics"]["relative_std_mm"].numpy() for r in rows])
    effect = np.stack([r["actual_effect_z_mm"].numpy() for r in rows])
    valid = np.stack([r["valid"].numpy() for r in rows]).astype(bool)
    if not np.isfinite(scores).all() or not np.isfinite(effect).all():
        raise ValueError("nonfinite panel values")
    if not valid.all():
        errors = {}
        for row in rows:
            for index, error in enumerate(row.get("pair_errors", [])):
                if error is not None:
                    errors.setdefault(str(index), {})[error] = errors.setdefault(str(index), {}).get(error, 0) + 1
        result = dict(
            schema="ref2dex.cm_exact_paired_panel_audit.v1", run_status="COMPLETED",
            experiment_id="P-20261003-cm-exact-paired-panel",
            record=str(record.resolve()), record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
            rows=n, groups=int(len(set(f"{r['motion_id']}/{r['start_frame']}" for r in rows))),
            candidate_names=payload["candidate_names"], checkpoint_sha256=payload["checkpoint_sha256"],
            engineering_blocker="paired restore contract rejected hot-state candidates",
            pair_errors=errors,
            gates=dict(support_ok=False, coverage_ok=False, ranking_ok=False, local_utility_ok=False),
            probe_label="UNCLEAR",
            interpretation=("Engineering-boundary Probe only: rigid-body restore failed before "
                            "candidate effects were observable; no Cm utility or policy claim"),
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    top = scores.argmax(-1)
    top_effect = effect[np.arange(n), top] - effect[:, fixed]
    raw_rank = np.asarray([spearman(scores[i], effect[i]) for i in range(n)])
    groups = np.asarray([f"{r['motion_id']}/{r['start_frame']}" for r in rows])
    changed = top != fixed
    policy = dict(
        top_counts=np.bincount(top, minlength=scores.shape[1]).tolist(),
        changed_from_fixed=int(changed.sum()), changed_fraction=float(changed.mean()),
        top_vs_fixed=bootstrap(top_effect, groups),
        mean_panel_spearman=float(np.nanmean(raw_rank)),
        median_panel_spearman=float(np.nanmedian(raw_rank)),
        spearman_positive_fraction=float(np.mean(raw_rank > 0)),
    )
    # The candidate with maximum Cm score is the only prospective selection;
    # uncertainty is reported but not tuned into a second policy.
    result = dict(
        schema="ref2dex.cm_exact_paired_panel_audit.v1", run_status="COMPLETED",
        experiment_id="P-20261003-cm-exact-paired-panel",
        record=str(record.resolve()), record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
        rows=n, groups=int(np.unique(groups).size), candidate_names=payload["candidate_names"],
        checkpoint_sha256=payload["checkpoint_sha256"], policy=policy,
        effect_by_candidate=[bootstrap(effect[:, k] - effect[:, fixed], groups, seed=91003 + k)
                             for k in range(scores.shape[1])],
        uncertainty_top_margin_mean=float(np.mean(scores[np.arange(n), top] - scores[:, fixed])),
        uncertainty_top_std_mean=float(np.mean(std[np.arange(n), top])),
        gates=dict(
            support_ok=bool(n >= 20 and np.unique(groups).size >= 12 and valid.all()),
            coverage_ok=bool(changed.mean() >= .25),
            ranking_ok=bool(policy["mean_panel_spearman"] > 0 and policy["spearman_positive_fraction"] >= .60),
            local_utility_ok=bool(policy["top_vs_fixed"]["lower90"] > 0),
        ),
        interpretation="Same-state one-step physical effect ranking; no trained-policy or final-success claim",
    )
    result["probe_label"] = "PROMISING" if all(result["gates"].values()) else "UNPROMISING"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.record, args.output)


if __name__ == "__main__":
    main()
