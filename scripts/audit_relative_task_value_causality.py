#!/usr/bin/env python3
"""Audit retrospective HF26 features against the native decision-time contract.

This is an artifact/feature audit, not a new policy-selection experiment.
Small double-precision ridge solves are statistical controls, run on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from probe_relative_task_value_calibration import features, predict, ridge_fit, split_group


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def at_decision(record):
    """Construct the original head's t=0 inputs without reading any suffix."""
    return {
        **{k: record[k] for k in ("state", "rest_z", "history", "initial_clearance")},
        "program": record["program"][:, :1].expand(-1, 10),
        "diagnostics": {k: v[:, :1].expand(-1, 10, -1, -1)
                        for k, v in record["diagnostics"].items()},
        # `features` returns the supervised target separately; not part of x.
        "outcome": record["outcome"],
    }


def metrics(prediction, target):
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite audit statistic")
    return {"rmse_mm": float((prediction-target).square().mean().sqrt()),
            "spearman": float(spearmanr(prediction.numpy(), target.numpy()).statistic)}


def run(args):
    if args.output.exists():
        raise ValueError("audit output must be new")
    torch.set_num_threads(2)
    rows = []
    poisoned = []
    for path in args.records:
        record = torch.load(path, map_location="cpu", weights_only=False)
        full, target, raw, fixed = features(record)
        causal, causal_target, causal_raw, causal_fixed = features(at_decision(record))
        changed = dict(record)
        changed["diagnostics"] = {k: v.clone() for k, v in record["diagnostics"].items()}
        changed["program"] = record["program"].clone()
        for value in changed["diagnostics"].values():
            value[:, 1:] = 123.0
        changed["program"][:, 1:] = 0
        poisoned.append({
            "path": str(path),
            "retrospective_features_change": bool((features(changed)[0] != full).any()),
            "decision_features_unchanged": torch.equal(features(at_decision(changed))[0], causal),
        })
        # current state, native observation, and all candidate predictions at
        # offset 0 precede the first physical intervention.
        rows.append(dict(full=full.double(), causal=causal.double(), target=target.double(),
                         causal_target=causal_target.double(), raw=raw.double(), fixed=fixed.double(),
                         causal_raw=causal_raw.double(), causal_fixed=causal_fixed.double(),
                         outcome=record["outcome"]["score_mm"].double(),
                         assignment=record["assignment"],
                         motion=record["motion_id"], start=record["start_frame"]))
    data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
    bucket = split_group(data["motion"], data["start"])
    fit, held = bucket < 4, bucket >= 6
    if fit.sum() == 0 or held.sum() == 0:
        raise ValueError("empty partition")
    retrospective = {}
    for name, columns in (("full", list(range(12))), ("common_fixed_only", [8]),
                          ("common_fixed_and_initial_state", [8, 9, 10, 11])):
        x = data["full"][:, columns]
        head = ridge_fit(x[fit], data["target"][fit])
        retrospective[name] = metrics(predict(head, x)[held], data["target"][held])

    causal = {}
    # All models below are compared on the same actual H10 signed supported
    # height change.  No model-conditioned subtraction shared with the target.
    for name, columns in (("current_state", [9, 10, 11]),
                          ("common_fixed_and_state", [8, 9, 10, 11]),
                          ("cm_first_decision", list(range(12)))):
        x = data["causal"][:, columns]
        head = ridge_fit(x[fit], data["outcome"][fit])
        causal[name] = metrics(predict(head, x)[held], data["outcome"][held])
    original = torch.load(args.head, map_location="cpu", weights_only=False)["bundle"]
    original = {k: v.double() for k, v in original.items()}
    causal["deployed_head_at_start"] = metrics(
        (predict(original, data["causal"])+data["causal_fixed"])[held], data["outcome"][held])
    causal["raw_cm_first_step"] = metrics(
        (data["causal_raw"]+data["causal_fixed"])[held], data["outcome"][held])
    counts = {name: {"rows": int(mask.sum()),
                     "groups": len(set(zip(data["motion"][mask].tolist(), data["start"][mask].tolist())))}
              for name, mask in (("fit", fit), ("held", held))}
    result = {
        "schema": "ref2dex.relative_task_value_causality_audit.v1",
        "run_status": "COMPLETED", "label": "UNCLEAR",
        "decision_gate_valid": False, "allow_native_followup": False,
        "rows": len(data["full"]), "split": counts,
        "retrospective_target_metrics": retrospective,
        "decision_time_actual_outcome_metrics": causal,
        "future_poison": poisoned,
        "reasons": [
            "Original x averages predictions on actual post-intervention states at offsets 1--9.",
            "Online suffix repetition was never the distribution used to fit or hold out the head.",
            "Original target subtracts mean fixed prediction on the same post-intervention trajectory; it is not Q_fixed(s0).",
            "Native panel restricts old physical-model held groups only, and includes relative-head fit groups.",
            "Pooled factual Spearman does not identify within-state candidate ranking or value beyond direct-Q.",
        ],
        "input_records_sha256": {str(p.resolve()): sha(p) for p in args.records},
        "head_sha256": sha(args.head), "script_sha256": sha(Path(__file__)),
        "device": "cpu", "device_reason": "12-column double-precision statistical solves and label audit; no neural model computation",
        "scope": "Revokes retrospective calibration as a decision gate; current-only factual metrics are diagnostics, not causal action advantage.",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("label", "split", "retrospective_target_metrics", "decision_time_actual_outcome_metrics")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
