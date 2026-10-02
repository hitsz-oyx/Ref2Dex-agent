#!/usr/bin/env python3
"""Fit a decision-time-only relative task-value head on frozen HF16 records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from probe_relative_task_value_calibration import ridge_fit, split_group, summarize


def current_features(record):
    diagnostic = record["diagnostics"]
    program = record["program"][:, 0].long()
    score = diagnostic["score_mm"][:, 0, 0]
    fixed = score[:, 7]
    selected = score.gather(-1, program[:, None]).squeeze(-1)
    selected_std = diagnostic["relative_std_mm"][:, 0, 0].gather(-1, program[:, None]).squeeze(-1)
    fixed_std = diagnostic["relative_std_mm"][:, 0, 0, 7]
    selected_retention = diagnostic["retention"][:, 0, 0].gather(-1, program[:, None]).squeeze(-1)
    fixed_retention = diagnostic["retention"][:, 0, 0, 7]
    selected_release = diagnostic["release"][:, 0, 0].gather(-1, program[:, None]).squeeze(-1)
    fixed_release = diagnostic["release"][:, 0, 0, 7]
    relative = selected - fixed
    height = record["state"][:, 38] - record["rest_z"]
    contact = record["history"][:, -1, 49:51].bool().all(-1).float()
    x = torch.stack([
        relative, relative, relative, relative,
        selected_std, selected_std - fixed_std,
        selected_retention - fixed_retention, selected_release - fixed_release,
        fixed, height, record["initial_clearance"], contact,
    ], -1)
    target = record["outcome"]["score_mm"] - fixed
    return x, target, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.records:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("schema") != "ref2dex.native_pd_closed_loop.v1":
            raise ValueError(f"unexpected HF16 schema: {path}")
        x, y, raw = current_features(record)
        rows.append(dict(x=x, y=y, raw=raw, motion=record["motion_id"], start=record["start_frame"]))
    data = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    bucket = split_group(data["motion"], data["start"])
    fit, cal, held = bucket < 4, (bucket >= 4) & (bucket < 6), bucket >= 6
    groups = torch.unique(torch.stack((data["motion"][held], data["start"][held]), -1), dim=0)
    if int(held.sum()) < 100 or len(groups) < 8:
        raise ValueError("held support below current-step contract")
    bundle = ridge_fit(data["x"][fit], data["y"][fit], lam=1.0)
    from probe_relative_task_value_calibration import predict
    prediction = predict(bundle, data["x"])
    reports = {name: summarize(prediction, data["raw"], data["y"], mask)
               for name, mask in (("fit", fit), ("cal", cal), ("held", held))}
    held_report = reports["held"]
    improvement = dict(
        spearman_delta=held_report["spearman_calibrated"] - held_report["spearman_raw"],
        rmse_ratio=held_report["rmse_calibrated"] / max(held_report["rmse_raw"], 1e-8),
    )
    gates = dict(support=True, spearman_gain=improvement["spearman_delta"] >= .10,
                 rmse_gain=improvement["rmse_ratio"] <= .90)
    label = "PROMISING" if all(gates.values()) else "UNPROMISING"
    output = dict(
        schema="ref2dex.current_decision_task_value.v1", run_status="COMPLETED",
        experiment_id="P-20261003-current-decision-task-value", label=label,
        rows=len(data["y"]), split_rows={"fit": int(fit.sum()), "cal": int(cal.sum()), "held": int(held.sum())},
        held_groups=int(len(groups)), reports=reports, improvement=improvement,
        gates=gates, ridge_lambda=1.0,
        input_records=[str(p.resolve()) for p in args.records],
        input_checkpoint=str(args.checkpoint.resolve()),
        input_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        target="actual H10 retained supported height minus current decision-time frozen fixed-Cup prediction",
        feature_mode="current decision only; no post-intervention suffix",
        scope="decision-time relative task-value calibration only; no success classifier, native control, or PPO",
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    torch.save(dict(schema=output["schema"], bundle={key: value.cpu() for key, value in bundle.items()},
                    input_checkpoint=output["input_checkpoint"], input_checkpoint_sha256=output["input_checkpoint_sha256"],
                    feature_mode=output["feature_mode"]), args.output.with_suffix(".pt"))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
