#!/usr/bin/env python3
"""Fit-only relative task-value calibration on frozen HF16 Cm predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def split_group(motion: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    # The first draft used modulo 100; all 28 groups in this sample then fell
    # in 70--99, leaving no fit/cal rows.  Keep the same fixed group hash but
    # use its decimal bucket so the predeclared three-way split is populated.
    values = [int(hashlib.sha256(f"9851/{int(m)}/{int(s)}".encode()).hexdigest()[:8], 16) % 10
              for m, s in zip(motion.cpu(), start.cpu())]
    return torch.tensor(values, dtype=torch.long)


def features(record):
    diagnostic = record["diagnostics"]
    program = record["program"].long()
    batch = torch.arange(len(program))[:, None]
    step = torch.arange(program.shape[1])[None, :]
    selected = diagnostic["score_mm"][:, :, 0].gather(-1, program[..., None]).squeeze(-1)
    fixed = diagnostic["score_mm"][:, :, 0, 7]
    selected_std = diagnostic["relative_std_mm"][:, :, 0].gather(-1, program[..., None]).squeeze(-1)
    fixed_std = diagnostic["relative_std_mm"][:, :, 0, 7]
    selected_retention = diagnostic["retention"][:, :, 0].gather(-1, program[..., None]).squeeze(-1)
    fixed_retention = diagnostic["retention"][:, :, 0, 7]
    selected_release = diagnostic["release"][:, :, 0].gather(-1, program[..., None]).squeeze(-1)
    fixed_release = diagnostic["release"][:, :, 0, 7]
    relative = selected - fixed
    relative_std = selected_std - fixed_std
    relative_retention = selected_retention - fixed_retention
    relative_release = selected_release - fixed_release
    height = record["state"][:, 38] - record["rest_z"]
    contact = record["history"][:, -1, 49:51].bool().all(-1).float()
    clearance = record["initial_clearance"]
    values = [relative.mean(-1), relative[:, -1], relative.amin(-1), relative.amax(-1),
              selected_std.mean(-1), relative_std.mean(-1), relative_retention.mean(-1),
              relative_release.mean(-1), fixed.mean(-1), height, clearance, contact]
    x = torch.stack(values, -1)
    # The head is relative to frozen fixed-Cup model value; no success label is
    # used.  This is a continuous task-utility calibration target.
    target = record["outcome"]["score_mm"] - fixed.mean(-1)
    return x, target, relative.mean(-1), fixed.mean(-1)


def load(paths):
    rows = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("schema") != "ref2dex.native_pd_closed_loop.v1":
            raise ValueError(f"unexpected HF16 schema: {path}")
        x, y, raw, fixed = features(data)
        rows.append(dict(x=x, y=y, raw=raw, fixed=fixed,
                         assignment=data["assignment"].long(),
                         motion=data["motion_id"].long(), start=data["start_frame"].long()))
    return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}


def ridge_fit(x, y, lam=1.0):
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(.001)
    xm = (x - mean) / std
    y_mean = y.mean()
    yc = y - y_mean
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    beta = torch.linalg.solve(xm.T @ xm + lam * eye, xm.T @ yc)
    return dict(mean=mean, std=std, beta=beta, y_mean=y_mean)


def predict(bundle, x):
    # Parenthesize the normalization before the matrix product.  Without this,
    # Python's operator precedence makes the expression ambiguous and can turn
    # the intended per-feature scaling into a different matmul expression.
    normalized = (x - bundle["mean"]) / bundle["std"]
    return normalized @ bundle["beta"] + bundle["y_mean"]


def spearman(x, y):
    # Average ranks are unnecessary for this diagnostic because ties are rare;
    # sorting twice gives deterministic ranks without a scipy dependency.
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    return float(torch.corrcoef(torch.stack((rx, ry)))[0, 1])


def summarize(pred, raw, target, mask):
    pred, raw, target = pred[mask], raw[mask], target[mask]
    return dict(rows=int(mask.sum()), spearman_calibrated=spearman(pred, target),
                spearman_raw=spearman(raw, target),
                rmse_calibrated=float((pred - target).square().mean().sqrt()),
                rmse_raw=float((raw - target).square().mean().sqrt()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    data = load(args.records)
    bucket = split_group(data["motion"], data["start"])
    fit, cal, held = bucket < 4, (bucket >= 4) & (bucket < 6), bucket >= 6
    if int(held.sum()) < 100 or int(torch.unique(torch.stack((data["motion"][held], data["start"][held]), -1), dim=0).shape[0]) < 8:
        raise ValueError("held support is below the predeclared calibration contract")
    x = data["x"].to(device); y = data["y"].to(device); raw = data["raw"].to(device)
    bundle = ridge_fit(x[fit.to(device)], y[fit.to(device)], lam=1.0)
    pred = predict(bundle, x)
    reports = {name: summarize(pred, raw, y, mask.to(device))
               for name, mask in (("fit", fit), ("cal", cal), ("held", held))}
    held_report = reports["held"]
    improvement = dict(spearman_delta=held_report["spearman_calibrated"] - held_report["spearman_raw"],
                       rmse_ratio=held_report["rmse_calibrated"] / max(held_report["rmse_raw"], 1e-8))
    gates = dict(support=True, spearman_gain=improvement["spearman_delta"] >= .10,
                 rmse_gain=improvement["rmse_ratio"] <= .90)
    label = "PROMISING_CALIBRATION" if all(gates.values()) else "UNPROMISING"
    output = dict(schema="ref2dex.relative_task_value_calibration.v1", run_status="COMPLETED",
                  experiment_id="P-20261003-relative-task-value-calibration", label=label,
                  rows=len(y), split_rows={"fit": int(fit.sum()), "cal": int(cal.sum()), "held": int(held.sum())},
                  held_groups=int(torch.unique(torch.stack((data["motion"][held], data["start"][held]), -1), dim=0).shape[0]),
                  reports=reports, improvement=improvement, gates=gates, ridge_lambda=1.0,
                  input_records=[str(p.resolve()) for p in args.records],
                  input_checkpoint=str(args.checkpoint.resolve()),
                  input_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  target="actual H10 retained supported height minus frozen Cm fixed-Cup prediction",
                  scope="relative task-value calibration only; no success classifier, native control, or PPO")
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    torch.save(dict(schema=output["schema"], bundle={key: value.cpu() for key, value in bundle.items()},
                    input_checkpoint=output["input_checkpoint"], input_checkpoint_sha256=output["input_checkpoint_sha256"],
                    features="frozen Cm candidate score/uncertainty/retention/release relative to fixed-Cup"),
              args.output.with_suffix(".pt"))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
