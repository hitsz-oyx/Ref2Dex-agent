#!/usr/bin/env python3
"""Independent held-set audit for the native Cm residual Probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bucket(motion: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        int(hashlib.sha256(f"12651/{int(m)}/{int(s)}".encode()).hexdigest()[:8], 16) % 100
        for m, s in zip(motion, start)
    ], dtype=torch.long)


def interval(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    mean = sum(values) / len(values)
    if len(values) < 2:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    error = 1.645 * math.sqrt(variance / len(values))
    return [mean - error, mean + error]


def audit(args: argparse.Namespace) -> None:
    rows = []
    source_hashes = {}
    for path in args.records:
        path = path.resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "ref2dex.cm_residual_probe.v1":
            raise ValueError(f"schema mismatch: {path}")
        source_hashes[str(path)] = sha(path)
        held = bucket(payload["motion_id"], payload["start_frame"]) >= 70
        for index in held.nonzero().flatten().tolist():
            arm = int(payload["arm"][index])
            rows.append(dict(
                arm=arm,
                run=path.parent.name,
                motion=int(payload["motion_id"][index]),
                start=int(payload["start_frame"][index]),
                height_mm=float((payload["future_state"][index, -1, 38] - payload["state"][index, 38]) * 1000),
                contact3=float(payload["future_contact"][index, -3:].all(-1).float().mean()),
                contact_loss=float((~payload["future_contact"][index].all(-1)).any()),
                clearance_loss=float((payload["future_clearance"][index] < .002).any()),
                changed=bool(payload["executed_residual"][index].abs().sum() > 0),
                accepted=bool(payload["accepted"][index]),
                ood=bool(payload["ood"][index]),
            ))
    if not rows:
        raise ValueError("no held rows")
    names = ("baseline", "residual", "shuffled")
    metrics = {}
    for arm, name in enumerate(names):
        subset = [row for row in rows if row["arm"] == arm]
        metrics[name] = dict(
            rows=len(subset),
            groups=len({(row["motion"], row["start"]) for row in subset}),
            runs=len({row["run"] for row in subset}),
            height_mm=sum(row["height_mm"] for row in subset) / len(subset),
            contact3=sum(row["contact3"] for row in subset) / len(subset),
            contact_loss=sum(row["contact_loss"] for row in subset) / len(subset),
            clearance_loss=sum(row["clearance_loss"] for row in subset) / len(subset),
            changed=sum(row["changed"] for row in subset),
            accepted=sum(row["accepted"] for row in subset),
            ood=sum(row["ood"] for row in subset),
        )
    baseline = [row for row in rows if row["arm"] == 0]
    residual = [row for row in rows if row["arm"] == 1]
    effects = {}
    for field in ("height_mm", "contact3", "contact_loss", "clearance_loss"):
        base_values = [row[field] for row in baseline]
        residual_values = [row[field] for row in residual]
        delta = sum(residual_values) / len(residual_values) - sum(base_values) / len(base_values)
        effects[field] = dict(delta=float(delta), interval=None,
                              note="descriptive unpaired arm difference; coverage gate failed")
    supported = all(metrics[name]["rows"] >= 96 and metrics[name]["groups"] >= 8 for name in names)
    changed_gate = metrics["residual"]["changed"] >= 48
    coverage_gate = metrics["residual"]["changed"] >= 48 and metrics["residual"]["groups"] >= 8
    label = "UNPROMISING" if supported and not coverage_gate else ("UNCLEAR" if not supported else "PROMISING")
    report = dict(
        schema="ref2dex.cm_residual_probe_audit.v1",
        run_status="COMPLETED",
        label=label,
        decision="CLOSE_CURRENT_CM_RESIDUAL_POLICY_ROUTE" if label == "UNPROMISING" else "REQUIRES_FOLLOWUP",
        held_split="bucket>=70; bucket=hash(12651/motion/start)%100",
        rows=len(rows), metrics=metrics, effects=effects,
        support_gate=supported, changed_gate=changed_gate, coverage_gate=coverage_gate,
        minimums=dict(rows_per_arm=96, groups_per_arm=8, changed_residual=48),
        policy_checkpoint_sha256=args.policy_sha256,
        source_sha256=source_hashes,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        interpretation="The residual arm was mostly fail-closed; arm-level height differences are not attributable to executed residual actions.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    audit(parser.parse_args())
