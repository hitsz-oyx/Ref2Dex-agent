"""Audit phase-specific lift/contact effects in the duck randomized Probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_randomized_followup import effects


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "outputs/CmResidual/agent_duck_contact_action_s210_h10_n64/transitions.pt"
BINS = {"early": (50, 90), "middle": (100, 140), "late": (150, 190)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.randomized_action_followup.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("followup_horizon") != 10):
        raise ValueError("wrong or incomplete randomized H10 record")
    record = payload["records"]
    steps = record["global_step"].numpy()
    assignment = record["assignment"].numpy()
    if (len(steps) != 15 * 64 or not np.array_equal(np.unique(steps),
                                                     np.arange(50, 191, 10)) or
            not np.all(np.isin(assignment, [-1, 0, 1]))):
        raise ValueError("unexpected assignment layout")
    values = {
        "contact_supported_dz_mm": (
            (record["followup_object_state"][:, 2] -
             record["object_state"][:, 2]).numpy() * 1000 *
            record["followup_contact_count"].numpy() / 10),
        "contact_fraction": record["followup_contact_count"].numpy() / 10,
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("nonfinite followup outcome")
    bins = {}
    for name, (start, end) in BINS.items():
        mask = (steps >= start) & (steps <= end)
        if mask.sum() != 5 * 64:
            raise ValueError(f"incomplete {name} phase")
        counts = {"plus": int((assignment[mask] == 1).sum()),
                  "minus": int((assignment[mask] == -1).sum())}
        if min(counts.values()) < 40:
            raise ValueError(f"insufficient arm support in {name} phase")
        estimate = effects({key: value[mask] for key, value in values.items()},
                           assignment[mask], steps[mask], 64,
                           bootstraps=500, seed=24211)
        bins[name] = {"steps": list(range(start, end + 1, 10)),
                      "treated": sum(counts.values()), "arm_counts": counts,
                      "effects": estimate}
    late = bins["late"]
    passed = (late["treated"] >= 100 and
              late["effects"]["contact_supported_dz_mm"]["plus_minus"] >= 10 and
              late["effects"]["contact_fraction"]["plus_minus"] >= 0)
    report = {"experiment_id": "P-20260925-duck-contact-phase-audit",
              "source": str(SOURCE), "source_seed": 210,
              "bins": bins, "late_phase_gate_pass": bool(passed),
              "interpretation_boundary": "post hoc phase question on one randomized seed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"late_phase_gate_pass": passed,
                      "late": late}, sort_keys=True))


if __name__ == "__main__":
    main()
