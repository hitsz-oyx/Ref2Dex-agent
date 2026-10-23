#!/usr/bin/env python3
"""Offline wrist representation invariance replay for brief46.

This reads the frozen brief44 evidence and trajectories only.  It does not
instantiate Isaac Gym, train a model, or call a physics step.  The report
consumer uses the JSON printed on stdout; no output file is created here.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = ROOT / "tmp/s40-deathcause/evidence.json"
COLLECT = ROOT / "tmp/s40-deathcause/collect.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("brief44_collect", COLLECT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finger_gate(q, limits):
    mask = np.asarray(limits["has_limits"], dtype=bool)
    lower = np.asarray(limits["lower"], dtype=float)
    upper = np.asarray(limits["upper"], dtype=float)
    ids = np.arange(6, 18)[mask[6:18]]
    return float(np.maximum(lower[ids] - 0.10 - q[ids], q[ids] - upper[ids] - 0.10).max())


def main():
    helpers = load_helpers()
    prior = json.loads(EVIDENCE.read_text())
    rows = []
    for record in prior["records"]:
        if record.get("signature") != "physical_joint_limit":
            continue
        trajectory = ROOT / record["source_files"]["trajectory"]
        run = trajectory.parent.parent
        manifest = helpers.read(run / "manifest.json")
        audit = manifest.get("native_initial_audit") or helpers.read(trajectory.parent / "native-audit.json")
        limits = audit["physical_joint_limits"]
        with np.load(trajectory) as z:
            q = z["q"][:, 0].astype(float)
            pd = z["pd_targets_actual"].astype(float)
        # The final state is the recorded abort state.  pd[-1] is the target
        # sent during the preceding command interval, whose measured state is
        # q[-2].  This is the same timing convention used by brief44.
        q_pre = q[-2, 3:6]
        q_abort = q[-1, 3:6]
        target = pd[-1, 3:6]
        raw_error = target - q_pre
        unwrapped = np.unwrap(q[:, 3:6], axis=0, period=2 * np.pi)
        pre_shift = unwrapped[-2] - q_pre
        abort_shift = unwrapped[-1] - q_abort
        target_u = target + pre_shift
        q_pre_u = q_pre + pre_shift
        q_abort_u = q_abort + abort_shift
        rows.append(
            {
                "batch": record["batch"],
                "label": record["label"],
                "tick": int(record["terminal_tick"]),
                "trigger_dofs": record["final_safety"]["joint_violation_indices"],
                "q_pre": q_pre.tolist(),
                "q_abort": q_abort.tolist(),
                "q_pre_u": q_pre_u.tolist(),
                "q_abort_u": q_abort_u.tolist(),
                "q_target": target.tolist(),
                "q_target_u": target_u.tolist(),
                "guard_pre": finger_gate(q[-2], limits),
                "guard_abort": finger_gate(q[-1], limits),
                "guard_pre_u": finger_gate(np.r_[q[-2, :3], q_pre_u, q[-2, 6:]], limits),
                "guard_abort_u": finger_gate(np.r_[q[-1, :3], q_abort_u, q[-1, 6:]], limits),
                "pd_error": raw_error.tolist(),
                "pd_error_u": (target_u - q_pre_u).tolist(),
                "pd_error_norm": float(np.linalg.norm(raw_error)),
                "pd_error_u_norm": float(np.linalg.norm(target_u - q_pre_u)),
                "abort": True,
                "abort_u": True,
            }
        )
    assert len(rows) == 21, len(rows)
    changed = sum(r["abort"] != r["abort_u"] for r in rows)
    pd_changed = sum(not np.allclose(r["pd_error"], r["pd_error_u"], atol=1e-12, rtol=0) for r in rows)
    return {
        "scope": "21 existing physical_joint_limit abort records; wrist-only equivalent remapping",
        "n": len(rows),
        "period": "2pi",
        "changed_guard_decisions": changed,
        "changed_guard_fraction": changed / len(rows),
        "changed_pd_errors": pd_changed,
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
