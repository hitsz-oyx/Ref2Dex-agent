#!/usr/bin/env python
"""Offline representation-invariance replay for the existing wrist events.

This reads only the frozen evidence/trajectory archives from brief44/brief40.
It does not import Isaac Gym, run a model, or call a simulator.
"""
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = ROOT / "tmp/s40-deathcause/evidence.json"
ACTIVE = np.asarray((0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15))


def _pd_target(action, q, offset, scale):
    """Replay the frozen native action->PD conversion without importing it."""
    action = np.asarray(action, dtype=float)
    q = np.asarray(q, dtype=float)
    full = np.zeros(action.shape[:-1] + (18,), dtype=float)
    full[..., ACTIVE] = np.clip(action, -1.0, 1.0)
    full[..., 6:] = (1.0 + full[..., 6:]) / 2.0
    target = np.asarray(offset, dtype=float) + np.asarray(scale, dtype=float) * full
    target[..., :6] += q[..., :6]
    for dst, src, ratio in ((7, 6, 1.05), (9, 8, 1.05),
                            (11, 10, 1.05), (13, 12, 1.05),
                            (16, 15, .6), (17, 15, .8)):
        target[..., dst] = target[..., src] * ratio
    return target


def _guard(q, lower, upper, has_limits):
    """The recorded revolute/finger guard read, with its 0.10 rad margin."""
    indices = np.arange(6, 18)[has_limits[6:18]]
    values = q[indices]
    reads = np.maximum(lower[indices] - 0.10 - values,
                       values - upper[indices] - 0.10)
    return float(reads.max()), indices[reads > 0].astype(int).tolist()


def _case(record, root):
    trajectory = root / record["source_files"]["trajectory"]
    run_manifest = trajectory.parent.parent / "manifest.json"
    manifest = json.loads(run_manifest.read_text())
    audit = manifest.get("native_initial_audit")
    if audit is None:
        audit = json.loads((trajectory.parent / "native-audit.json").read_text())
    limits = audit["physical_joint_limits"]
    has_limits = np.asarray(limits["has_limits"], dtype=bool)
    lower = np.asarray(limits["lower"], dtype=float)
    upper = np.asarray(limits["upper"], dtype=float)
    assert not has_limits[3:6].any()
    offset = np.asarray(audit["controller"]["offset"], dtype=float)
    scale = np.asarray(audit["controller"]["scale"], dtype=float)
    with np.load(trajectory) as archive:
        q = np.asarray(archive["q"][:, 0], dtype=float)
        action = np.asarray(archive["action"], dtype=float)
        pd = np.asarray(archive["pd_targets_actual"], dtype=float)
    tick = int(record["terminal_tick"])
    assert tick == len(q) - 1 == len(pd)
    q_before, q_after = q[tick - 1], q[tick]
    target = pd[tick - 1, 3:6]
    # The command at state tick-1 is action[tick-1].  Recompute the full
    # native target (including finger coupling) before any representation
    # transform, and require agreement with the archived target.
    recomputed = _pd_target(action[tick - 1], q_before, offset, scale)
    assert np.allclose(recomputed, pd[tick - 1], atol=1e-5, rtol=0)
    g_before, i_before = _guard(q_before, lower, upper, has_limits)
    g_after, i_after = _guard(q_after, lower, upper, has_limits)
    assert g_before <= 0.0 and g_after > 0.0

    out = dict(batch=record["batch"], label=record["label"], tick=tick,
               wrist_dof=record["joints"][0]["index"] if record["joints"] else None,
               q_before=q_before[3:6].tolist(), q_after=q_after[3:6].tolist(),
               q_target=target.tolist(), guard_before=g_before,
               guard_after=g_after, guard_indices_before=i_before,
               guard_indices_after=i_after,
               archived_pd_replay_abs_error=float(np.max(np.abs(recomputed - pd[tick - 1]))))
    for label, multiple in (("2pi", 1), ("4pi", 2)):
        period = 2.0 * math.pi * multiple
        unwrapped = np.unwrap(q[:, 3:6], axis=0, period=period)
        q_u_before, q_u_after = unwrapped[tick - 1], unwrapped[tick]
        # The target is moved by exactly the same branch offset as q at
        # command time. This is the physical-equivalence transformation.
        shift_before = q_u_before - q_before[3:6]
        shift_after = q_u_after - q_after[3:6]
        target_u_before = target + shift_before
        target_u_after = target + shift_after
        recomputed_u = _pd_target(action[tick - 1],
                                  np.r_[q_before[:3], q_u_before, q_before[6:]],
                                  offset, scale)
        assert np.allclose(recomputed_u[3:6], target_u_before, atol=1e-5, rtol=0)
        guard_u_before, i_u_before = _guard(
            np.r_[q_before[:3], q_u_before, q_before[6:]], lower, upper,
            has_limits)
        guard_u_after, i_u_after = _guard(
            np.r_[q_after[:3], q_u_after, q_after[6:]], lower, upper,
            has_limits)
        e_cmd = target - q_before[3:6]
        e_cmd_u = target_u_before - q_u_before
        e_post = target - q_after[3:6]
        e_post_u = target_u_after - q_u_after
        assert np.allclose(e_cmd, e_cmd_u, atol=1e-12, rtol=0)
        assert np.allclose(e_post, e_post_u, atol=1e-12, rtol=0)
        assert (guard_u_before, i_u_before) == (g_before, i_before)
        assert (guard_u_after, i_u_after) == (g_after, i_after)
        out[label] = dict(
            shift_before=shift_before.tolist(), shift_after=shift_after.tolist(),
            q_before=q_u_before.tolist(), q_after=q_u_after.tolist(),
            q_target=target_u_before.tolist(),
            q_target_post=target_u_after.tolist(), guard_before=guard_u_before,
            guard_after=guard_u_after, guard_indices_before=i_u_before,
            guard_indices_after=i_u_after,
            command_error_before=e_cmd.tolist(),
            command_error_after=e_cmd_u.tolist(),
            command_error_norm=float(np.linalg.norm(e_cmd_u)),
            post_error_before=e_post.tolist(),
            post_error_after=e_post_u.tolist(),
            post_error_norm=float(np.linalg.norm(e_post_u)),
            abort_before=True, abort_after=True,
        )
    return out


def analyze(root=ROOT):
    evidence = json.loads((root / "tmp/s40-deathcause/evidence.json").read_text())
    records = [r for r in evidence["records"]
               if r["signature"] == "physical_joint_limit"]
    assert len(records) == 21
    cases = [_case(r, root) for r in records]
    summary = {
        "cases": len(cases),
        "period_2pi_changed_guard": sum(c["2pi"]["abort_after"] != c["2pi"]["abort_before"] for c in cases),
        "period_2pi_changed_pd_error": sum(
            not np.allclose(c["2pi"]["command_error_before"], c["2pi"]["command_error_after"], atol=1e-12, rtol=0)
            or not np.allclose(c["2pi"]["post_error_before"], c["2pi"]["post_error_after"], atol=1e-12, rtol=0)
            for c in cases),
        "period_4pi_changed_guard": sum(c["4pi"]["abort_after"] != c["4pi"]["abort_before"] for c in cases),
        "period_4pi_changed_pd_error": sum(
            not np.allclose(c["4pi"]["command_error_before"], c["4pi"]["command_error_after"], atol=1e-12, rtol=0)
            or not np.allclose(c["4pi"]["post_error_before"], c["4pi"]["post_error_after"], atol=1e-12, rtol=0)
            for c in cases),
        "max_archived_pd_replay_abs_error": max(
            c["archived_pd_replay_abs_error"] for c in cases),
        "cases": cases,
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, ensure_ascii=False))
