"""Audit the first saved joint-limit anomaly without running a simulator.

The input runs are immutable rollout artifacts.  This tool deliberately keeps
the source/A0, WAM/A1 and source-ceiling R4 execution chains separate.  It
reports the first out-of-limit state, the preceding command and contact/force
evidence; it does not treat a failed numerical IK solve or a limit label as a
mechanism proof.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


DT = 1.0 / 30.0
CONTACT_FORCE_N = 1.0
OUTWARD_EPS = 1e-6
VIOLATION_EPS = 1e-6
DEFAULT_DOF_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
]


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _first_true(values):
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(indices[0]) if len(indices) else None


def _first_force(z, key, valid):
    if key == "object":
        force = np.linalg.norm(z["object_contact_force"][:, 0], axis=-1)
    else:
        force = np.linalg.norm(z["contact_forces"][:, 0], axis=-1).max(axis=-1)
    mask = np.asarray(valid, dtype=bool) & (force >= CONTACT_FORCE_N)
    return _first_true(mask), force


def _limit_state(z, limits):
    q = z["q"][:, 0]
    has = np.asarray(limits["has_limits"], dtype=bool)
    lower = np.asarray(limits["lower"], dtype=float)
    upper = np.asarray(limits["upper"], dtype=float)
    excess = np.maximum(np.maximum(lower[None, :] - q, q - upper[None, :]), 0.0)
    excess[:, ~has] = 0.0
    indices = np.flatnonzero((excess > VIOLATION_EPS).any(axis=1))
    return (int(indices[0]) if len(indices) else None), excess, lower, upper, has


def _command_direction(target, q, lower, upper):
    if q < lower:
        direction = "lower"
        outward = target < lower - OUTWARD_EPS
        at_limit = abs(target - lower) <= OUTWARD_EPS
    elif q > upper:
        direction = "upper"
        outward = target > upper + OUTWARD_EPS
        at_limit = abs(target - upper) <= OUTWARD_EPS
    else:
        direction = "unknown"
        outward = False
        at_limit = False
    return direction, bool(outward), bool(at_limit)


def _contact_evidence(z, consumer):
    valid = np.asarray(z["contact_force_valid"][:, 0], dtype=bool)
    measured = np.asarray([bool(x.get("measured_contact", False)) for x in consumer], dtype=bool)
    measured_first = _first_true(measured)
    object_first, object_force = _first_force(z, "object", valid)
    hand_first, hand_force = _first_force(z, "hand", valid)
    evidence = [i for i in (measured_first, object_first, hand_first) if i is not None]
    first = min(evidence) if evidence else None
    return dict(
        measured_contact_first=measured_first,
        measured_contact_first_s=None if measured_first is None else measured_first * DT,
        object_force_first_ge_1n=object_first,
        object_force_first_ge_1n_s=None if object_first is None else object_first * DT,
        hand_force_first_ge_1n=hand_first,
        hand_force_first_ge_1n_s=None if hand_first is None else hand_first * DT,
        first_available_contact_evidence=first,
        first_available_contact_evidence_s=None if first is None else first * DT,
        object_force_max_n=float(np.max(object_force)) if len(object_force) else 0.0,
        hand_force_max_n=float(np.max(hand_force)) if len(hand_force) else 0.0,
    )


def _stage_first(stages, name):
    for i, row in enumerate(stages):
        if bool(row.get("achieved", {}).get(name, False)):
            return i
    return None


def _published_death(row):
    if row.get("final_success"):
        return "success"
    if row.get("physical_abort"):
        return row["physical_abort"]
    stage = row.get("stage_achieved", {})
    if not stage.get("contact"):
        return "stage_contact_not_certified"
    if not stage.get("lift"):
        return "contact_without_lift"
    if not stage.get("hold"):
        return "lift_without_continuous_hold"
    return "held_off_goal" if row.get("final_hold") else "held_then_lost"


def audit_row(row, folder):
    folder = Path(folder)
    audit = json.loads((folder / "native-audit.json").read_text())
    consumer = json.loads((folder / "consumer.json").read_text())
    stages_path = folder / "stages.json"
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else []
    with np.load(folder / "trajectory.npz") as z:
        dof_names = audit.get("native_dof_names", audit.get("dof_names", DEFAULT_DOF_NAMES))
        limit_tick, excess, lower, upper, has = _limit_state(z, audit["physical_joint_limits"])
        contact = _contact_evidence(z, consumer)
        q = z["q"][:, 0]
        dq = z["dq"][:, 0]
        controls = int(len(z["action"]))
        if limit_tick is None:
            return dict(
                id=row["id"], trial=row["trial"], sequence=row["sequence"], object=row["object"],
                arm=row["arm"], controls=controls, physical_abort=row.get("physical_abort"),
                first_limit_tick=None, first_limit_time_s=None,
                final_success=bool(row.get("final_success", False)),
                ever_success=bool(row.get("ever_success", False)),
                stage_achieved=row.get("stage_achieved", {}), contact=contact,
                phase_first_limit=None, no_limit_record=True,
                recorded_limit_label=row.get("physical_abort") == "physical_joint_limit",
                published_death=_published_death(row),
            )

        affected = np.flatnonzero(excess[limit_tick] > VIOLATION_EPS).tolist()
        command_tick = min(max(limit_tick - 1, 0), len(z["pd_targets_actual"]) - 1)
        phase_index = min(command_tick, len(consumer) - 1)
        phase = consumer[phase_index].get("phase") if consumer else None
        pd = np.asarray(z["pd_targets_actual"])
        raw = np.asarray(z["raw_action"])
        adapter = np.asarray(z["adapter_raw_action"])
        action = np.asarray(z["action"])
        window = range(max(0, command_tick - 4), command_tick + 1)
        joint_rows = []
        for j in affected:
            direction, outward, at_limit = _command_direction(
                float(pd[command_tick, j]), float(q[limit_tick, j]), lower[j], upper[j]
            )
            outward_window = []
            for k in window:
                d, o, _ = _command_direction(float(pd[k, j]), float(q[min(k, len(q) - 1), j]), lower[j], upper[j])
                outward_window.append(bool(o or (d == "lower" and pd[k, j] < lower[j] - OUTWARD_EPS) or (d == "upper" and pd[k, j] > upper[j] + OUTWARD_EPS)))
            joint_rows.append(dict(
                index=int(j), name=dof_names[j],
                direction=direction, lower=float(lower[j]), upper=float(upper[j]),
                q_at_first_violation=float(q[limit_tick, j]),
                dq_at_first_violation=float(dq[limit_tick, j]),
                signed_margin_rad=float(min(q[limit_tick, j] - lower[j], upper[j] - q[limit_tick, j])),
                excess_rad=float(excess[limit_tick, j]),
                pd_target_at_prelimit=float(pd[command_tick, j]),
                target_outward_at_prelimit=outward,
                target_at_limit=at_limit,
                target_status="outward" if outward else ("at_limit" if at_limit else "legal_inward_or_neutral"),
                outward_target_count_last5=int(sum(outward_window)),
                outward_target_fraction_last5=float(np.mean(outward_window)),
                q_prelimit=float(q[command_tick, j]),
                dq_prelimit=float(dq[command_tick, j]),
            ))

        reach = consumer[phase_index].get("reachability") if consumer else None
        stage_first = {name: _stage_first(stages, name) for name in ("approach", "contact", "lift", "hold", "release")}
        first_contact = contact["first_available_contact_evidence"]
        contact_relation = (
            "contact_before_first_limit" if first_contact is not None and first_contact < limit_tick
            else "contact_only_at_or_after_first_limit" if first_contact is not None
            else "no_contact_evidence_in_saved_trace"
        )
        reached_before = [name for name, tick in stage_first.items() if tick is not None and tick < limit_tick]
        tau_valid = np.asarray(z["tau_valid"], dtype=bool)
        tau_at = None
        if bool(tau_valid[min(limit_tick, len(tau_valid) - 1)]):
            tau_at = np.asarray(z["tau"][min(limit_tick, len(z["tau"]) - 1)]).tolist()
        return dict(
            id=row["id"], trial=row["trial"], sequence=row["sequence"], object=row["object"], arm=row["arm"],
            controls=controls, physical_abort=row.get("physical_abort"), final_success=bool(row.get("final_success", False)),
            ever_success=bool(row.get("ever_success", False)), stage_achieved=row.get("stage_achieved", {}),
            recorded_limit_label=row.get("physical_abort") == "physical_joint_limit",
            published_death=_published_death(row),
            first_limit_tick=limit_tick, first_limit_time_s=limit_tick * DT, command_tick=command_tick,
            command_time_s=command_tick * DT, phase_first_limit=phase, contact_relation=contact_relation,
            contact=contact, stage_first_ticks=stage_first, stages_reached_before_first_limit=reached_before,
            joints=joint_rows, reachability=reach,
            native_target_admitted=consumer[phase_index].get("native_target_admitted") if consumer else None,
            wrist_target=consumer[phase_index].get("wrist_target") if consumer else None,
            geometry=consumer[phase_index].get("geometry") if consumer else None,
            action_at_prelimit=action[command_tick].tolist() if action.ndim == 2 else None,
            raw_action_at_prelimit=raw[command_tick].tolist() if raw.ndim == 2 else None,
            adapter_raw_action_at_prelimit=adapter[command_tick].tolist() if adapter.ndim == 2 else None,
            tau_valid_at_first_limit=bool(tau_valid[min(limit_tick, len(tau_valid) - 1)]),
            tau_valid_count=int(tau_valid.sum()), tau_at_first_limit=tau_at,
            tau_shape=list(z["tau"].shape[1:]),
            available_trace_window=dict(
                state_ticks=[int(k) for k in range(max(0, limit_tick - 5), min(len(q), limit_tick + 2))],
                q_before=[q[k].tolist() for k in range(max(0, limit_tick - 5), min(len(q), limit_tick + 2))],
                dq_before=[dq[k].tolist() for k in range(max(0, limit_tick - 5), min(len(q), limit_tick + 2))],
            ),
        )


def load_rows(root, predicate):
    rows = []
    for manifest in sorted(Path(root).glob("g*/manifest.json")):
        data = json.loads(manifest.read_text())
        for row in data.get("runs", []):
            if predicate(row):
                rows.append((row, manifest.parent / row["id"]))
    return rows


def summarize_limits(records):
    labelled = [r for r in records if r.get("recorded_limit_label")]
    limited = [r for r in labelled if r.get("first_limit_tick") is not None]
    phase = Counter(r["phase_first_limit"] for r in limited)
    relation = Counter(r["contact_relation"] for r in limited)
    joint = Counter(j["name"] for r in limited for j in r["joints"])
    direction = Counter((j["name"], j["direction"]) for r in limited for j in r["joints"])
    target = Counter(j["target_status"] for r in limited for j in r["joints"])
    reached = sum(bool(r["stages_reached_before_first_limit"]) for r in limited)
    return dict(
        n=len(records), recorded_limit_labels=len(labelled), limit_records=len(limited),
        no_recorded_limit_records=len(records) - len(labelled),
        saved_q_violation_records=sum(r.get("first_limit_tick") is not None for r in records),
        first_limit_phase=dict(phase), contact_relation=dict(relation), first_limit_joint=dict(joint),
        first_limit_joint_direction={f"{k[0]}::{k[1]}": v for k, v in direction.items()},
        target_status=dict(target), reached_stage_before_first_limit=reached,
        target_outward_joint_count=sum(j["target_status"] == "outward" for r in limited for j in r["joints"]),
        target_at_limit_joint_count=sum(j["target_status"] == "at_limit" for r in limited for j in r["joints"]),
        target_legal_inward_or_neutral_joint_count=sum(j["target_status"] == "legal_inward_or_neutral" for r in limited for j in r["joints"]),
    )


def summarize_r4(records):
    phase = Counter(r.get("phase_first_limit") for r in records)
    stage_contact = sum(bool(r.get("stage_achieved", {}).get("contact", False)) for r in records)
    ever_contact = sum(r.get("contact", {}).get("first_available_contact_evidence") is not None for r in records)
    ever_close = sum(r.get("ever_close", False) for r in records)
    return dict(
        n=len(records), final_success=sum(r.get("final_success", False) for r in records),
        ever_success=sum(r.get("ever_success", False) for r in records),
        controls_total=sum(r.get("controls", 0) for r in records),
        controls_unique=sorted(set(r.get("controls", 0) for r in records)),
        published_death=dict(Counter(r.get("published_death") for r in records)),
        final_phase=dict(Counter(r.get("final_phase") for r in records)),
        stage_contact_achieved=stage_contact, stage_contact_not_achieved=len(records) - stage_contact,
        any_contact_force_or_measured=ever_contact, any_close_phase=ever_close,
        no_close_and_no_contact_evidence=sum(not r.get("ever_close", False) and r.get("contact", {}).get("first_available_contact_evidence") is None for r in records),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wam-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--r4-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wam_rows = load_rows(args.wam_run, lambda r: r.get("arm") == "A1" and r.get("subset") == "unseen")
    source_rows = load_rows(args.source_run, lambda r: r.get("arm") == "A0")
    r4_rows = load_rows(args.r4_run, lambda r: r.get("arm") == "A0")
    if len(wam_rows) != 40 or len(source_rows) != 40 or len(r4_rows) != 40:
        raise ValueError("expected WAM unseen A1=40, source A0=40 and R4 A0=40")
    wam = [audit_row(row, folder) for row, folder in wam_rows]
    source = [audit_row(row, folder) for row, folder in source_rows]
    r4 = []
    for row, folder in r4_rows:
        rec = audit_row(row, folder)
        rec["final_phase"] = json.loads((Path(folder) / "consumer.json").read_text())[-1].get("phase")
        rec["ever_close"] = any(x.get("phase") == "close" for x in json.loads((Path(folder) / "consumer.json").read_text()))
        r4.append(rec)
    source_keys = {(row["sequence"], row["object"]) for row, _ in source_rows}
    wam_keys = {(row["sequence"], row["object"]) for row, _ in wam_rows}
    shared_keys = source_keys & wam_keys
    source_by_key = {(row["sequence"], row["object"]): row for row, _ in source_rows}
    wam_by_key = {(row["sequence"], row["object"]): row for row, _ in wam_rows}
    witness = sorted(
        [
            {"sequence": key[0], "object": key[1], "source_id": source_by_key[key]["id"], "wam_id": wam_by_key[key]["id"]}
            for key in shared_keys
            if source_by_key[key].get("final_success") and not wam_by_key[key].get("final_success")
        ],
        key=lambda x: (x["sequence"], x["object"]),
    )
    report = dict(
        schema="ref2dex.executor-first-anomaly-audit.v1",
        status="COMPLETED",
        protocol=dict(
            no_new_simulation=True, contact_force_threshold_n=CONTACT_FORCE_N,
            limit_violation_threshold_rad=VIOLATION_EPS,
            first_limit_is_first_saved_q_state_outside_native_bound=True,
            contact_relation_uses_measured_contact_or_validated_force_evidence=True,
            ik_claim="not_performed; a single failed numerical IK solve is not treated as an inaccessibility proof",
        ),
        chains=dict(
            wam_unseen40=dict(root=str(args.wam_run), arm="A1", subset="unseen", summary=summarize_limits(wam), records=wam),
            source40=dict(root=str(args.source_run), arm="A0", subset="interface_unseen", summary=summarize_limits(source), records=source),
            source_r4=dict(root=str(args.r4_run), arm="A0", subset="development_same_v11", summary=summarize_r4(r4), records=r4),
        ),
        panel_gate=dict(
            source40_panel_n=len(source_keys), wam_unseen40_panel_n=len(wam_keys),
            shared_sequence_object_n=len(shared_keys), same_initial_panel=len(shared_keys) == 40,
            source_success_and_wam_failure_witnesses=witness,
            decision="no_intersection_offline_witness_or_short_replay" if not shared_keys else "intersection_available_for_offline_witness_check",
        ),
        interpretation=dict(
            limit_label_is_not_root_cause=True,
            final_success_and_final_limit_can_coexist_in_other_protocols=True,
            execution_chains_not_pooled=True,
            category_definitions={
                "contact_before_first_limit": "contact evidence precedes the first saved out-of-bound q state",
                "contact_only_at_or_after_first_limit": "no contact evidence precedes the first limit; later evidence exists",
                "no_contact_evidence_in_saved_trace": "no measured/force evidence in saved trace",
                "reached_stage_before_first_limit": "an achieved approach/contact/lift/hold/release stage tick precedes first limit",
            },
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["summary"] for k, v in report["chains"].items()}, indent=2))


if __name__ == "__main__":
    main()
