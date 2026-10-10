"""Distinguish recorded tracker drops from raw-motion and teacher-reference placing."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="tracker")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    output = args.output.resolve()
    output.relative_to(root / "outputs/consequence-evaluator")
    if output.exists():
        raise ValueError("fresh task-owned diagnostic output required")
    inputs = json.loads(args.inputs.read_text())
    manifest = json.loads((args.evaluation / "manifest.json").read_text())
    result = json.loads((args.evaluation / "result.json").read_text())
    for path in (args.inputs, args.reference):
        if manifest["input_sha256"][str(path.resolve())] != sha(path):
            raise ValueError("recorded evaluation input drift")
    motion = Path(inputs["motions"]) / "s3_airplane_lift/interaction_hand_inspire.pt"
    if sha(motion) != inputs["input_sha256"][str(motion)]:
        raise ValueError("raw motion does not match pinned input")
    # CPU is appropriate: deserialization and array/trajectory statistics only,
    # no neural-model inference, training, or simulator execution.
    import torch
    import yaml
    raw = torch.load(motion, map_location="cpu", weights_only=True).numpy()
    cfg = yaml.safe_load(Path(inputs["cfg_env"]).read_text())
    dt = cfg["env"]["controlFrequencyInv"] / 60.
    with args.reference.open("rb") as stream:
        source = pickle.load(stream)
    with np.load(args.evaluation / "trajectory.npz", allow_pickle=False) as stream:
        data = {key: stream[key] for key in stream.files}
    if raw.shape != (543, 598) or data["object_pose"].shape != (543, 64, 4, 4):
        raise ValueError("frozen full single-motion Probe required")
    z = data["object_pose"][:, :, 2, 3]
    support = data["table_footprint"] & (np.abs(data["support_gap"]) <= .02)
    held = (data["surface_gap"] <= .01) & ~support & (z - z[0] >= .03)
    held[0] = False
    raw_lift = raw[:, 200] - raw[0, 200]
    raw_contact = np.round(raw[:, 222:238]).astype(bool).any(-1)
    source_lift = source["object_pose"][:, 0, 2, 3] - source["object_pose"][0, 0, 2, 3]
    source_supported = source["table_footprint"][:, 0] & (np.abs(source["support_gap"][:, 0]) <= .02)
    source_held = (source["surface_gap"][:, 0] <= .01) & ~source_supported & (source_lift >= .03)
    rows = []
    for outcome in result["outcomes"]:
        if outcome["role"] != args.role:
            continue
        env = outcome["env"]
        if held[-1, env] or not held[:, env].any():
            continue
        last = int(np.flatnonzero(held[:, env])[-1])
        release = last + 1
        indices = np.arange(release, min(last + 12, 543))
        air = indices[(data["surface_gap"][indices, env] > .01) & ~support[indices, env]]
        acceleration = np.diff(data["object_velocity"][:, env, 2]) / dt
        gravity = [int(t) for t in air if t + 1 in air and abs(float(acceleration[t]) + 9.81) < .5]
        rows.append(dict(env=env, last_held_tick=last, separation_tick=release,
                         unsupported_separated_ticks=air.tolist(), gravity_like_intervals=gravity,
                         reference_held_at_separation=bool(source_held[release]),
                         reference_support_gap_m=float(source["support_gap"][release, 0]),
                         raw_motion_lift_m=float(raw_lift[release]),
                         raw_motion_contact=bool(raw_contact[release]),
                         tracker_lift_m=float(z[release, env] - z[0, env]),
                         tracker_support_gap_m=float(data["support_gap"][release, env]),
                         tracker_hand_object_gap_m=float(data["surface_gap"][release, env]),
                         tracker_vertical_velocity_mps=float(data["object_velocity"][release, env, 2]),
                         tracker_force_pair=bool(data["pair"][release, env])))
    if any(not row["reference_held_at_separation"] for row in rows):
        raise ValueError("the teacher-reference holding interpretation changed")
    diagnostic = dict(schema="ref2dex.reference-tracker-drop-diagnostic.v1",
                      control_dt_s=dt, role=args.role,
                      episodes=sum(row["role"] == args.role for row in result["outcomes"]),
                      terminal_held_count=sum(bool(held[-1, row["env"]]) for row in result["outcomes"] if row["role"] == args.role),
                      raw_motion=dict(initial_z_m=float(raw[0, 200]), final_z_m=float(raw[-1, 200]),
                                      final_lift_m=float(raw_lift[-1]), peak_lift_m=float(raw_lift.max()),
                                      final_contact=bool(raw_contact[-1]),
                                      last_contact_tick=int(np.flatnonzero(raw_contact)[-1])),
                      teacher_reference=dict(final_lift_m=float(source_lift[-1]),
                                             final_support_gap_m=float(source["support_gap"][-1, 0]),
                                             final_held=bool(source_held[-1])),
                      unsupported_separation_count=sum(bool(row["unsupported_separated_ticks"]) for row in rows),
                      gravity_like_drop_count=sum(bool(row["gravity_like_intervals"]) for row in rows),
                      reference_held_at_all_separations=True, tracker_rows=rows,
                      interpretation="Raw motion includes placing/releasing; this measured teacher reference stays airborne. Remaining lost terminal holds are diagnosed individually; unsupported detached gravity-like descent is not controlled placement.",
                      limitation="Rows without terminal loss, or without any held interval, are excluded from separation details. This diagnoses recorded behavior, not its exact policy/contact failure cause or original-task success.",
                      input_sha256={str(path.resolve()): sha(path) for path in (
                          args.inputs, args.reference, motion, Path(inputs["cfg_env"]),
                          args.evaluation / "trajectory.npz", args.evaluation / "result.json", Path(__file__))})
    output.mkdir(parents=True)
    (output / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ticks = np.arange(543)
    env = rows[0]["env"] if rows else next(row["env"] for row in result["outcomes"] if row["role"] == args.role)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(ticks, raw_lift, label="Raw motion")
    axes[0].plot(ticks, source_lift, label="Teacher reference")
    axes[0].plot(ticks, z[:, env] - z[0, env], label="{} env{}".format(args.role, env))
    axes[0].set_ylabel("Lift relative to reset (m)"); axes[0].legend(ncol=3)
    axes[1].plot(ticks, data["surface_gap"][:, env], label="Hand-object gap")
    axes[1].plot(ticks, data["support_gap"][:, env], label="Object-table gap")
    axes[1].axhline(.01, linestyle="--", color="gray", linewidth=1)
    axes[1].set_ylabel("Gap (m)"); axes[1].legend(ncol=2)
    axes[2].plot(ticks, source["object_velocity"][:, 0, 2], label="Reference vertical velocity")
    axes[2].plot(ticks, data["object_velocity"][:, env, 2], label="Tracker vertical velocity")
    axes[2].set_ylabel("Vertical velocity (m/s)"); axes[2].set_xlabel("Native control tick")
    axes[2].legend(ncol=2)
    for axis in axes:
        if rows:
            axis.axvline(rows[0]["separation_tick"], linestyle=":", color="red", linewidth=1)
        axis.grid(alpha=.2)
    fig.suptitle("Raw task placing vs teacher-reference holding and tracker separation")
    fig.tight_layout(); fig.savefig(output / "reference-vs-drop.png", dpi=150); plt.close(fig)
    print(json.dumps({key: diagnostic[key] for key in ("raw_motion", "teacher_reference",
                      "unsupported_separation_count", "gravity_like_drop_count")}, indent=2))


if __name__ == "__main__":
    main()
