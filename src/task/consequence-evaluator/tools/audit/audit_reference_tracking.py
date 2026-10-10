"""Audit frozen reference-tracking behavior without model or simulator execution."""
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
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracker-role", default="tracker")
    parser.add_argument("--control-role", default="nominal")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    output = args.output.resolve()
    output.relative_to(root / "outputs/consequence-evaluator")
    if output.exists():
        raise ValueError("fresh task-owned audit output required")
    train = json.loads((args.training / "manifest.json").read_text())
    evaluation = json.loads((args.evaluation / "manifest.json").read_text())
    result = json.loads((args.evaluation / "result.json").read_text())
    if train["status"] != "COMPLETED" or evaluation["status"] != "COMPLETED":
        raise ValueError("completed frozen training and evaluation required")
    reference_path = str(args.reference.resolve())
    checkpoint = args.training / "final.pt"
    if (train["input_sha256"][reference_path] != sha(args.reference)
            or evaluation["input_sha256"][reference_path] != sha(args.reference)
            or evaluation["input_sha256"][str(checkpoint.resolve())] != sha(checkpoint)):
        raise ValueError("reference/checkpoint provenance mismatch")
    with args.reference.open("rb") as stream:
        reference = pickle.load(stream)
    with np.load(args.evaluation / "trajectory.npz", allow_pickle=False) as stream:
        data = {key: stream[key] for key in stream.files}
    n = evaluation["num_envs"]
    if data["action"].shape != (542, n, 18) or data["dof_position"].shape != (543, n, 18):
        raise ValueError("complete reference evaluation trajectory required")
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError("nonfinite evaluation field")
    exact_initial = {}
    for key in ("object_pose", "hand_keypoints", "dof_position", "dof_velocity"):
        if key not in reference:
            continue
        expected = np.broadcast_to(reference[key][0, 0], data[key][0].shape)
        exact_initial[key] = bool(np.array_equal(data[key][0], expected))
    if not all(exact_initial.values()):
        raise ValueError("initial reference/live state alignment failed")
    controller = evaluation["native_controller"]
    scale = np.asarray(controller["scale"], np.float32)
    offset = np.asarray(controller["offset"], np.float32)
    action = data["action"].copy()
    action[..., 6:] = (1 + action[..., 6:]) / 2
    target = offset + scale * action
    target[..., :6] += data["dof_position"][:-1, :, :6]
    for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
                                  (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
        target[..., distal] = target[..., parent] * ratio
    target_error = float(np.abs(target - data["pd_targets"]).max())
    if target_error > 1e-6:
        raise ValueError("native PD target/action reconstruction mismatch")
    rows = result["outcomes"]
    teacher = [row for row in rows if row["role"] == "teacher"]
    tracker = [row for row in rows if row["role"] == args.tracker_role]
    nominal = [row for row in rows if row["role"] == args.control_role]
    teacher_usable = sum(row["maximum_held_frames"] >= 45 for row in teacher) >= len(teacher) / 2
    near_teacher = [row for row in tracker if row["maximum_held_frames"] >= .9 * result["reference_held"]
                    and row["intermediate_loss_events"] == 0]
    clipping_rate = sum(row["clipping_count"] for row in tracker) / (542 * len(tracker))
    median_advantage = float(np.median([row["maximum_held_frames"] for row in tracker])
                             - np.median([row["maximum_held_frames"] for row in nominal]))
    passed = (teacher_usable and len(near_teacher) >= len(tracker) / 2
              and median_advantage >= 45 and clipping_rate < .01)
    status = "UNCLEAR" if not teacher_usable else ("PROMISING" if passed else "UNPROMISING")
    poses = data["object_pose"]
    height = poses[:, :, 2, 3] - poses[0, :, 2, 3]
    supported = data["table_footprint"] & (np.abs(data["support_gap"]) <= .02)
    held = (data["surface_gap"] <= .01) & ~supported & (height >= .03)
    held[0] = False
    roles = np.asarray([row["role"] for row in rows])
    terminal_held = {role: int(held[-1, roles == role].sum()) for role in sorted(set(roles))}
    audit = dict(schema="ref2dex.reference-tracker-audit.v1", status=status,
                 screen="predeclared near-teacher native holding screen", passed=passed,
                 teacher_usable=teacher_usable, tracker_role=args.tracker_role, control_role=args.control_role,
                 qualifying_tracker_count=len(near_teacher),
                 median_tracker_advantage_frames=median_advantage, tracker_clipping_rate=clipping_rate,
                 exact_initial=exact_initial, maximum_pd_reconstruction_error=target_error,
                 terminal_held=terminal_held, summary=result["summary"],
                 interpretation="Transient holding benefit can coexist with failure of the near-teacher screen. No tau-only completion or method refutation.",
                 loss_metric_caveat="Intermediate loss counts exclude supported states; zero counts do not imply no slip/drop back to table.",
                 input_sha256={str(path.resolve()): sha(path) for path in (
                     args.reference, args.training / "manifest.json", checkpoint,
                     args.evaluation / "manifest.json", args.evaluation / "result.json",
                     args.evaluation / "trajectory.npz", Path(__file__))})
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time_s = np.arange(543)  # The native control frequency is recorded separately; show ticks.
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    source = reference["object_pose"][:, 0, 2, 3]
    axes[0].plot(time_s, source - source[0], "k--", label="Fixed reference")
    for role, color in zip(sorted(set(roles)), ("C0", "C1", "C2", "C3")):
        group = roles == role
        values = height[:, group]
        axes[0].plot(time_s, np.median(values, axis=1), color=color, label=role)
        low, high = np.quantile(values, [.25, .75], axis=1)
        axes[0].fill_between(time_s, low, high, color=color, alpha=.12)
        axes[1].plot(time_s, held[:, group].mean(axis=1), color=color, label=role)
    axes[0].set_ylabel("Object lift (m)")
    axes[0].legend(ncol=4, fontsize=9)
    axes[1].set_ylabel("Held fraction of role rows")
    axes[1].set_xlabel("Native control tick")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Single-motion oracle reference Probe: descriptive rows, not independent samples")
    fig.tight_layout()
    fig.savefig(output / "behavior.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: audit[key] for key in ("status", "passed", "exact_initial",
                      "maximum_pd_reconstruction_error", "terminal_held", "summary")}, indent=2))


if __name__ == "__main__":
    main()
