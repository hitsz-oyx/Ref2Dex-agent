"""Build approximate H-matched seven-candidate panels from frozen rollouts.

Panel composition uses only query-time H/state distance and future tau
geometry.  Frozen U32 labels are computed after composition and are never used
to choose candidates.  This is an exploratory bridge dataset; its H match is a
tolerance screen, not the exact fork identity contract of the old panel.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_evaluator.data import interaction_future, object_effect, sha
from consequence_evaluator.old_utility import teacher
from audit_history_candidate_bank import validate


SCHEMA = "ref2dex.history-candidate-panel.v1"
HORIZON = 24
QUERY_TICKS = (16, 24, 32)
H_RMS = 0.03
HAND_RMS_M = 0.003
Q_RMS = 0.03
PANELS_PER_TICK = {"train": 12, "val": 8, "test": 8}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def current_tau(pose, hand):
    current_inverse = np.linalg.inv(pose[0])
    return (np.einsum("ij,tkj->tki", current_inverse[:3, :3], hand[1:])
            + current_inverse[:3, 3])


def panel_labels(data, tick):
    pose = data["object_pose"]
    pair = data["pair"]
    _, utility = teacher(pose[tick, :, 2, 3], pair[tick],
                         pose[tick + 1:tick + 33, :, 2, 3].T,
                         pair[tick + 1:tick + 33].T,
                         data["reference_object_pose"][0, :, 2, 3])
    return np.asarray(utility, dtype=np.float32)


def farthest_candidates(anchor, candidates, tau):
    """Select six candidates by tau diversity, without using labels."""
    candidates = list(map(int, candidates))
    chosen = []
    while candidates and len(chosen) < 6:
        if not chosen:
            score = {env: float(np.sqrt(np.mean(np.square(tau[env] - tau[anchor]))))
                     for env in candidates}
        else:
            score = {env: min(float(np.sqrt(np.mean(np.square(tau[env] - tau[other]))))
                               for other in chosen)
                     for env in candidates}
        best = max(candidates, key=lambda env: (score[env], -env))
        chosen.append(best)
        candidates.remove(best)
    return chosen


def build_split(name, manifest, data, panel_start):
    rows = []
    panel_meta = []
    next_panel = panel_start
    episode_count = data["actor_observation"].shape[1]
    for tick in QUERY_TICKS:
        h = data["actor_observation"][tick]
        hand = data["hand_keypoints"][tick]
        q = data["dof_position"][tick]
        h_distance = np.sqrt(np.mean(np.square(h[:, None] - h[None]), axis=2))
        hand_distance = np.sqrt(np.mean(np.square(hand[:, None] - hand[None]), axis=(2, 3)))
        q_distance = np.sqrt(np.mean(np.square(q[:, None] - q[None]), axis=2))
        neighbor = (h_distance <= H_RMS) & (hand_distance <= HAND_RMS_M) & (q_distance <= Q_RMS)
        np.fill_diagonal(neighbor, False)
        tau = np.stack([
            current_tau(data["object_pose"][tick:tick + HORIZON + 1, env],
                        data["hand_keypoints"][tick:tick + HORIZON + 1, env])
            for env in range(episode_count)])
        labels = panel_labels(data, tick)
        anchors = sorted((env for env in range(episode_count) if neighbor[env].sum() >= 6),
                         key=lambda env: (-int(neighbor[env].sum()), env))
        selected_anchors = anchors[:PANELS_PER_TICK[name]]
        for anchor in selected_anchors:
            candidates = farthest_candidates(anchor, np.flatnonzero(neighbor[anchor]), tau)
            if len(candidates) != 6:
                continue
            panel_envs = [anchor] + candidates
            pose_window = data["object_pose"][tick:tick + HORIZON + 1]
            for candidate, env in enumerate(panel_envs):
                pose = pose_window[:, env]
                hand_window = data["hand_keypoints"][tick:tick + HORIZON + 1, env]
                effect = object_effect(pose)[:, :3].reshape(HORIZON, 12)
                future_relative = interaction_future(pose, hand_window).reshape(HORIZON, 33)
                inverse = np.linalg.inv(pose[0])
                hand_history = data["hand_keypoints"][tick - 3:tick + 1, env]
                object_history = data["object_pose"][tick - 3:tick + 1, env]
                pw_hand_history = (np.einsum("ij,tkj->tki", inverse[:3, :3], hand_history)
                                   + inverse[:3, 3])
                pw_object_history = np.einsum("ij,tjk->tik", inverse, object_history)
                rows.append(dict(
                    history=data["actor_observation"][tick, env],
                    future=np.concatenate((effect, future_relative), axis=-1),
                    effect_gt=effect,
                    pw_hand_future=tau[env],
                    pw_hand_history=pw_hand_history,
                    pw_object_history=pw_object_history,
                    label=labels[env], tick=tick, source_env=env,
                    episode=f"{manifest['seed']}:{env}", split=name,
                    split_group=f"{name}:{manifest['seed']}", panel=next_panel,
                    candidate=candidate,
                    h_to_anchor=float(h_distance[anchor, env]),
                    hand_to_anchor_mm=float(hand_distance[anchor, env] * 1000.),
                    q_to_anchor=float(q_distance[anchor, env]),
                ))
            panel_meta.append(dict(panel=next_panel, split=name, tick=tick,
                                   anchor_env=int(anchor), candidates=panel_envs,
                                   label_range=float(np.ptp(labels[panel_envs])),
                                   max_h_rms=float(np.max(h_distance[anchor, panel_envs])),
                                   max_hand_rms_mm=float(np.max(hand_distance[anchor, panel_envs]) * 1000.),
                                   max_q_rms=float(np.max(q_distance[anchor, panel_envs]))))
            next_panel += 1
    return rows, panel_meta, next_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_parent = (ROOT / "outputs/consequence-evaluator").resolve()
    if output.exists() or output.parent != expected_parent:
        raise ValueError("fresh task-owned panel output required")
    paths = {name: path.resolve() for name, path in (
        ("train", args.train), ("val", args.val), ("test", args.test))}
    loaded = {name: validate(path) for name, path in paths.items()}
    actors = {value[1].get("actor_sha256") for value in loaded.values()}
    if len(actors) != 1 or None in actors:
        raise ValueError("all panel splits must use one frozen actor")
    rows = []
    panel_meta = []
    panel = 0
    for name in ("train", "val", "test"):
        new_rows, meta, panel = build_split(name, loaded[name][1], loaded[name][2], panel)
        rows.extend(new_rows); panel_meta.extend(meta)
    if not panel_meta:
        raise ValueError("no complete seven-candidate panels")
    keys = rows[0].keys()
    arrays = {key: np.stack([row[key] for row in rows]) for key in keys}
    panel_ids = np.unique(arrays["panel"])
    panel_rows = np.asarray([
        np.flatnonzero(arrays["panel"] == panel_id) for panel_id in panel_ids])
    if panel_rows.shape[1] != 7 or not np.all(arrays["candidate"][panel_rows] == np.arange(7)):
        raise ValueError("candidate panels are not complete 0..6 rows")
    for panel_row in panel_rows:
        if len(np.unique(arrays["split"][panel_row])) != 1:
            raise ValueError("panel crosses episode split")
    hashes = {str((path / name).resolve()): sha(path / name)
              for path in paths.values() for name in ("manifest.json", "trajectory.npz")}
    for path in (Path(__file__).resolve(), TASK / "src/consequence_evaluator/data.py",
                 TASK / "src/consequence_evaluator/old_utility.py"):
        hashes[str(path.resolve())] = sha(path)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETED",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": hashes, "panel_count": int(len(panel_rows)),
        "rows": int(len(rows)), "panels_by_split": {
            name: int(np.sum(arrays["split"][panel_rows[:, 0]] == name))
            for name in ("train", "val", "test")},
        "panel_contract": {
            "h_rms_max": H_RMS, "hand_rms_max_m": HAND_RMS_M, "q_rms_max": Q_RMS,
            "candidate_selection": "farthest-point tau diversity within query-time H/state neighbors; labels not used",
        },
        "query_ticks": list(QUERY_TICKS), "panel_meta": panel_meta,
        "actor_sha256": next(iter(actors)),
    }
    manifest["windows_sha256"] = None
    output.mkdir(parents=True)
    np.savez_compressed(output / "windows.npz", **arrays)
    manifest["windows_sha256"] = sha(output / "windows.npz")
    write(output / "manifest.json", manifest)
    print(json.dumps({"panels": manifest["panel_count"],
                      "panels_by_split": manifest["panels_by_split"],
                      "rows": manifest["rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
