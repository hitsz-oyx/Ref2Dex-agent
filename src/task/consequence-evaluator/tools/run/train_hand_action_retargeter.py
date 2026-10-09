"""Fit the ref7_2 full-action hand retargeter on episode-split rollouts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.hand_action_retargeter import (
    ACTION_DIM, HORIZON, SCHEMA, HandActionRetargeter, Standardizer,
    hand_object_context, trajectory_input)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_windows(root, stride):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    trajectory_path = root / "trajectory.npz"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "COMPLETED" or manifest.get("mode") != "retarget"
            or manifest.get("structured_residual_schema") != "ref2dex.structured-residual.v1"):
        raise ValueError("completed structured retarget rollout required: " + str(root))
    with np.load(trajectory_path, allow_pickle=False) as data:
        required = {"hand_keypoints", "object_pose", "dof_position", "dof_velocity", "action", "done", "length",
                    "structured_residual", "actor_action", "structured_mode", "structured_phase"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("retarget rollout missing fields %s" % sorted(missing))
        hand = np.asarray(data["hand_keypoints"], dtype="float32")
        object_pose = np.asarray(data["object_pose"], dtype="float32")
        q = np.asarray(data["dof_position"], dtype="float32")
        dq = np.asarray(data["dof_velocity"], dtype="float32")
        action = np.asarray(data["action"], dtype="float32")
        done = np.asarray(data["done"], dtype=bool)
        lengths = np.asarray(data["length"], dtype=np.int64)
        residual = np.asarray(data["structured_residual"], dtype="float32")
        mode = np.asarray(data["structured_mode"], dtype=np.int8)
        phase = np.asarray(data["structured_phase"], dtype=np.int8)
    if (hand.ndim != 4 or hand.shape[2:] != (11, 3)
            or object_pose.shape != (len(hand), q.shape[1], 4, 4) or q.shape[2:] != (18,)
            or dq.shape != q.shape or action.shape != (len(q) - 1, q.shape[1], ACTION_DIM)
            or done.shape != action.shape[:2] or residual.shape != action.shape
            or mode.shape != action.shape[:2] or phase.shape != (len(action),)
            or len(lengths) != q.shape[1]):
        raise ValueError("retarget rollout tensor shape mismatch")
    if not all(np.isfinite(value).all() for value in (hand, q, dq, action, residual)):
        raise ValueError("nonfinite retarget rollout tensor")
    if np.abs(action).max() > 1.00001 or np.abs(residual).max() > .12001:
        raise ValueError("retarget rollout action/residual bound mismatch")
    if np.any(done[:-1] & (np.arange(len(done) - 1)[:, None] < lengths[None] - 1)):
        raise ValueError("early termination inside a retained episode")

    hands = []
    states = []
    contexts = []
    targets = []
    ticks = []
    episodes = []
    phases = []
    # The action at t is paired with q/dq and hand at t, then hand t+1..t+24.
    for env, length in enumerate(lengths.tolist()):
        if length < HORIZON:
            continue
        for tick in range(0, int(length) - HORIZON + 1, int(stride)):
            hands.append(trajectory_input(hand[tick, env], hand[tick + 1:tick + 1 + HORIZON, env]))
            states.append(np.concatenate((q[tick, env], dq[tick, env])))
            previous = action[tick - 1, env] if tick else np.zeros(ACTION_DIM, dtype="float32")
            contexts.append(hand_object_context(hand[tick, env], object_pose[tick, env], previous))
            targets.append(action[tick:tick + HORIZON, env])
            ticks.append(tick)
            episodes.append("%s:%d" % (manifest["seed"], env))
            phases.append(int(phase[tick]))
    if not hands:
        raise ValueError("no complete 24-step windows in " + str(root))
    arrays = dict(hand=np.asarray(hands, dtype="float32"), state=np.asarray(states, dtype="float32"),
                  context=np.asarray(contexts, dtype="float32"),
                  action=np.asarray(targets, dtype="float32"), tick=np.asarray(ticks, dtype=np.int32),
                  episode=np.asarray(episodes), phase=np.asarray(phases, dtype=np.int8))
    return arrays, manifest, {str(manifest_path): sha(manifest_path), str(trajectory_path): sha(trajectory_path)}


def concat_split(paths, stride):
    chunks = []
    manifests = []
    hashes = {}
    for path in paths:
        chunk, manifest, frozen = load_windows(path, stride)
        chunks.append(chunk); manifests.append(manifest); hashes.update(frozen)
    result = {key: np.concatenate([part[key] for part in chunks], axis=0)
              for key in chunks[0]}
    return result, manifests, hashes


def load_teacher_windows(path, stride):
    """Load one preserved full teacher launch as a motion-rich anchor set."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        packet = pickle.load(stream)
    if (packet.get("engineering_only") is not True or packet.get("role_names", [None])[0] != "reactive_teacher"
            or packet.get("seed") is None or packet["hand_keypoints"].shape != (543, 4, 11, 3)
            or packet["object_pose"].shape != (543, 4, 4, 4)
            or packet["dof_position"].shape != (543, 4, 18)
            or packet["dof_velocity"].shape != (543, 4, 18)
            or packet["actions"].shape != (542, 4, ACTION_DIM)
            or packet["done"][:-1].any()):
        raise ValueError("complete teacher anchor packet required: " + str(path))
    hand = np.asarray(packet["hand_keypoints"][:, 0], dtype="float32")
    object_pose = np.asarray(packet["object_pose"][:, 0], dtype="float32")
    q = np.asarray(packet["dof_position"][:, 0], dtype="float32")
    dq = np.asarray(packet["dof_velocity"][:, 0], dtype="float32")
    action = np.asarray(packet["actions"][:, 0], dtype="float32")
    if not all(np.isfinite(value).all() for value in (hand, q, dq, action)):
        raise ValueError("nonfinite teacher anchor tensor")
    hands = []; states = []; contexts = []; targets = []; ticks = []; episodes = []; phases = []
    for tick in range(0, len(action) - HORIZON + 1, int(stride)):
        hands.append(trajectory_input(hand[tick], hand[tick + 1:tick + 1 + HORIZON]))
        states.append(np.concatenate((q[tick], dq[tick])))
        previous = action[tick - 1] if tick else np.zeros(ACTION_DIM, dtype="float32")
        contexts.append(hand_object_context(hand[tick], object_pose[tick], previous))
        targets.append(action[tick:tick + HORIZON])
        # The same packet filename (``act.pkl``) is used by several retained
        # teacher launches.  Split identity must include the launch directory;
        # otherwise train/val/test anchor packets look like one leaked episode.
        episode_id = "teacher:%s:0" % path.parent.name
        ticks.append(tick); episodes.append(episode_id)
        phases.append(0 if tick < 120 else (1 if tick < 240 else 2))
    data = dict(hand=np.asarray(hands, dtype="float32"), state=np.asarray(states, dtype="float32"),
                context=np.asarray(contexts, dtype="float32"),
                action=np.asarray(targets, dtype="float32"), tick=np.asarray(ticks, dtype=np.int32),
                episode=np.asarray(episodes), phase=np.asarray(phases, dtype=np.int8))
    manifest = dict(run_id="teacher-anchor:" + path.parent.name, actor_sha256=None,
                    seed=int(packet["seed"]), source=str(path), anchor=True,
                    role_outcome=packet["role_outcomes"]["reactive_teacher"])
    return data, manifest, {str(path): sha(path)}


def append_data(base, extra):
    return {key: np.concatenate((base[key], extra[key]), axis=0) for key in base}


def episode_disjoint(splits):
    seen = {}
    for name, data in splits.items():
        for episode in np.unique(data["episode"]):
            previous = seen.get(str(episode))
            if previous is not None and previous != name:
                raise ValueError("episode appears in more than one split: %s" % episode)
            seen[str(episode)] = name


def metric(prediction, target, data):
    error = prediction - target
    result = dict(
        l1=float(np.mean(np.abs(error))),
        wrist_translation_mae=float(np.mean(np.abs(error[..., :3]))),
        wrist_rotation_mae=float(np.mean(np.abs(error[..., 3:6]))),
        finger_mae=float(np.mean(np.abs(error[..., 6:]))),
        horizon0_l1=float(np.mean(np.abs(error[:, 0]))),
        horizon23_l1=float(np.mean(np.abs(error[:, -1]))),
    )
    for name, mask in (("contact_onset", (data["tick"] >= 96) & (data["tick"] < 192)),
                       ("hold_stage", data["tick"] >= 240)):
        if mask.any():
            result[name + "_finger_mae"] = float(np.mean(np.abs(error[mask, ..., 6:])))
            result[name + "_windows"] = int(mask.sum())
        else:
            result[name + "_finger_mae"] = None
            result[name + "_windows"] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-train", type=Path, nargs="*", default=[])
    parser.add_argument("--teacher-val", type=Path, nargs="*", default=[])
    parser.add_argument("--teacher-test", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=293)
    args = parser.parse_args()
    if not 1 <= args.steps <= 4000 or not 30 <= args.seconds <= 1200 or not 1 <= args.stride <= 24:
        parser.error("bounded fit budget required")
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned fit output required")
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied: " + occupied)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical index 0.
    device = torch.device("cuda:0")

    splits = {}
    manifests = {}
    frozen = {str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
              str((TASK / "src/consequence_evaluator/hand_action_retargeter.py").resolve()):
                  sha(TASK / "src/consequence_evaluator/hand_action_retargeter.py")}
    for name in ("train", "val", "test"):
        value, meta, hashes = concat_split(getattr(args, name), args.stride)
        teacher_paths = getattr(args, "teacher_" + name)
        for teacher_path in teacher_paths:
            anchor, anchor_meta, anchor_hashes = load_teacher_windows(teacher_path, args.stride)
            value = append_data(value, anchor)
            meta.append(anchor_meta); hashes.update(anchor_hashes)
        splits[name] = value; manifests[name] = meta; frozen.update(hashes)
    episode_disjoint(splits)
    actor_hashes = {m["actor_sha256"] for group in manifests.values() for m in group
                    if m.get("actor_sha256") is not None}
    if len(actor_hashes) != 1:
        raise ValueError("all splits must use one frozen actor")

    train = splits["train"]
    stats = {"hand": Standardizer.fit(train["hand"]),
             "state": Standardizer.fit(train["state"]),
             "action": Standardizer.fit(train["action"])}
    tensors = {}
    for name, data in splits.items():
        tensors[name] = dict(
            hand=torch.as_tensor(stats["hand"].encode(data["hand"]), device=device),
            state=torch.as_tensor(stats["state"].encode(data["state"]), device=device),
            action=torch.as_tensor(stats["action"].encode(data["action"]), device=device))
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status="RUNNING", run_id=output.name,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    task="consequence-evaluator", route="ref7_2", gpu=args.gpu, seed=args.seed,
                    steps_cap=args.steps, seconds_cap=args.seconds, batch=args.batch, stride=args.stride,
                    splits={name: dict(rollouts=[m["run_id"] for m in manifests[name]],
                                      episodes=int(len(np.unique(splits[name]["episode"]))),
                                      windows=int(len(splits[name]["hand"])))
                            for name in splits},
                    teacher_anchor_sources={name: [m["source"] for m in manifests[name] if m.get("anchor")]
                                            for name in splits},
                    input_sha256=frozen, input_contract="future hand displacement t+1:t+24 plus q_t,dq_t",
                    target_contract="captured full native action A_t:t+24, no actor action at inference",
                    normalization="train-only per horizon x action coordinate; hand/state likewise",
                    action_groups={"wrist_translation":[0,1,2], "wrist_rotation":[3,4,5],
                                   "finger":list(range(6,18))})
    write(output / "manifest.json", manifest)

    model = HandActionRetargeter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best = float("inf")
    best_step = None
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        elapsed = time.monotonic() - started
        if elapsed > args.seconds:
            break
        model.train()
        index = torch.as_tensor(rng.integers(len(train["hand"]), size=args.batch), device=device)
        prediction = model(tensors["train"]["hand"][index], tensors["train"]["state"][index])
        loss = (prediction - tensors["train"]["action"][index]).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite retargeter loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_prediction = model(tensors["val"]["hand"], tensors["val"]["state"])
                val_loss = float((val_prediction - tensors["val"]["action"]).abs().mean())
            row = dict(step=step, train_l1=float(loss), val_l1=val_loss, elapsed_s=elapsed,
                       eta_s=max(0., elapsed / step * (args.steps - step)))
            print(json.dumps(row), flush=True); history.append(row)
            if val_loss < best:
                best = val_loss; best_step = step
                torch.save(dict(schema=SCHEMA, width=model.width, state_dict=model.state_dict(),
                                statistics={k: v.as_dict() for k, v in stats.items()},
                                best_step=step, best_val_l1=best, manifest=manifest), output / "best.pt")
    if best_step is None:
        raise RuntimeError("fit ended before a validation checkpoint")
    payload = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.eval()
    metrics = {}
    with torch.no_grad():
        for name, data in splits.items():
            prediction = stats["action"].decode(model(tensors[name]["hand"], tensors[name]["state"]).cpu().numpy())
            metrics[name] = metric(prediction, data["action"], data)
            if name == "test":
                metrics[name]["zero_hand_l1"] = float(np.mean(np.abs(
                    stats["action"].decode(model(
                        torch.zeros_like(tensors[name]["hand"]), tensors[name]["state"]).cpu().numpy())
                    - data["action"])))
    result = dict(status="COMPLETED", best_step=best_step, best_val_l1=best,
                  elapsed_s=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  checkpoint_sha256=sha(output / "best.pt"), metrics=metrics, history=history,
                  claim="offline full-action fit only; GT execution upper bound pending")
    write(output / "result.json", result)
    manifest.update(status="COMPLETED", best_step=best_step, checkpoint_sha256=result["checkpoint_sha256"],
                    elapsed_s=result["elapsed_s"])
    write(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
