"""Fit the contact-context full-action retargeter from ref7_2 rollouts."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.hand_action_retargeter import (
    ACTION_DIM, CONTEXT_DIM, CONTEXT_SCHEMA, HORIZON, ContextHandActionRetargeter,
    Standardizer)
from train_hand_action_retargeter import (
    concat_split, episode_disjoint, load_teacher_windows, metric, sha, write)


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
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=406)
    args = parser.parse_args()
    if not 1 <= args.steps <= 4000 or not 30 <= args.seconds <= 1200 or not 1 <= args.stride <= 24:
        parser.error("bounded contextual fit budget required")
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned contextual fit output required")
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied: " + occupied)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    splits = {}
    manifests = {}
    frozen = {
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str((TASK / "src/consequence_evaluator/hand_action_retargeter.py").resolve()):
            sha(TASK / "src/consequence_evaluator/hand_action_retargeter.py"),
        str((TASK / "tools/run/train_hand_action_retargeter.py").resolve()):
            sha(TASK / "tools/run/train_hand_action_retargeter.py"),
    }
    for name in ("train", "val", "test"):
        value, meta, hashes = concat_split(getattr(args, name), args.stride)
        for teacher_path in getattr(args, "teacher_" + name):
            anchor, anchor_meta, anchor_hashes = load_teacher_windows(teacher_path, args.stride)
            value = {key: np.concatenate((value[key], anchor[key]), axis=0) for key in value}
            meta.append(anchor_meta)
            hashes.update(anchor_hashes)
        splits[name] = value
        manifests[name] = meta
        frozen.update(hashes)
    episode_disjoint(splits)
    actor_hashes = {m["actor_sha256"] for group in manifests.values() for m in group
                    if m.get("actor_sha256") is not None}
    if len(actor_hashes) != 1:
        raise ValueError("all structured splits must use one frozen actor")
    train = splits["train"]
    stats = {"hand": Standardizer.fit(train["hand"]),
             "state": Standardizer.fit(train["state"]),
             "context": Standardizer.fit(train["context"]),
             "action": Standardizer.fit(train["action"])}
    tensors = {}
    for name, data in splits.items():
        tensors[name] = {
            key: torch.as_tensor(stats[key].encode(data[key]), device=device)
            for key in ("hand", "state", "context", "action")
        }
    output.mkdir(parents=True)
    manifest = dict(
        schema=CONTEXT_SCHEMA, status="RUNNING", run_id=output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        task="consequence-evaluator", route="ref7_2-contextual", gpu=args.gpu, seed=args.seed,
        steps_cap=args.steps, seconds_cap=args.seconds, batch=args.batch, stride=args.stride,
        splits={name: dict(rollouts=[m["run_id"] for m in manifests[name]],
                           episodes=int(len(np.unique(splits[name]["episode"]))),
                           windows=int(len(splits[name]["hand"]))) for name in splits},
        teacher_anchor_sources={name: [m["source"] for m in manifests[name] if m.get("anchor")]
                                for name in splits},
        input_sha256=frozen,
        input_contract="future hand displacement t+1:t+24 + q_t,dq_t + current hand-object geometry + previous native command",
        target_contract="captured full native action A_t:t+24, no actor action at inference",
        context_contract="query-time hand points in object frame plus A_{t-1}; no future object/contact label",
        normalization="train-only per horizon x action coordinate; hand/state/context likewise",
        action_groups={"wrist_translation": [0, 1, 2], "wrist_rotation": [3, 4, 5],
                       "finger": list(range(6, 18))}, context_dim=CONTEXT_DIM,
    )
    write(output / "manifest.json", manifest)

    model = ContextHandActionRetargeter().to(device)
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
        prediction = model(tensors["train"]["hand"][index], tensors["train"]["state"][index],
                           tensors["train"]["context"][index])
        loss = (prediction - tensors["train"]["action"][index]).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite contextual retargeter loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_prediction = model(tensors["val"]["hand"], tensors["val"]["state"],
                                       tensors["val"]["context"])
                val_loss = float((val_prediction - tensors["val"]["action"]).abs().mean())
            row = dict(step=step, train_l1=float(loss), val_l1=val_loss, elapsed_s=elapsed,
                       eta_s=max(0., elapsed / step * (args.steps - step)))
            print(json.dumps(row), flush=True)
            history.append(row)
            if val_loss < best:
                best = val_loss
                best_step = step
                torch.save(dict(schema=CONTEXT_SCHEMA, width=model.width, state_dict=model.state_dict(),
                                statistics={key: value.as_dict() for key, value in stats.items()},
                                best_step=step, best_val_l1=best, manifest=manifest), output / "best.pt")
    if best_step is None:
        raise RuntimeError("contextual fit ended before a validation checkpoint")
    payload = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metrics = {}
    with torch.no_grad():
        for name, data in splits.items():
            prediction = stats["action"].decode(model(
                tensors[name]["hand"], tensors[name]["state"], tensors[name]["context"]).cpu().numpy())
            metrics[name] = metric(prediction, data["action"], data)
            if name == "test":
                zero_hand = torch.zeros_like(tensors[name]["hand"])
                zero_pred = stats["action"].decode(model(
                    zero_hand, tensors[name]["state"], tensors[name]["context"]).cpu().numpy())
                metrics[name]["zero_hand_l1"] = float(np.mean(np.abs(zero_pred - data["action"])))
    result = dict(status="COMPLETED", best_step=best_step, best_val_l1=best,
                  elapsed_s=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  checkpoint_sha256=sha(output / "best.pt"), metrics=metrics, history=history,
                  claim="offline contextual full-action fit only; GT execution upper bound pending")
    write(output / "result.json", result)
    manifest.update(status="COMPLETED", best_step=best_step,
                    checkpoint_sha256=result["checkpoint_sha256"], elapsed_s=result["elapsed_s"])
    write(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
