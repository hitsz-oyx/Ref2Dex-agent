"""Fit a matched v2 contextual baseline and query-time contact-context arm.

This is an offline Probe only.  Both arms use the same structured rollout
windows, episode split, seed, batches, optimizer, and shared initialization.
The augmented arm receives exactly ten fields measured at query time ``t``;
future contact/state fields are never passed to the model.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.hand_action_retargeter import (
    CONTACT_CONTEXT_DIM, CONTACT_CONTEXT_FIELDS, CONTACT_CONTEXT_SCHEMA,
    CONTEXT_SCHEMA, ContextHandActionRetargeter, Standardizer)
from train_hand_action_retargeter import concat_split, episode_disjoint, sha, write


HORIZON = 24


def action_metrics(prediction, target, mask=None):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target must be [N,24,18]")
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(target),):
        raise ValueError("metric mask shape mismatch")
    if not mask.any():
        return {"n": 0}
    error = np.abs(prediction[mask] - target[mask])
    return dict(
        n=int(mask.sum()),
        l1=float(error.mean()),
        wrist_translation_mae=float(error[..., :3].mean()),
        wrist_rotation_mae=float(error[..., 3:6].mean()),
        finger_mae=float(error[..., 6:].mean()),
        horizon0_l1=float(error[:, 0].mean()),
        horizon23_l1=float(error[:, -1].mean()),
        horizon0_finger_mae=float(error[:, 0, 6:].mean()),
        horizon23_finger_mae=float(error[:, -1, 6:].mean()),
    )


def evaluate(model, tensors, raw, stats, device, use_contact, contact_override=None):
    model.eval()
    with torch.inference_mode():
        contact = tensors["contact"] if use_contact else None
        if contact_override is not None:
            contact = contact_override
        output = model(tensors["hand"], tensors["state"], tensors["context"], contact)
        prediction = stats["action"].decode(output.cpu().numpy())
    return prediction


def fit_val_loss(model, tensors, name, use_contact):
    model.eval()
    with torch.inference_mode():
        contact = tensors[name]["contact"] if use_contact else None
        output = model(tensors[name]["hand"], tensors[name]["state"],
                       tensors[name]["context"], contact)
        return float((output - tensors[name]["action"]).abs().mean())


def save_checkpoint(path, schema, model, stats, manifest, best_step, best_val_l1,
                    contact_dim, arm):
    statistics = {key: value.as_dict() for key, value in stats.items()
                  if key != "contact" or contact_dim}
    torch.save(dict(schema=schema, arm=arm, width=model.width, contact_dim=contact_dim,
                    state_dict=model.state_dict(), statistics=statistics,
                    best_step=best_step, best_val_l1=best_val_l1,
                    manifest=dict(manifest, arm=arm, model_schema=schema)), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=406)
    args = parser.parse_args()
    if not 1 <= args.steps <= 4000 or not 30 <= args.seconds <= 1200:
        parser.error("bounded contact fit budget required")
    if not 1 <= args.stride <= 24 or not 32 <= args.batch <= 1024:
        parser.error("invalid stride/batch")
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned contact fit output required")
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
        "--format=csv,noheader"], text=True).strip()
    if occupied:
        raise RuntimeError("GPU occupied: " + occupied)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    splits = {}
    split_manifests = {}
    frozen = {
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str((TASK / "src/consequence_evaluator/hand_action_retargeter.py").resolve()):
            sha(TASK / "src/consequence_evaluator/hand_action_retargeter.py"),
        str((TASK / "tools/run/train_hand_action_retargeter.py").resolve()):
            sha(TASK / "tools/run/train_hand_action_retargeter.py"),
    }
    for name in ("train", "val", "test"):
        value, meta, hashes = concat_split(
            getattr(args, name), args.stride, include_contact=True, skip_reset=True)
        splits[name] = value
        split_manifests[name] = meta
        frozen.update(hashes)
    episode_disjoint(splits)
    actor_hashes = {m["actor_sha256"] for group in split_manifests.values() for m in group
                    if m.get("actor_sha256") is not None}
    if len(actor_hashes) != 1:
        raise ValueError("all structured splits must use one frozen actor")
    if any(data["contact"].shape[1:] != (CONTACT_CONTEXT_DIM,)
           for data in splits.values()):
        raise ValueError("contact feature dimension drift")

    train = splits["train"]
    stats = {
        "hand": Standardizer.fit(train["hand"]),
        "state": Standardizer.fit(train["state"]),
        "context": Standardizer.fit(train["context"]),
        "contact": Standardizer.fit(train["contact"]),
        "action": Standardizer.fit(train["action"]),
    }
    tensors = {}
    for name, data in splits.items():
        tensors[name] = {
            key: torch.as_tensor(stats[key].encode(data[key]), device=device)
            for key in ("hand", "state", "context", "contact", "action")
        }

    output.mkdir(parents=True)
    manifest = dict(
        schema=CONTACT_CONTEXT_SCHEMA, baseline_schema=CONTEXT_SCHEMA,
        status="RUNNING", run_id=output.name, task="consequence-evaluator",
        route="ref7_2-contact-context", git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        gpu=args.gpu, seed=args.seed, steps_cap=args.steps, seconds_cap=args.seconds,
        batch=args.batch, stride=args.stride,
        splits={name: dict(rollouts=[m["run_id"] for m in split_manifests[name]],
                           episodes=int(len(np.unique(splits[name]["episode"]))),
                           windows=int(len(splits[name]["hand"])))
                for name in splits},
        input_sha256=frozen,
        input_contract=("future hand displacement t+1:t+24 + q_t,dq_t + current "
                        "hand-object geometry + previous native command + current "
                        "contact proxies at t only"),
        target_contract="captured full native action A_t:t+24, no actor action at inference",
        contact_contract=dict(
            fields=list(CONTACT_CONTEXT_FIELDS),
            dim=CONTACT_CONTEXT_DIM,
            state_frame="query t after command t-1 and before command t",
            object_velocity_frame="world frame; linear m/s, angular rad/s",
            pair_semantics="configured five-body/object net-force threshold proxy, not exact collision pair",
            reset_frame_policy="frame 0 excluded from windows because contact cache is reset-invalid",
            transform="raw values retained; train-only mean/std standardization, no clipping",
            future_contact_exposed=False),
        normalization="train-only per-coordinate hand/state/context/contact/action",
        action_groups={"wrist_translation": [0, 1, 2], "wrist_rotation": [3, 4, 5],
                       "finger": list(range(6, 18))},
    )
    write(output / "manifest.json", manifest)

    baseline = ContextHandActionRetargeter().to(device)
    initial_baseline = {key: value.detach().cpu().clone()
                        for key, value in baseline.state_dict().items()}
    augmented = ContextHandActionRetargeter(contact_dim=CONTACT_CONTEXT_DIM).to(device)
    missing, unexpected = augmented.load_state_dict(initial_baseline, strict=False)
    expected_missing = {"contact.weight", "contact.bias"}
    if set(missing) != expected_missing or unexpected:
        raise ValueError("shared v2/v3 initialization contract drift")
    if not all(torch.count_nonzero(value).item() == 0
               for name, value in augmented.state_dict().items() if name.startswith("contact.")):
        raise ValueError("contact branch must start at zero for matched initialization")
    augmented_initial = {key: value.detach().cpu().clone()
                         for key, value in augmented.state_dict().items()}
    torch.save(dict(schema="ref2dex.contact-context-initial.v1",
                    baseline=initial_baseline, augmented=augmented_initial), output / "initial.pt")
    with torch.inference_mode():
        probe_count = min(4, len(tensors["train"]["hand"]))
        base_probe = baseline(
            tensors["train"]["hand"][:probe_count], tensors["train"]["state"][:probe_count],
            tensors["train"]["context"][:probe_count])
        augmented_probe = augmented(
            tensors["train"]["hand"][:probe_count], tensors["train"]["state"][:probe_count],
            tensors["train"]["context"][:probe_count],
            tensors["train"]["contact"][:probe_count] * 0.)
        if not torch.equal(base_probe, augmented_probe):
            raise ValueError("zero-contact augmented arm is not matched to v2 initialization")

    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=3e-4, weight_decay=1e-4)
    augmented_optimizer = torch.optim.AdamW(augmented.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best = {"baseline": float("inf"), "augmented": float("inf")}
    best_step = {"baseline": None, "augmented": None}
    history = []
    started = time.monotonic()

    for step in range(1, args.steps + 1):
        elapsed = time.monotonic() - started
        if elapsed > args.seconds:
            break
        index = torch.as_tensor(rng.integers(len(train["hand"]), size=args.batch), device=device)
        baseline.train()
        baseline_prediction = baseline(
            tensors["train"]["hand"][index], tensors["train"]["state"][index],
            tensors["train"]["context"][index])
        baseline_loss = (baseline_prediction - tensors["train"]["action"][index]).abs().mean()
        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.)
        baseline_optimizer.step()

        augmented.train()
        augmented_prediction = augmented(
            tensors["train"]["hand"][index], tensors["train"]["state"][index],
            tensors["train"]["context"][index], tensors["train"]["contact"][index])
        augmented_loss = (augmented_prediction - tensors["train"]["action"][index]).abs().mean()
        augmented_optimizer.zero_grad(set_to_none=True)
        augmented_loss.backward()
        torch.nn.utils.clip_grad_norm_(augmented.parameters(), 1.)
        augmented_optimizer.step()
        if not torch.isfinite(baseline_loss) or not torch.isfinite(augmented_loss):
            raise FloatingPointError("nonfinite matched contact fit loss")

        if step == 1 or step % 200 == 0 or step == args.steps:
            baseline_val = fit_val_loss(baseline, tensors, "val", False)
            augmented_val = fit_val_loss(augmented, tensors, "val", True)
            row = dict(step=step, baseline_train_l1=float(baseline_loss),
                       augmented_train_l1=float(augmented_loss),
                       baseline_val_l1=baseline_val, augmented_val_l1=augmented_val,
                       elapsed_s=time.monotonic() - started,
                       eta_s=max(0., (time.monotonic() - started) / step * (args.steps - step)))
            print(json.dumps(row), flush=True)
            history.append(row)
            if baseline_val < best["baseline"]:
                best["baseline"] = baseline_val
                best_step["baseline"] = step
                save_checkpoint(output / "baseline.pt", CONTEXT_SCHEMA, baseline,
                                stats, manifest, step, baseline_val, 0, "matched-v2-baseline")
            if augmented_val < best["augmented"]:
                best["augmented"] = augmented_val
                best_step["augmented"] = step
                save_checkpoint(output / "contact.pt", CONTACT_CONTEXT_SCHEMA, augmented,
                                stats, manifest, step, augmented_val, CONTACT_CONTEXT_DIM,
                                "contact-v3-augmented")

    if best_step["baseline"] is None or best_step["augmented"] is None:
        raise RuntimeError("matched contact fit ended before a validation checkpoint")
    baseline_payload = torch.load(output / "baseline.pt", map_location=device, weights_only=False)
    contact_payload = torch.load(output / "contact.pt", map_location=device, weights_only=False)
    baseline.load_state_dict(baseline_payload["state_dict"], strict=True)
    augmented.load_state_dict(contact_payload["state_dict"], strict=True)

    predictions = {
        "baseline": evaluate(baseline, tensors["test"], splits["test"], stats, device, False),
        "augmented": evaluate(augmented, tensors["test"], splits["test"], stats, device, True),
    }
    test_contact_mean = torch.zeros_like(tensors["test"]["contact"])
    permutation = torch.as_tensor(np.random.default_rng(args.seed + 1).permutation(
        len(splits["test"]["contact"])), device=device)
    predictions["augmented_mean_contact"] = evaluate(
        augmented, tensors["test"], splits["test"], stats, device, True, test_contact_mean)
    predictions["augmented_shuffled_contact"] = evaluate(
        augmented, tensors["test"], splits["test"], stats, device, True,
        tensors["test"]["contact"][permutation])

    test_data = splits["test"]
    pair = test_data["contact"][:, 0] > .5
    gap = test_data["contact"][:, 1]
    phase = test_data["phase"]
    masks = {
        "all": np.ones(len(test_data["action"]), dtype=bool),
        "pair_true": pair,
        "pair_false": ~pair,
        "contact_onset": (test_data["tick"] >= 96) & (test_data["tick"] < 192),
        "hold": test_data["tick"] >= 240,
        "phase_0": phase == 0,
        "phase_1": phase == 1,
        "phase_2": phase == 2,
        "gap_le_0.01": gap <= .01,
        "gap_0.01_0.05": (gap > .01) & (gap <= .05),
        "gap_gt_0.05": gap > .05,
        "near_gap": gap <= .01,
    }
    metrics = {}
    for arm, prediction in predictions.items():
        metrics[arm] = {name: action_metrics(prediction, test_data["action"], mask)
                        for name, mask in masks.items()}
    base_finger = metrics["baseline"]["all"]["finger_mae"]
    augmented_finger = metrics["augmented"]["all"]["finger_mae"]
    relative_finger_improvement = (base_finger - augmented_finger) / base_finger
    gate = dict(
        relative_finger_improvement=float(relative_finger_improvement),
        required_relative_finger_improvement=.10,
        contact_onset_not_worse=(metrics["augmented"]["contact_onset"]["finger_mae"]
                                  <= metrics["baseline"]["contact_onset"]["finger_mae"]),
        hold_not_worse=(metrics["augmented"]["hold"]["finger_mae"]
                        <= metrics["baseline"]["hold"]["finger_mae"]),
        pair_positive_not_worse=(metrics["augmented"]["pair_true"]["finger_mae"]
                                 <= metrics["baseline"]["pair_true"]["finger_mae"]),
        near_gap_not_worse=(metrics["augmented"]["near_gap"]["finger_mae"]
                            <= metrics["baseline"]["near_gap"]["finger_mae"]),
        horizon23_not_worse_5pct=(metrics["augmented"]["all"]["horizon23_l1"]
                                  <= 1.05 * metrics["baseline"]["all"]["horizon23_l1"]),
    )
    gate["passed"] = bool(
        gate["relative_finger_improvement"] >= gate["required_relative_finger_improvement"]
        and gate["contact_onset_not_worse"] and gate["hold_not_worse"]
        and gate["pair_positive_not_worse"] and gate["near_gap_not_worse"]
        and gate["horizon23_not_worse_5pct"])
    result = dict(
        status="PROMISING" if gate["passed"] else "UNPROMISING",
        schema=CONTACT_CONTEXT_SCHEMA, best_step=best_step,
        best_val_l1=best, elapsed_s=time.monotonic() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        checkpoint_sha256={"baseline": sha(output / "baseline.pt"),
                           "contact": sha(output / "contact.pt")},
        initial_sha256=sha(output / "initial.pt"), metrics=metrics, gate=gate,
        contact_use=dict(
            mean_contact=metrics["augmented_mean_contact"]["all"],
            shuffled_contact=metrics["augmented_shuffled_contact"]["all"]),
        history=history,
        claim=("matched offline current-state contact-context Probe only; no native "
               "execution, H-to-hand, PointWorld, evaluator, or Cm claim"))
    write(output / "result.json", result)
    manifest.update(status="COMPLETED", best_step=best_step,
                    checkpoint_sha256=result["checkpoint_sha256"],
                    initial_sha256=result["initial_sha256"], elapsed_s=result["elapsed_s"],
                    result_status=result["status"])
    write(output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in result.items() if key != "history"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
