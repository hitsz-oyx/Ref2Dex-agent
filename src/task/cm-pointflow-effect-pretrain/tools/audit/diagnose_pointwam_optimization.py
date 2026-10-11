#!/usr/bin/env python3
"""Separate learning-rate and multi-clip effects in the PointWAM learner.

This is a bounded engineering diagnosis.  It reuses the frozen qualified
cohort and normalization contract, never reads dev targets during training,
and reports only fit-loss behavior.  The main protocol uses two random fit
clips and 1024 points; the single-clip arms use one fixed fit clip and 512
fixed initial scene identities so the two variables can be separated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import sha256
from oakink_wm.pointwam_learning import JointForecaster, batch, joint_loss, load_qualified


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measure(model, clips, ids, contract, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = batch(clips, ids, contract, device)
        _, parts = joint_loss(model(**inputs), targets)
    return {key: float(value) for key, value in parts.items()}


def train_case(name, selector, eval_clips, eval_ids, contract, *, lr, seed,
               steps, device, output):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    model = JointForecaster(len(contract["vocabulary"]) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    initial = measure(model, eval_clips, eval_ids, contract, device)
    history = []
    for step in range(1, steps + 1):
        model.train()
        train_clips, train_ids = selector(step)
        inputs, targets = batch(train_clips, train_ids, contract, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = joint_loss(model(**inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError(f"{name}: non-finite loss at step {step}")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError(f"{name}: non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append(dict(step=step, **{key: float(value.detach()) for key, value in parts.items()}))
    final = measure(model, eval_clips, eval_ids, contract, device)
    return dict(name=name, learning_rate=lr, seed=seed, steps=steps, initial=initial,
                final=final, ratios={key: final[key] / initial[key] for key in initial},
                history=history, peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="completed pointwam-joint-learning run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--scene-points", type=int, default=1024)
    parser.add_argument("--single-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(status="RUNNING", run_id=args.output.name,
                    experiment_id="P-20261011-pointwam-optimization-diagnosis",
                    source_run=str(args.run.resolve()), seed=args.seed,
                    steps=args.steps, scene_points=args.scene_points,
                    single_points=args.single_points, seconds_cap=args.seconds,
                    cuda_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                    command=[sys.executable, *sys.argv],
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    code_sha256=sha256(Path(__file__)))
    write(args.output / "manifest.json", manifest)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for this diagnosis")
        torch.set_num_threads(4)
        source = json.loads((args.run / "manifest.json").read_text())
        qualification = Path(source["qualification"])
        if source["status"] != "COMPLETED" or sha256(qualification) != source["qualification_sha256"]:
            raise ValueError("source run or qualification identity changed")
        contract_path = args.run / "contract.json"
        contract = json.loads(contract_path.read_text())
        clips = load_qualified(qualification)
        fit = [c for c in clips if c["partition"] == "fit_candidate"]
        if contract["fit_pack_hashes"] != [c["sha256"] for c in fit]:
            raise ValueError("fit normalization identity changed")
        device = "cuda:0"
        rng = np.random.default_rng(args.seed)
        fixed_single = [rng.choice(fit[0]["pack"]["scene_xyz_m"].shape[1], args.single_points,
                                   replace=False)]
        fixed_multi = [rng.choice(c["pack"]["scene_xyz_m"].shape[1], args.scene_points,
                                  replace=False) for c in fit]
        np.save(args.output / "single_fixed_scene_ids.npy", fixed_single[0])
        np.save(args.output / "multi_fixed_scene_ids.npy", np.asarray(fixed_multi, dtype=object),
                allow_pickle=True)

        def single_selector(_step):
            return [fit[0]], fixed_single

        def full_selector(_step):
            return fit, [fixed_multi[i] for i in range(len(fit))]

        def make_pair_fixed_selector():
            # Preserve the original two-clip batch schedule but reuse one
            # fixed point subset per clip. This isolates point re-sampling
            # noise from clip-selection noise.
            local = np.random.default_rng(args.seed)

            def selector(_step):
                selected = local.choice(len(fit), 2, replace=False)
                return ([fit[i] for i in selected],
                        [fixed_multi[i] for i in selected])

            return selector

        def make_main_selector():
            # Match the original single continuous RNG stream and its
            # two-clip/1024-point sampling rule. Each learning-rate arm gets
            # a fresh stream so the arms differ only in learning rate.
            local = np.random.default_rng(args.seed)

            def selector(_step):
                selected = local.choice(len(fit), 2, replace=False)
                selected_clips = [fit[i] for i in selected]
                ids = [local.choice(c["pack"]["scene_xyz_m"].shape[1], args.scene_points,
                                    replace=False) for c in selected_clips]
                return selected_clips, ids

            return selector

        cases = []
        # Single-clip arms use the same fixed IDs and differ only in learning rate.
        for lr in (1e-4, 1e-3):
            cases.append(train_case(f"single_fixed_lr_{lr:g}", single_selector,
                                     [fit[0]], fixed_single, contract, lr=lr, seed=args.seed,
                                     steps=args.steps, device=device, output=args.output))
        # Multi-clip arms use the main stochastic two-clip sampling protocol.
        for lr in (1e-4, 1e-3):
            cases.append(train_case(f"multi_random_lr_{lr:g}", make_main_selector(),
                                     fit, fixed_multi, contract, lr=lr, seed=args.seed,
                                     steps=args.steps, device=device, output=args.output))
        # Pair-fixed arms preserve the two-clip batch size and only remove
        # point-ID resampling noise.
        for lr in (1e-4, 1e-3):
            cases.append(train_case(f"pair_fixed_lr_{lr:g}", make_pair_fixed_selector(),
                                     fit, fixed_multi, contract, lr=lr, seed=args.seed,
                                     steps=args.steps, device=device, output=args.output))
        # A deterministic full-fit arm checks whether cross-clip interference
        # remains when point and clip sampling noise is removed.
        for lr in (1e-4, 1e-3):
            cases.append(train_case(f"full_fixed_lr_{lr:g}", full_selector,
                                     fit, fixed_multi, contract, lr=lr, seed=args.seed,
                                     steps=args.steps, device=device, output=args.output))
        manifest.update(status="COMPLETED", cases=cases,
                        interpretation=("fit-only diagnosis; no dev evaluation, "
                                         "method or policy claim"),
                        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_s"] = time.monotonic() - started
        write(args.output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
