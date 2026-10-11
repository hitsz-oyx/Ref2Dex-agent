#!/usr/bin/env python3
"""Fixed-budget current-only scene/hand learner; an explicit architecture substitute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import sha256
from oakink_wm.pointwam_learning import (JointForecaster, batch, fit_contract,
    joint_loss, load_qualified, metric_prediction)


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def error_metrics(pack, prediction):
    out = {}
    for kind in ("scene", "hand"):
        target = pack[kind + "_displacement_m"].reshape(30, -1, 3)
        valid = pack[kind + "_target_valid"].reshape(30, -1)
        error = np.linalg.norm(prediction[kind] - target, axis=-1)
        out[kind + "_epe_mm"] = float(error[valid].mean() * 1000)
        end = valid[-1]
        out[kind + "_endpoint_mm"] = float(error[-1, end].mean() * 1000)
        out[kind + "_valid_targets"] = int(valid.sum())
        if kind == "scene":
            moving = end & (np.linalg.norm(target[-1], axis=-1) >= .02)
            mask = valid & moving[None]
            out["moving_scene_points"] = int(moving.sum())
            out["moving_scene_epe_mm"] = float(error[mask].mean() * 1000) if mask.any() else None
    return out


@torch.no_grad()
def evaluate(model, clips, contract, device, save=None):
    model.eval()
    rows = []
    for clip in clips:
        pack = clip["pack"]
        ids = np.arange(pack["scene_xyz_m"].shape[1])
        inputs, targets = batch([clip], [ids], contract, device)
        normalized = model(**inputs)
        if not all(torch.isfinite(x).all() for x in normalized.values()):
            raise ValueError("non-finite evaluation prediction")
        _, loss = joint_loss(normalized, targets)
        prediction = {kind: metric_prediction(normalized[kind], kind, contract)[0]
                      for kind in ("scene", "hand")}
        controls = {"model": prediction}
        controls["static"] = {kind: np.zeros_like(value) for kind, value in prediction.items()}
        controls["fit_mean"] = {kind: np.broadcast_to(
            np.asarray(contract[kind + "_mean_m"], np.float32)[:, None], value.shape)
            for kind, value in prediction.items()}
        rows.append(dict(episode=clip["episode"], partition=clip["partition"], pack=clip["path"],
                         loss={key: float(value) for key, value in loss.items()},
                         metrics={name: error_metrics(pack, value) for name, value in controls.items()}))
        if save and clip["partition"] == "dev_candidate":
            name = clip["episode"].replace("/", "__") + "-" + Path(clip["path"]).parent.name
            np.savez_compressed(save / (name + "-prediction.npz"), **prediction)
    # Equal weight to original records; overlapping windows do not become independent records.
    records = {}
    for row in rows:
        records.setdefault(row["episode"], []).append(row)
    record_metrics = {}
    for episode, group in records.items():
        record_metrics[episode] = dict(partition=group[0]["partition"], clips=len(group),
            loss={key: float(np.mean([r["loss"][key] for r in group])) for key in ("scene", "hand")},
            metrics={name: {key: (float(np.mean([r["metrics"][name][key] for r in group]))
                        if all(r["metrics"][name][key] is not None for r in group) else None)
                      for key in group[0]["metrics"][name]} for name in ("model", "static", "fit_mean")})
    macro = {}
    for partition in ("fit_candidate", "dev_candidate"):
        group = [r for r in record_metrics.values() if r["partition"] == partition]
        macro[partition] = dict(records=len(group),
            loss={key: float(np.mean([r["loss"][key] for r in group])) for key in ("scene", "hand")},
            metrics={name: {key: (float(np.mean([r["metrics"][name][key] for r in group]))
                       if all(r["metrics"][name][key] is not None for r in group) else None)
                     for key in group[0]["metrics"][name]} for name in ("model", "static", "fit_mean")})
    return dict(clips=rows, records=record_metrics, macro=macro)


def gpu_monitor(stop, samples):
    while not stop.is_set():
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
        samples.append(dict(time_s=time.time(), rows=result.stdout.strip().splitlines()))
        stop.wait(2)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(status="RUNNING", run_id=output.name, experiment_id="P-20261011-pointwam-joint-learning",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        work_version="git:" + subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        pid=os.getpid(), command=[sys.executable, *sys.argv],
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"), seed=args.seed,
        smoke=args.smoke, steps=args.steps, scene_points=args.scene_points, seconds_cap=args.seconds,
        output_bytes_cap=1024 ** 3, qualification=str(args.qualification.resolve()),
        qualification_sha256=sha256(args.qualification), architecture="small RGB/geometry set MLP; learned word table",
        author_architecture=False, training_inputs="current frame only; teacher future targets are supervision")
    write(output / "manifest.json", manifest)
    stop = threading.Event(); samples = []
    monitor = threading.Thread(target=gpu_monitor, args=(stop, samples), daemon=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("this Probe requires a free GPU")
        torch.set_num_threads(4)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = "cuda:0"
        clips = load_qualified(args.qualification)
        manifest["code_sha256"] = {str(path): sha256(path) for path in (
            Path(__file__), TASK / "src/oakink_wm/pointwam_learning.py", TASK / "src/oakink_wm/pointwam_data.py")}
        fit = [c for c in clips if c["partition"] == "fit_candidate"]
        contract = fit_contract(clips)
        write(output / "contract.json", contract)
        write(output / "admission.json", dict(experiment_id=manifest["experiment_id"],
            pack_flags_unchanged=True, authorization="user-approved bounded joint learning on qualified official TRAIN",
            sources=[dict(pack=c["path"], sha256=c["sha256"], source_episode=c["episode"],
                          permission="fit" if c["partition"] == "fit_candidate" else "evaluation_only") for c in clips]))
        model = JointForecaster(len(contract["vocabulary"]) + 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        manifest["parameters"] = sum(p.numel() for p in model.parameters())
        monitor.start()
        initial = evaluate(model, clips, contract, device)
        write(output / "initial_metrics.json", initial)
        rng = np.random.default_rng(args.seed)
        history = []; head_gradient_seen = {"scene": False, "hand": False}
        training_started = time.monotonic()
        for step in range(1, args.steps + 1):
            if time.monotonic() - started > args.seconds:
                raise TimeoutError("fixed wall-time budget exhausted")
            model.train()
            selected = [fit[i] for i in rng.choice(len(fit), 2, replace=False)]
            # Sampling uses only initially selected scene identities, never future validity/motion.
            ids = [rng.choice(c["pack"]["scene_xyz_m"].shape[1], args.scene_points, replace=False)
                   for c in selected]
            inputs, target = batch(selected, ids, contract, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = joint_loss(model(**inputs), target)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("non-finite gradient")
            for kind in head_gradient_seen:
                head = getattr(model, kind + "_head")
                head_gradient_seen[kind] |= any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                               for p in head.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                elapsed = time.monotonic() - training_started
                row = dict(step=step, elapsed_s=elapsed, eta_s=elapsed / step * (args.steps - step),
                           **{key: float(value.detach()) for key, value in parts.items()})
                history.append(row); write(output / "progress.json", row)
                print(json.dumps(row), flush=True)
        if not all(head_gradient_seen.values()):
            raise ValueError("joint supervision did not reach both heads")
        if any(sha256(Path(c["path"])) != c["sha256"] for c in clips):
            raise ValueError("admitted input changed during training")
        final = evaluate(model, clips, contract, device, output)
        write(output / "final_metrics.json", final)
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                        contract=contract, manifest=manifest), output / "final.pt")
        loaded = JointForecaster(len(contract["vocabulary"]) + 1).to(device)
        loaded.load_state_dict(torch.load(output / "final.pt", map_location="cpu", weights_only=True)["state_dict"])
        model.eval(); loaded.eval()
        check, _ = batch([clips[0]], [np.arange(clips[0]["pack"]["scene_xyz_m"].shape[1])], contract, device)
        with torch.no_grad():
            before, after = model(**check), loaded(**check)
        if not all(torch.equal(before[kind], after[kind]) for kind in before):
            raise ValueError("checkpoint replay changed predictions")
        fit_initial = initial["macro"]["fit_candidate"]["loss"]
        fit_final = final["macro"]["fit_candidate"]["loss"]
        fit_gate = all(fit_final[key] <= .5 * fit_initial[key] for key in ("scene", "hand"))
        dev = final["macro"]["dev_candidate"]["metrics"]
        dev_records = [r for r in final["records"].values() if r["partition"] == "dev_candidate"]
        hand_wins = sum(r["metrics"]["model"]["hand_epe_mm"] < r["metrics"]["static"]["hand_epe_mm"] for r in dev_records)
        moving = dev["model"]["moving_scene_epe_mm"]
        dev_gate = (dev["model"]["hand_epe_mm"] <= .95 * min(dev["static"]["hand_epe_mm"], dev["fit_mean"]["hand_epe_mm"])
                    and moving is not None and moving <= 1.05 * dev["static"]["moving_scene_epe_mm"] and hand_wins >= 2)
        result = "ENGINEERING_SMOKE" if args.smoke else ("PROMISING" if fit_gate and dev_gate else
                 "UNPROMISING" if fit_gate else "UNCLEAR")
        manifest.update(status="COMPLETED", result=result, fit_gate=bool(fit_gate), dev_gate=bool(dev_gate),
                        dev_hand_record_wins=hand_wins, both_heads_receive_gradient=head_gradient_seen,
                        checkpoint_replay_exact=True, loss_history=history,
                        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
    except Exception as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join(timeout=5)
        manifest["elapsed_s"] = time.monotonic() - started
        manifest["output_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if manifest["output_bytes"] > 1024 ** 3 or manifest["elapsed_s"] > args.seconds:
            manifest.update(status="FAILED", error="resource budget exceeded")
        write(output / "gpu_samples.json", samples)
        write(output / "manifest.json", manifest)
    if manifest["status"] != "COMPLETED":
        raise RuntimeError(manifest.get("error", "run incomplete"))
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--scene-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=247)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.steps <= 600 and 1 <= args.scene_points <= 1024 and 1 <= args.seconds <= 900):
        parser.error("budget capped at 600steps /1024points /900seconds")
    if not args.smoke and (args.steps != 600 or args.scene_points != 1024 or args.seed != 247):
        parser.error("main Probe uses the frozen 600step/1024point/247seed protocol")
    run(args)


if __name__ == "__main__":
    main()
