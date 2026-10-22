"""Score fixed-panel zero-motion and constant-velocity hand baselines.

This script deliberately imports the fixed-panel reader/validation and uses the
same hand error, validity, phase, first-eight, and contact-window masks as the
checkpoint scorer.  It also runs B1 in the same pass so paired frame records do
not depend on a second data traversal.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from evaluate_tau_fixed_panel import Accumulator, fixed_indices, read, sha256, validate_panel

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from ptv3_shared_wam_v6.panel_uncertainty import cluster_median_ci


def bootstrap_median_ci(values, seed, repeats=2000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"median_mm": None, "ci95_mm": [None, None], "bootstrap_repeats": 0}
    rng = np.random.default_rng(seed)
    medians = np.empty(repeats, dtype=np.float64)
    # Bounded blocks avoid allocating repeats x frames for the 12k-frame panel.
    for begin in range(0, repeats, 100):
        count = min(100, repeats - begin)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        medians[begin:begin + count] = np.median(values[sample], axis=1)
    return {
        "median_mm": float(np.median(values)),
        "ci95_mm": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "bootstrap_repeats": repeats,
        "bootstrap_seed": int(seed),
    }


def paired_summary(b1, b0, valid, seed, *, sequence_ids):
    """Return strict win rate and paired B1-B0 frame difference statistics."""
    valid = np.asarray(valid, dtype=bool)
    diff = np.asarray(b1, dtype=np.float64)[valid] - np.asarray(b0, dtype=np.float64)[valid]
    wins = int(np.sum(diff < 0.0))
    losses = int(np.sum(diff > 0.0))
    ties = int(np.sum(diff == 0.0))
    comparable = wins + losses
    result = {
        "frame_count": int(diff.size),
        "wins_b1_lt_b0a": wins,
        "losses_b1_gt_b0a": losses,
        "ties": ties,
        "win_rate_excluding_ties": float(wins / comparable) if comparable else None,
        "tie_rate": float(ties / len(diff)) if len(diff) else None,
        "mean_paired_difference_mm_b1_minus_b0a": float(np.mean(diff)) if len(diff) else None,
        "p95_paired_difference_mm_b1_minus_b0a": float(np.percentile(diff, 95)) if len(diff) else None,
    }
    result["paired_difference_median_frame_bootstrap"] = bootstrap_median_ci(diff, seed)
    groups = np.broadcast_to(np.asarray(sequence_ids)[:, None], valid.shape)[valid]
    result["paired_difference_median_bootstrap"] = cluster_median_ci(
        diff, groups, cluster_universe=sequence_ids, seed=seed)
    return result


def frame_errors(hand_error, hand_valid):
    valid = hand_valid.bool()
    count = valid.sum(dim=(2, 3))
    values = (hand_error * valid).sum(dim=(2, 3)) / count.clamp_min(1)
    return values.detach().cpu().numpy(), (count > 0).detach().cpu().numpy()


def finish_metrics(values_by_name):
    return {name: acc.finish() for name, acc in values_by_name.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True,
                        help="Existing B1 receipt; it is read but never overwritten here")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=TASK / "configs/tau-fixed-panel-v1.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 16:
        raise ValueError("batch size must be in [1,16]")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refuse to overwrite output: %s" % output)
    baseline_path = args.baseline_json.resolve()
    baseline = read(baseline_path)
    if baseline.get("schema") != "ref2dex.tau-fixed-panel-score.v1":
        raise ValueError("baseline JSON is not the fixed-panel B1 receipt")
    if shutil.disk_usage(ROOT).free < 20 * 2 ** 30:
        raise RuntimeError("20 GiB storage floor is not available")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0 <= args.gpu <= 7:
        raise ValueError("physical GPU must be in [0,7]")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from ptv3_shared_wam_v6.data import V6Windows, collate
    from ptv3_shared_wam_v6.model import SharedWAMV6

    panel = read(args.panel)
    anchor_path = (ROOT / panel["anchor"]).resolve()
    anchor = read(anchor_path)
    rows = validate_panel(args.panel, panel, anchor)
    event_recipe = (ROOT / panel["event_recipe"]).resolve()
    stats_path = (ROOT / panel["stats"]).resolve()
    dataset = V6Windows(event_recipe, "train", cache_size=8)
    positions = fixed_indices(dataset, rows)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if int(config.get("flow_eval_seed", -1)) != int(panel["flow_eval_seed"]):
        raise ValueError("checkpoint flow_eval_seed differs from fixed panel")
    if not config.get("flow_matching"):
        raise ValueError("checkpoint is not the flow-matching v6 model under this panel")
    torch.manual_seed(int(panel["seed"]))
    np.random.seed(int(panel["seed"]))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    model = SharedWAMV6(read(stats_path), config).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    step = int(checkpoint["step"])
    del checkpoint

    started = time.monotonic()
    first_n = int(panel["metrics"]["first_n_frames"])
    threshold = float(panel["metrics"]["contact_confidence_threshold"])
    names = ("overall", "first_8_frames", "contact_window")
    totals = {"B1": {name: Accumulator() for name in names},
              "B0a": {name: Accumulator() for name in names},
              "B0b": {name: Accumulator() for name in names}}
    phase_totals = {method: {} for method in totals}
    n = len(rows)
    b1_frame = np.full((n, 24), np.nan, dtype=np.float32)
    b0a_frame = np.full((n, 24), np.nan, dtype=np.float32)
    b0b_frame = np.full((n, 24), np.nan, dtype=np.float32)
    frame_valid = np.zeros((n, 24), dtype=bool)
    contact_frames = 0
    batches = 0
    deterministic_check = None
    for begin in range(0, n, args.batch_size):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("fixed-panel inference exceeded --max-seconds")
        end = min(begin + args.batch_size, n)
        batch = {key: value.cuda(non_blocking=True)
                 for key, value in collate([dataset[positions[i]] for i in range(begin, end)]).items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            prediction = model(batch)
            if deterministic_check is None:
                repeat = model(batch)["trajectory"]
                deterministic_check = bool(torch.equal(prediction["trajectory"], repeat))
            target = batch["trajectory_target"].float()
            hand_valid = batch["trajectory_valid"].bool() & batch["start_hand_valid"].bool()[:, None]
            b1_error = (prediction["trajectory"].float() - target).norm(dim=-1) * 1000.
            # trajectory_target is a displacement from the history's last frame.
            # B0a holds that frame, hence zero displacement.  The first three
            # history channels are m/s, so B0b advances one frame by v/30.
            b0a_prediction = torch.zeros_like(target)
            velocity = batch["start_hand_history"][..., :3].float() / float(panel["preprocessing"]["fps"])
            horizon = torch.arange(1, target.shape[1] + 1, device=target.device, dtype=target.dtype)
            b0b_prediction = velocity[:, None] * horizon[None, :, None, None, None]
            b0a_error = (b0a_prediction - target).norm(dim=-1) * 1000.
            b0b_error = (b0b_prediction - target).norm(dim=-1) * 1000.
            contact_frame = (batch["interaction_valid"].bool() &
                             (batch["contact_target"].float() > threshold)).any(dim=(1, 3))
            first = torch.zeros_like(contact_frame)
            first[:, :first_n] = True
            masks = {
                "overall": hand_valid,
                "first_8_frames": hand_valid & first[:, :, None, None],
                "contact_window": hand_valid & contact_frame[:, :, None, None],
            }
            for method, error in (("B1", b1_error), ("B0a", b0a_error), ("B0b", b0b_error)):
                for name, mask in masks.items():
                    totals[method][name].add(error[mask].detach().cpu().numpy())
                for index, phase in enumerate(rows[begin:end][i] for i in range(end - begin)):
                    phase_name = phase["phase"]
                    phase_totals[method].setdefault(phase_name, Accumulator()).add(
                        error[index][hand_valid[index]].detach().cpu().numpy())
            b1_values, valid_frames = frame_errors(b1_error, hand_valid)
            b0a_values, _ = frame_errors(b0a_error, hand_valid)
            b0b_values, _ = frame_errors(b0b_error, hand_valid)
            b1_frame[begin:end] = b1_values
            b0a_frame[begin:end] = b0a_values
            b0b_frame[begin:end] = b0b_values
            frame_valid[begin:end] = valid_frames
            contact_frames += int(contact_frame.sum())
            batches += 1
        print(json.dumps({"batch": batches, "rows_scored": end,
                          "elapsed_s": time.monotonic() - started}), flush=True)
        del batch, prediction, b1_error, b0a_error, b0b_error, hand_valid
        torch.cuda.empty_cache()

    subset_valid = {
        "overall": frame_valid,
        "first_8_frames": frame_valid & (np.arange(24)[None, :] < first_n),
    }
    # Reconstruct the exact contact-frame mask once on CPU from the data.  This
    # keeps pairing aligned with the scorer's contact definition and row order.
    contact_mask = np.zeros((n, 24), dtype=bool)
    for begin in range(0, n, args.batch_size):
        end = min(begin + args.batch_size, n)
        batch = collate([dataset[positions[i]] for i in range(begin, end)])
        contact_mask[begin:end] = ((batch["interaction_valid"].bool() &
                                    (batch["contact_target"].float() > threshold)).any(dim=(1, 3)).numpy())
    subset_valid["contact_window"] = frame_valid & contact_mask
    pairwise = {}
    sequence_ids = np.array([row['sequence_id'] for row in rows])
    for name, valid in subset_valid.items():
        pairwise[name] = paired_summary(b1_frame, b0a_frame, valid, 172000 + len(name), sequence_ids=sequence_ids)
    for phase in sorted(set(row["phase"] for row in rows)):
        valid = frame_valid & np.asarray([[row["phase"] == phase] * 24 for row in rows], dtype=bool)
        pairwise["phase/" + phase] = paired_summary(b1_frame, b0a_frame, valid, 173000 + len(phase), sequence_ids=sequence_ids)

    metrics = {method: finish_metrics(totals[method]) for method in totals}
    metrics["B1"]["phases"] = finish_metrics(phase_totals["B1"])
    metrics["B0a"]["phases"] = finish_metrics(phase_totals["B0a"])
    metrics["B0b"]["phases"] = finish_metrics(phase_totals["B0b"])
    mean_differences = {}
    for name in names:
        mean_differences[name] = {
            "B1_minus_B0a_mm": metrics["B1"][name]["mean_mm"] - metrics["B0a"][name]["mean_mm"],
            "B1_minus_B0b_mm": metrics["B1"][name]["mean_mm"] - metrics["B0b"][name]["mean_mm"],
        }
    report = {
        "schema": "ref2dex.tau-fixed-panel-trivial-baselines.v2",
        "status": "COMPLETED",
        "experiment_id": panel["experiment_id"],
        "run_id": output.name,
        "task": panel["task"],
        "panel": {"config": str(args.panel.resolve()), "anchor": str(anchor_path),
                  "anchor_id": anchor["anchor_id"], "anchor_sha256": sha256(anchor_path),
                  "rows": len(rows), "seed": panel["seed"],
                  "flow_eval_seed": panel["flow_eval_seed"], "event_condition_dropout": False},
        "preprocessing": panel["preprocessing"],
        "checkpoint_B1": {"path": str(args.checkpoint.resolve()), "step": step,
                          "schema": config.get("schema"), "flow_steps": config.get("flow_steps"),
                          "config_flow_eval_seed": config.get("flow_eval_seed")},
        "baseline_reference": {"path": str(baseline_path), "sha256_before": sha256(baseline_path),
                                "run_id": baseline["run_id"]},
        "inference_budget": {"B1_flow_steps": int(config.get("flow_steps", 4)),
                             "B0a_flow_steps": 0, "B0b_flow_steps": 0,
                             "note": "B0 predictors are analytic and do not run Flow Matching integration."},
        "predictors": {
            "B0a": "future hand displacement is zero; future position equals history last hand position",
            "B0b": "future displacement is k * (last hand position - previous hand position), k=1..24",
        },
        "metrics": metrics,
        "mean_edge_differences_mm": mean_differences,
        "paired_frame_analysis": {"difference_definition": "B1 frame mean error minus B0a frame mean error; negative favors B1",
                                   "frame_error": "mean endpoint error over the same valid hand points in each sample-frame",
                                   "subsets": pairwise},
        "contact_window": {"threshold": threshold, "frame_count": contact_frames,
                            "definition": panel["metrics"]["contact_window"]},
        "deterministic_repeat_check": deterministic_check,
        "deterministic_algorithms": True,
        "environment": {"python": sys.version, "torch": torch.__version__,
                         "cuda": torch.version.cuda, "visible_physical_gpu": args.gpu,
                         "device_name": torch.cuda.get_device_name(0),
                         "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                cwd=ROOT, text=True).strip()},
        "timing": {"elapsed_s": time.monotonic() - started, "batches": batches,
                   "batch_size": args.batch_size, "max_seconds": args.max_seconds},
        "scope": "fixed TRAIN anchor prediction quality; no validation/generalization or executor claim",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'paired-frame-records.npz', b1_error_mm=b1_frame,
        b0a_error_mm=b0a_frame, b0b_error_mm=b0b_frame, frame_valid=frame_valid,
        contact_frame=contact_mask, sequence_id=sequence_ids,
        phase=np.array([row['phase'] for row in rows]),
        slot=np.array([row['slot'] for row in rows]), tick=np.array([row['tick'] for row in rows]))
    (output / "baseline-b0.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output / "baseline-b0.json"),
                      "elapsed_s": report["timing"]["elapsed_s"],
                      "B1_overall_mean_mm": metrics["B1"]["overall"]["mean_mm"],
                      "B0a_overall_mean_mm": metrics["B0a"]["overall"]["mean_mm"],
                      "B0b_overall_mean_mm": metrics["B0b"]["overall"]["mean_mm"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
