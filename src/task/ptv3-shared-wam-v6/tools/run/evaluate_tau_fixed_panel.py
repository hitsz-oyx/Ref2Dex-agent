"""Score any v6 checkpoint on the immutable TRAIN tau anchor panel.

The panel is checkpoint-independent: rows, seed, event handling and the v6
reader/collate path are fixed by ``configs/tau-fixed-panel-v1.json``.  This
command only performs inference and writes aggregate, machine-readable JSON;
it never updates or writes a checkpoint.
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

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Accumulator:
    def __init__(self):
        self.values = []

    def add(self, values):
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size:
            self.values.append(values)

    def finish(self):
        if not self.values:
            return {"count": 0, "mean_mm": None, "median_mm": None, "p95_mm": None}
        values = np.concatenate(self.values)
        return {
            "count": int(values.size),
            "mean_mm": float(values.mean()),
            "median_mm": float(np.percentile(values, 50)),
            "p95_mm": float(np.percentile(values, 95)),
        }


def fixed_indices(dataset, rows):
    index = np.asarray(dataset.index)
    lookup = {}
    for position, key in enumerate(map(tuple, index.tolist())):
        if key in lookup:
            raise ValueError("dataset index contains duplicate sample identity: %r" % (key,))
        lookup[key] = position
    positions = []
    for row in rows:
        key = (int(row["sequence_id"]), int(row["slot"]), int(row["tick"]))
        if key not in lookup:
            raise ValueError("anchor row is absent from frozen TRAIN index: %r" % (key,))
        positions.append(lookup[key])
    return positions


def validate_panel(panel_path, panel, anchor):
    if panel.get("schema") != "ref2dex.tau-fixed-panel.v1":
        raise ValueError("unsupported fixed-panel schema")
    if not panel.get("checkpoint_independent") or not panel.get("fixed_order"):
        raise ValueError("panel must declare checkpoint-independent fixed order")
    if anchor.get("schema") != "ref2dex.tau-train-anchor.v1":
        raise ValueError("unsupported train-anchor schema")
    if anchor.get("status") != "FIXED_BEFORE_EXPANSION":
        raise ValueError("train anchor is not frozen")
    rows = anchor.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty train anchor")
    phases = set(panel["metrics"]["phase_labels"])
    if not {"approach", "close", "contact", "hold"}.issubset(phases):
        raise ValueError("required phase labels missing")
    if int(panel["preprocessing"]["history"]) != 4 or int(panel["preprocessing"]["horizon"]) != 24:
        raise ValueError("panel preprocessing does not match v6 H4/K24")
    source = Path(anchor["source_manifest"]).resolve()
    expected = (ROOT / panel["data_manifest"]).resolve()
    if source != expected:
        raise ValueError("anchor source manifest differs from fixed panel data manifest")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=TASK / "configs/tau-fixed-panel-v1.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--max-seconds", type=float, default=1140.)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 16:
        raise ValueError("batch size must be in [1,16]")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refuse to overwrite output: %s" % output)
    if shutil.disk_usage(ROOT).free < 20 * 2 ** 30:
        raise RuntimeError("20 GiB storage floor is not available")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0 <= args.gpu <= 7:
        raise ValueError("physical GPU must be in [0,7]")
    # Remap the selected physical GPU before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    # The CUDA >=10.2 deterministic cuBLAS contract must be set before torch
    # initializes; otherwise identical fixed-panel rows can differ at 1e-7 m.
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
    phases = [row["phase"] for row in rows]
    first_n = int(panel["metrics"]["first_n_frames"])
    threshold = float(panel["metrics"]["contact_confidence_threshold"])
    totals = {"overall": Accumulator(), "first_8_frames": Accumulator(),
              "contact_window": Accumulator(), "phases": {}}
    contact_frames = 0
    batches = 0
    deterministic_check = None
    for begin in range(0, len(positions), args.batch_size):
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("fixed-panel inference exceeded --max-seconds")
        batch_rows = rows[begin:begin + args.batch_size]
        batch_phases = [row["phase"] for row in batch_rows]
        batch = {key: value.cuda(non_blocking=True)
                 for key, value in collate([dataset[positions[i]] for i in range(begin, min(begin + args.batch_size, len(positions))) ]).items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            prediction = model(batch)
            if deterministic_check is None:
                repeat = model(batch)["trajectory"]
                deterministic_check = bool(torch.equal(prediction["trajectory"], repeat))
        hand_error = (prediction["trajectory"].float() - batch["trajectory_target"].float()).norm(dim=-1) * 1000.
        hand_valid = batch["trajectory_valid"].bool() & batch["start_hand_valid"].bool()[:, None]
        contact_frame = (batch["interaction_valid"].bool() &
                         (batch["contact_target"].float() > threshold)).any(dim=(1, 3))
        first = torch.zeros_like(contact_frame)
        first[:, :first_n] = True
        masks = {
            "overall": hand_valid,
            "first_8_frames": hand_valid & first[:, :, None, None],
            "contact_window": hand_valid & contact_frame[:, :, None, None],
        }
        for name, mask in masks.items():
            totals[name].add(hand_error[mask].detach().cpu().numpy())
        for index, phase in enumerate(batch_phases):
            totals["phases"].setdefault(phase, Accumulator()).add(
                hand_error[index][hand_valid[index]].detach().cpu().numpy())
        contact_frames += int(contact_frame.sum())
        batches += 1
        print(json.dumps({"batch": batches, "rows_scored": begin + len(batch_rows),
                          "elapsed_s": time.monotonic() - started}), flush=True)
        del batch, prediction, hand_error, hand_valid, contact_frame
        torch.cuda.empty_cache()
    metrics = {"overall": totals["overall"].finish(),
               "first_8_frames": totals["first_8_frames"].finish(),
               "contact_window": totals["contact_window"].finish(),
               "phases": {name: acc.finish() for name, acc in sorted(totals["phases"].items())}}
    report = {
        "schema": "ref2dex.tau-fixed-panel-score.v1",
        "status": "COMPLETED",
        "experiment_id": panel["experiment_id"],
        "run_id": output.name,
        "task": panel["task"],
        "panel": {"config": str(args.panel.resolve()), "anchor": str(anchor_path),
                  "anchor_id": anchor["anchor_id"], "anchor_sha256": sha256(anchor_path),
                  "rows": len(rows), "seed": panel["seed"],
                  "flow_eval_seed": panel["flow_eval_seed"],
                  "event_condition_dropout": False},
        "preprocessing": panel["preprocessing"],
        "checkpoint": {"path": str(args.checkpoint.resolve()), "step": step,
                       "schema": config.get("schema"), "flow_steps": config.get("flow_steps"),
                       "config_flow_eval_seed": config.get("flow_eval_seed")},
        "metrics": metrics,
        "contact_window": {"threshold": threshold, "frame_count": contact_frames,
                            "definition": panel["metrics"]["contact_window"]},
        "deterministic_repeat_check": deterministic_check,
        "deterministic_algorithms": True,
        "environment": {"python": sys.version, "torch": torch.__version__,
                         "cuda": torch.version.cuda, "visible_physical_gpu": args.gpu,
                         "device_name": torch.cuda.get_device_name(0),
                         "git_commit": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "timing": {"elapsed_s": time.monotonic() - started, "batches": batches,
                   "batch_size": args.batch_size, "max_seconds": args.max_seconds},
        "scope": "fixed TRAIN anchor prediction quality; no validation/generalization or executor claim",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    overall = metrics["overall"]["mean_mm"]
    first = metrics["first_8_frames"]["mean_mm"]
    contact = metrics["contact_window"]["mean_mm"]
    summary = ("step=%d rows=%d overall_hand_mean_mm=%.6f first8_hand_mean_mm=%.6f "
               "contact_window_hand_mean_mm=%.6f elapsed_s=%.1f" %
               (step, len(rows), overall, first, contact, report["timing"]["elapsed_s"]))
    (output / "summary.txt").write_text(summary + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
