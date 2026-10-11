"""Audit the frozen C1 evaluator's object-effect branch on held panels.

This is a read-only wiring/data-distribution audit.  It reuses the exact
episode-split panel and C1 checkpoint from the history-panel Probe and never
fits a model or uses labels to construct the ablations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys

sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.old_utility import panel_metrics
from consequence_evaluator.trajectory_utility import TrajectoryUtility


SCHEMA = "ref2dex.history-candidate-utility-fit.v1"
PANEL_SCHEMA = "ref2dex.history-candidate-panel.v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_panel(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != PANEL_SCHEMA or manifest.get("status") != "COMPLETED":
        raise ValueError("completed history candidate panel required")
    with np.load(root / "windows.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    required = {"history", "pw_hand_future", "effect_gt", "label", "panel", "candidate", "split"}
    if not required.issubset(data):
        raise ValueError("candidate panel fields missing")
    n = len(data["label"])
    if (data["history"].shape != (n, 1442)
            or data["pw_hand_future"].shape != (n, 24, 11, 3)
            or data["effect_gt"].shape != (n, 24, 12)
            or data["label"].shape != (n,)):
        raise ValueError("candidate panel shape mismatch")
    for key in ("history", "pw_hand_future", "effect_gt", "label"):
        if not np.isfinite(data[key]).all():
            raise ValueError("nonfinite candidate panel field: " + key)
    panels = np.unique(data["panel"])
    rows = np.asarray([np.flatnonzero(data["panel"] == panel) for panel in panels])
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError("complete seven-candidate panels required")
    if not np.all(data["candidate"][rows] == np.arange(7)):
        raise ValueError("candidate order is not canonical")
    if any(np.unique(data["split"][row]).size != 1 for row in rows):
        raise ValueError("panel crosses episode split")
    test = rows[data["split"][rows[:, 0]] == "test"]
    if not len(test):
        raise ValueError("test panels are empty")
    return root, manifest, data, rows, test


def encode(value, statistics):
    mean, scale = statistics
    return ((np.asarray(value) - mean) / scale).astype("float32")


def predict(model, history, trajectory, effect, use_effect=True):
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(history), torch.from_numpy(trajectory),
                     torch.from_numpy(effect), use_effect).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=416)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned audit output required")

    data_root, panel_manifest, data, rows, test_rows = load_panel(args.data)
    fit_root = args.fit.resolve()
    fit_manifest = json.loads((fit_root / "manifest.json").read_text())
    if fit_manifest.get("schema") != SCHEMA or fit_manifest.get("status") != "COMPLETED":
        raise ValueError("completed history utility fit required")
    checkpoint = fit_root / "C1.pt"
    expected_checkpoint = fit_manifest.get("checkpoint_sha256", {}).get("C1")
    if expected_checkpoint and sha(checkpoint) != expected_checkpoint:
        raise ValueError("C1 checkpoint hash drift")
    expected_data = fit_manifest.get("input_sha256", {}).get(str(data_root / "windows.npz"))
    if expected_data and sha(data_root / "windows.npz") != expected_data:
        raise ValueError("panel data hash drift")

    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (packet.get("schema") != SCHEMA or packet.get("arm") != "C1"
            or set(packet.get("statistics", {})) != {"history", "trajectory", "effect"}):
        raise ValueError("C1 checkpoint contract mismatch")
    model = TrajectoryUtility(**{key: packet["architecture"][key]
                                 for key in ("history_dim", "width", "layers")})
    model.load_state_dict(packet["model"], strict=True)

    flat = test_rows.reshape(-1)
    # Keep donor indices in the panel's global row space.  The fit's original
    # audit uses the same global rows before selecting the held test panels.
    all_history = encode(data["history"], packet["statistics"]["history"])
    all_trajectory = encode(data["pw_hand_future"].reshape(-1, 24, 33),
                            packet["statistics"]["trajectory"])
    all_effect = encode(data["effect_gt"], packet["statistics"]["effect"])
    history = all_history[flat]
    trajectory = all_trajectory[flat]
    effect = all_effect[flat]
    target = data["label"][flat].reshape(len(test_rows), 7)
    rng = np.random.default_rng(args.seed)
    donor = np.stack([np.roll(row, int(rng.integers(1, 7))) for row in test_rows]).reshape(-1)
    shuffled_trajectory = all_trajectory[donor]
    shuffled_effect = all_effect[donor]
    zero_effect = np.zeros_like(effect)
    predictions = {
        "nominal": predict(model, history, trajectory, effect, True),
        "effect_zero": predict(model, history, trajectory, zero_effect, True),
        "effect_shuffle": predict(model, history, trajectory, shuffled_effect, True),
        "tau_shuffle": predict(model, history, shuffled_trajectory, effect, True),
        "both_shuffle": predict(model, history, shuffled_trajectory, shuffled_effect, True),
        "effect_disabled": predict(model, history, trajectory, effect, False),
    }
    metrics = {name: panel_metrics(target, value.reshape(len(test_rows), 7))
               for name, value in predictions.items()}
    nominal = metrics["nominal"]["pairwise_accuracy"]
    effect_zero = metrics["effect_zero"]["pairwise_accuracy"]
    effect_shuffle = metrics["effect_shuffle"]["pairwise_accuracy"]
    tau_shuffle = metrics["tau_shuffle"]["pairwise_accuracy"]
    result = dict(
        status="UNPROMISING",
        screen_gate=False,
        metrics=metrics,
        deltas=dict(
            effect_zero_minus_nominal=float(effect_zero - nominal),
            effect_shuffle_minus_nominal=float(effect_shuffle - nominal),
            nominal_minus_tau_shuffle=float(nominal - tau_shuffle),
            effect_zero_equals_disabled=bool(
                np.array_equal(predictions["effect_zero"], predictions["effect_disabled"])),
        ),
        gates=dict(
            nominal_pair_at_least_070=bool(nominal >= .70),
            effect_zero_not_more_than_3pp_better=bool(effect_zero - nominal <= .03),
            effect_shuffle_not_more_than_3pp_better=bool(effect_shuffle - nominal <= .03),
            tau_shuffle_drop_at_least_3pp=bool(nominal - tau_shuffle >= .03),
        ),
        test_panels=int(len(test_rows)),
        strict_pairs=int(metrics["nominal"]["strict_pairs"]),
        donor=donor.tolist(),
        seed=int(args.seed),
        claim="Frozen C1 effect-branch wiring/distribution audit; no selector or native claim",
        provenance=dict(
            panel_manifest=str(data_root / "manifest.json"),
            panel_manifest_sha256=sha(data_root / "manifest.json"),
            panel_windows=str(data_root / "windows.npz"),
            panel_windows_sha256=sha(data_root / "windows.npz"),
            fit_manifest=str(fit_root / "manifest.json"),
            fit_manifest_sha256=sha(fit_root / "manifest.json"),
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            script=str(Path(__file__).resolve()),
            script_sha256=sha(Path(__file__).resolve()),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        ),
    )
    output.mkdir(parents=True)
    np.savez_compressed(output / "predictions.npz", target=target, donor=donor,
                        **{name: value.reshape(len(test_rows), 7)
                           for name, value in predictions.items()})
    write(output / "result.json", result)
    write(output / "manifest.json", dict(schema="ref2dex.history-utility-effect-audit.v1",
                                          status="COMPLETED", run_id=output.name,
                                          data=str(data_root), fit=str(fit_root),
                                          result_sha256=sha(output / "result.json"),
                                          **result["provenance"]))
    print(json.dumps({key: result[key] for key in ("status", "metrics", "deltas", "gates")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
