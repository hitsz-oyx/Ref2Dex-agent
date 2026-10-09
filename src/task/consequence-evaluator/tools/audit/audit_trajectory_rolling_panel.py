"""Audit frozen C0/C1 ranking on recovered rolling same-state panels.

The ref13 recovery already contains four later seven-way forks.  This tool
reconstructs their measured 11-point hand/object windows without simulation,
then evaluates the frozen trajectory evaluator.  The result is a support/OOD
screen only: all forks come from one reconstructed actor/motion campaign and
no deployable policy is trained here.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.data import interaction_future, object_effect, sha
from consequence_evaluator.contracts import HAND_LINKS
from consequence_evaluator.old_utility import panel_metrics, teacher
from consequence_evaluator.trajectory_utility import TrajectoryUtility


# Keep one rolling selector history in the held bank.  The old-Y cohorts are
# valid engineering packets but are a separate route and must not be pooled
# with the new-Y panels for this screen.
FAMILIES = ("new-o8", "new-o16")
FAMILY_PROVENANCE = {
    "initial": {"parent_actual_run": "reanchor-baseline", "offset": 0, "query": 71},
    "new-o8": {"parent_actual_run": "new-o0-actual", "offset": 8, "query": 79},
    "new-o16": {"parent_actual_run": "new-o8-actual", "offset": 16, "query": 87},
}
HORIZON = 24


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_panel_rows(data):
    panels = np.unique(data["panel"][data["panel"] >= 0])
    rows = np.asarray([
        sorted(np.flatnonzero(data["panel"] == panel),
               key=lambda row: data["candidate"][row])
        for panel in panels])
    if rows.shape != (25, 7):
        raise ValueError("old utility input must contain its frozen 25x7 panel")
    if not all(np.array_equal(data["candidate"][row], np.arange(7))
               for row in rows):
        raise ValueError("old utility candidate order drift")
    if not np.all(data["history"][rows] == data["history"][rows[:, :1]]):
        raise ValueError("old utility panel H identity drift")
    return rows


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def build_rolling_panels(recovery, hashes):
    records = []
    panel_offset = 25
    for family_index, family in enumerate(FAMILIES):
        packets = []
        traces = []
        for candidate in range(7):
            folder = recovery / (family + "-k" + str(candidate))
            packet_path = folder / "panel.pt"
            trace_path = folder / "trace.pt"
            if not packet_path.is_file() or not trace_path.is_file():
                raise FileNotFoundError(folder)
            hashes[str(packet_path.resolve())] = sha(packet_path)
            hashes[str(trace_path.resolve())] = sha(trace_path)
            packet = torch.load(packet_path, map_location="cpu", weights_only=False)
            trace = torch.load(trace_path, map_location="cpu", weights_only=False)
            if int(packet["candidate"]) != candidate:
                raise ValueError("candidate identity drift: " + str(folder))
            if packet["post_window"] != 32:
                raise ValueError("rolling panel post-window drift: " + str(folder))
            packets.append(packet)
            traces.append(trace)

        triggers = np.stack([as_numpy(packet["triggers"]) for packet in packets])
        valid = (triggers >= 0).all(0) & (triggers == triggers[:1]).all(0)
        rows = np.flatnonzero(valid)
        if len(rows) != 25:
            raise ValueError("expected 25 complete synchronized anchors in " + family)
        if not all(packet["valid_steps"][rows].all() for packet in packets):
            raise ValueError("selected rolling anchors are incomplete in " + family)
        tolerance = torch.tensor([1e-4] * 5 + [1e-5, 0.])
        if any(packet["clipped_steps"][rows].any() or
               (packet["full_world_prefix_errors"][rows] > tolerance).any()
               for packet in packets):
            raise ValueError("rolling prefix/clipping contract drift in " + family)
        for candidate in range(1, 7):
            if not torch.equal(packets[candidate]["before"][rows], packets[0]["before"][rows]):
                raise ValueError("before-state mismatch in " + family)
            if not torch.equal(packets[candidate]["actor_obs"][rows], packets[0]["actor_obs"][rows]):
                raise ValueError("actor observation mismatch in " + family)

        for env in rows:
            query = int(triggers[0, env])
            if query < 3 or query + HORIZON >= traces[0]["progress_geometry"]["object_pose"].shape[0]:
                raise ValueError("query lacks four past/24 future geometry in " + family)
            for candidate in range(1, 7):
                g0 = traces[0]["progress_geometry"]
                gc = traces[candidate]["progress_geometry"]
                if not torch.equal(gc["object_pose"][:query + 1, env],
                                    g0["object_pose"][:query + 1, env]):
                    raise ValueError("object prefix mismatch in " + family)
                if not torch.equal(gc["hand_keypoints"][:query + 1, env],
                                    g0["hand_keypoints"][:query + 1, env]):
                    raise ValueError("hand prefix mismatch in " + family)

            state = as_numpy(traces[0]["decision_states"][query]["actor_obs"][env]).astype("float32")
            if state.shape != (1442,):
                raise ValueError("decision H shape mismatch in " + family)

            for candidate, packet in enumerate(packets):
                # The synchronized prefix is shared, but post-query geometry
                # is the candidate treatment and must come from its own trace.
                trace = traces[candidate]
                geometry = trace["progress_geometry"]
                if tuple(geometry["hand_links"]) != HAND_LINKS:
                    raise ValueError("rolling hand-link order drift in " + family)
                poses = as_numpy(geometry["object_pose"][:, env]).astype("float32")
                hands = as_numpy(geometry["hand_keypoints"][:, env]).astype("float32")
                pose_window = poses[query:query + HORIZON + 1]
                hand_window = hands[query:query + HORIZON + 1]
                if pose_window.shape != (25, 4, 4) or hand_window.shape != (25, 11, 3):
                    raise ValueError("rolling geometry shape mismatch in " + family)
                current = pose_window[0]
                inverse = np.linalg.inv(current)
                hand_history_world = hands[query - 3:query + 1]
                hand_future_world = hands[query + 1:query + HORIZON + 1]
                object_history = inverse @ poses[query - 3:query + 1]
                hand_history = (hand_history_world - current[:3, 3]) @ current[:3, :3]
                hand_future = (hand_future_world - current[:3, 3]) @ current[:3, :3]
                future = np.concatenate((
                    object_effect(pose_window)[:, :3].reshape(HORIZON, 12),
                    interaction_future(pose_window, hand_window).reshape(HORIZON, 33)), axis=-1)
                before = as_numpy(packet["before"][env])
                height = as_numpy(packet["height"][env])
                pair = as_numpy(packet["pair"][env]).astype(bool)
                label = teacher(before[2], before[71] > .5, height, pair,
                                float(as_numpy(packet["rest_height"][env])))
                action = np.zeros((HORIZON, 18), dtype="float32")
                action[:8] = as_numpy(packet["delta"])[candidate]
                records.append(dict(
                    history=state, action=action, future=future,
                    pw_object_history=object_history.astype("float32"),
                    pw_hand_history=hand_history.astype("float32"),
                    pw_hand_future=hand_future.astype("float32"),
                    label=float(label[1]), panel=panel_offset + family_index * 25 + int(np.flatnonzero(rows == env)[0]),
                    candidate=candidate, family=family, source_env=int(env), query=query,
                    y=label[0].astype("float32")))
    return records


def load_panel_pack(data_root, recovery):
    hashes = {}
    for path in (data_root / "manifest.json", data_root / "windows.npz"):
        hashes[str(path.resolve())] = sha(path)
    with np.load(data_root / "windows.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    rows = load_panel_rows(data)
    flat = rows.ravel()
    keys = ("history", "action", "future", "pw_object_history", "pw_hand_history",
            "pw_hand_future", "label")
    records = []
    for index, row in enumerate(flat):
        records.append(dict(
            history=data["history"][row].astype("float32"),
            action=data["action"][row].astype("float32"),
            future=data["future"][row].astype("float32"),
            pw_object_history=data["pw_object_history"][row].astype("float32"),
            pw_hand_history=data["pw_hand_history"][row].astype("float32"),
            pw_hand_future=data["pw_hand_future"][row].astype("float32"),
            label=float(data["label"][row]), panel=index // 7, candidate=index % 7,
            family="initial", source_env=-1, query=int(data["tick"][row])))
    records.extend(build_rolling_panels(recovery, hashes))
    arrays = {}
    for key in keys:
        arrays[key] = np.stack([record[key] for record in records])
    metadata = {key: np.asarray([record[key] for record in records])
                for key in ("panel", "candidate", "family", "source_env", "query")}
    arrays.update(metadata)
    panel_ids = np.unique(arrays["panel"])
    panel_rows = np.asarray([
        np.flatnonzero(arrays["panel"] == panel)
        for panel in panel_ids])
    if panel_rows.shape != (75, 7):
        raise ValueError("expanded panel count drift")
    for panel in panel_rows:
        if not np.array_equal(arrays["candidate"][panel], np.arange(7)):
            raise ValueError("expanded candidate order drift")
        if not np.all(arrays["history"][panel] == arrays["history"][panel[:1]]):
            raise ValueError("expanded H identity drift")
    return arrays, panel_rows, hashes


def score_checkpoint(path, arrays, panel_rows, device, use_effect):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    model = TrajectoryUtility(**{key: packet["architecture"][key]
                                 for key in ("history_dim", "width", "layers")}).to(device).eval()
    model.load_state_dict(packet["model"], strict=True)
    stats = packet["statistics"]
    history = arrays["history"]
    tau = arrays["pw_hand_future"].reshape(-1, HORIZON, 33)
    effect = arrays["future"][:, :, :12]
    values = {}
    for key, raw in (("history", history), ("trajectory", tau), ("effect", effect)):
        mean, scale = stats[key]
        values[key] = torch.as_tensor((raw - np.asarray(mean)) / np.asarray(scale), device=device).float()
    scores = []
    with torch.inference_mode():
        for begin in range(0, len(history), 256):
            end = min(begin + 256, len(history))
            scores.append(model(values["history"][begin:end], values["trajectory"][begin:end],
                                values["effect"][begin:end], use_effect).cpu().numpy())
    return np.concatenate(scores).reshape(len(panel_rows), 7)


def trajectory_spread(arrays, panel_rows):
    result = {}
    families = arrays["family"][panel_rows[:, 0]]
    for family in np.unique(families):
        selected = panel_rows[families == family]
        tau = arrays["pw_hand_future"][selected].reshape(len(selected), 7, HORIZON, 11, 3)
        delta = tau - tau[:, :1]
        rms = np.sqrt(np.mean(np.square(delta), axis=(2, 3, 4)))
        result[str(family)] = dict(mean_rms_m=float(rms.mean()), max_rms_m=float(rms.max()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--c0", type=Path, required=True)
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.parent.resolve() != (ROOT / "outputs/consequence-evaluator").resolve():
        raise ValueError("fresh task-owned rolling-panel audit output required")
    if subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
            "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied")
    arrays, panel_rows, hashes = load_panel_pack(args.data.resolve(), args.recovery.resolve())
    for path in (args.c0.resolve(), args.c1.resolve(), Path(__file__).resolve(),
                 TASK / "src/consequence_evaluator/old_utility.py",
                 TASK / "src/consequence_evaluator/trajectory_utility.py",
                 TASK / "src/consequence_evaluator/contracts.py"):
        hashes[str(path)] = sha(path)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    labels = arrays["label"].reshape(len(panel_rows), 7)
    c0 = score_checkpoint(args.c0, arrays, panel_rows, device, False)
    c1 = score_checkpoint(args.c1, arrays, panel_rows, device, True)
    rng = np.random.default_rng(411)
    donor = np.stack([np.roll(row, int(rng.integers(1, 7))) for row in panel_rows]).ravel()
    shuffled_arrays = {key: value.copy() for key, value in arrays.items()}
    shuffled_arrays["pw_hand_future"] = arrays["pw_hand_future"][donor]
    c1_shuffle = score_checkpoint(args.c1, shuffled_arrays, panel_rows, device, True)
    metrics = {"C0": panel_metrics(labels, c0), "C1": panel_metrics(labels, c1)}
    shuffle_metrics = {"C1": panel_metrics(labels, c1_shuffle)}
    informative = np.abs(labels - labels[:, :1]).max(1) > .02
    family_counts = {str(family): int((arrays["family"] == family).sum() // 7)
                     for family in np.unique(arrays["family"])}
    result = dict(
        status="COMPLETED",
        schema="ref2dex.trajectory-rolling-panel-audit.v1",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, input_sha256=hashes,
        panel_count=int(len(panel_rows)), rows=int(len(arrays["label"])),
        family_panel_counts=family_counts,
        family_provenance=FAMILY_PROVENANCE,
        trajectory_candidate_spread=trajectory_spread(arrays, panel_rows),
        informative_anchors=int(informative.sum()),
        strict_pairs=int((np.abs(labels[:, :, None] - labels[:, None, :]) > .02).sum() // 2),
        metrics=metrics, shuffle_metrics=shuffle_metrics,
        C1_gain_vs_C0=float(metrics["C1"]["pairwise_accuracy"] - metrics["C0"]["pairwise_accuracy"]),
        C1_tau_shuffle_drop=float(metrics["C1"]["pairwise_accuracy"] - shuffle_metrics["C1"]["pairwise_accuracy"]),
        screen_gate=bool(metrics["C1"]["pairwise_accuracy"] >= .70 and
                         metrics["C1"]["pairwise_accuracy"] - metrics["C0"]["pairwise_accuracy"] >= .03 and
                         metrics["C1"]["pairwise_accuracy"] - shuffle_metrics["C1"]["pairwise_accuracy"] >= .03 and
                         metrics["C1"]["mean_regret"] <= metrics["C0"]["mean_regret"]),
        scope=("Recovered single actor/motion campaign; expanded rolling panels are held "
               "offline evidence, not independent Validation or control results."))
    output.mkdir(parents=True)
    np.savez_compressed(output / "panel-predictions.npz", labels=labels, panel_rows=panel_rows,
                        C0=c0, C1=c1, C1_tau_shuffle=c1_shuffle,
                        tau=arrays["pw_hand_future"],
                        family=arrays["family"], source_env=arrays["source_env"], query=arrays["query"])
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({key: result[key] for key in (
        "panel_count", "family_panel_counts", "informative_anchors", "strict_pairs",
        "metrics", "shuffle_metrics", "C1_gain_vs_C0", "C1_tau_shuffle_drop", "screen_gate")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
