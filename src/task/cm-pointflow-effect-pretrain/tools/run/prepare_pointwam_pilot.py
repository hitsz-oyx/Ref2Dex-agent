"""Bounded EgoDex -> PointWAM scene/hand data engineering smoke.

Uses original camera annotations, offline CoTracker3, and explicitly declared
depth and palm-offset backends. Never admits official test data to training.
Run from the intended worktree; external inputs/environments remain read-only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import (ARKIT_TO_ZUP, candidate_clips, lift_tracks,
    load_episode, pack_clip, palm_keypoints, project_world, recover_metric_scale,
    resize_intrinsics, sha256)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def decode_clip(video_path: Path, ids: np.ndarray, expected_frames: int) -> tuple[np.ndarray, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("video cannot be opened")
        if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames
                or abs(cap.get(cv2.CAP_PROP_FPS) - 30) > 1e-3):
            raise ValueError("video/annotation frame count or 30Hz clock mismatch")
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
        for source_id in range(int(ids[0]), int(ids[-1]) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"video decode failed at {source_id}")
            if source_id in ids:
                frames.append(cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB))
        return np.stack(frames), wh
    finally:
        cap.release()


def depth_from_spatracker(rgb: np.ndarray, vendor_root: Path, checkpoint: Path) -> np.ndarray:
    """Engineering substitute; VGGT4Track is not the paper's VGGT-Omega."""
    import torch
    sys.path.insert(0, str(vendor_root))
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    resized = np.stack([cv2.resize(x, (518, 294)) for x in rgb])
    video = torch.from_numpy(resized).permute(0, 3, 1, 2).float()[None].cuda()/255
    model = VGGT4Track.from_pretrained(str(checkpoint), local_files_only=True).eval().cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(video)
    depth = prediction["points_map"][..., 2].float().cpu().numpy().reshape(len(rgb), 294, 518)
    confidence = prediction["unc_metric"].float().cpu().numpy().reshape(depth.shape)
    depth[~np.isfinite(depth) | (depth <= 0) | (confidence < .5)] = 0
    del model, video, prediction
    torch.cuda.empty_cache()
    return np.stack([cv2.resize(x, (640, 360)) for x in depth])


def track_clip(rgb: np.ndarray, vendor: Path, checkpoint: Path, query_batch: int,
               deadline: float, progress) -> tuple[np.ndarray, np.ndarray]:
    import torch
    sys.path.insert(0, str(vendor))
    from cotracker.predictor import CoTrackerPredictor
    # Every second pixel of the first frame, no object or future-motion selection.
    yy, xx = np.mgrid[0:360:2, 0:640:2]
    queries = np.stack((np.zeros(xx.size), xx.ravel(), yy.ravel()), -1).astype(np.float32)
    video = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()[None].cuda()
    predictor = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).eval().cuda()
    tracks, visibility = [], []
    started = time.monotonic()
    for lo in range(0, len(queries), query_batch):
        if time.monotonic() > deadline:
            raise TimeoutError("tracking wall-time budget")
        q = torch.from_numpy(queries[lo:lo+query_batch])[None].cuda()
        with torch.inference_mode():
            xy, visible = predictor(video, queries=q, backward_tracking=False)
        tracks.append(xy[0].float().cpu().numpy())
        visibility.append(visible[0].cpu().numpy().astype(bool))
        done = min(lo+query_batch, len(queries))
        elapsed = time.monotonic()-started
        progress(dict(stage="TRACKING", query_done=done, query_total=len(queries),
                      tracking_elapsed_s=elapsed, tracking_eta_s=elapsed*(len(queries)-done)/done))
    del predictor, video, q, xy, visible
    torch.cuda.empty_cache()
    return np.concatenate(tracks, 1), np.concatenate(visibility, 1)


def visualize(destination: Path, rgb: np.ndarray, raw_hand: np.ndarray,
              hand_valid: np.ndarray, intrinsic: np.ndarray, camera: np.ndarray,
              pack: dict, depth: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    uv, _ = project_world(raw_hand, intrinsic, camera)
    tiles = []
    for t in (0, 5, 10):
        image = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR)
        for side in range(2):
            for keypoint in range(10):
                x, y = uv[t, side, keypoint]
                if hand_valid[t, side, keypoint] and np.isfinite([x, y]).all() and 0 <= x < 640 and 0 <= y < 360:
                    color = (0, 240, 0) if keypoint < 6 else (255, 130, 0)
                    cv2.circle(image, (int(round(x)), int(round(y))), 4, color, -1)
                    cv2.putText(image, str(keypoint), (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
        cv2.putText(image, f"source observation {t}/10", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        heatmap = cv2.applyColorMap(np.clip(depth[t]/2*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        tiles.append(np.concatenate((image, heatmap), axis=0))
    cv2.imwrite(str(destination / "projection_depth.jpg"), np.concatenate(tiles, axis=1))
    n = len(pack["track_ids"])
    show = np.linspace(0, n-1, min(n, 1200)).astype(int)
    xyz = pack["scene_xyz_m"][:, show]
    valid = pack["scene_valid"][:, show]
    colors = [f"rgb({r},{g},{b})" for r, g, b in pack["scene_rgb"][show]]
    hand = pack["hand_xyz_m"].reshape(31, 20, 3)
    hmask = pack["hand_valid"].reshape(31, 20)

    def traces(t):
        s = xyz[t].copy(); s[~valid[t]] = np.nan
        h = hand[t].copy(); h[~hmask[t]] = np.nan
        return [go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="markers",
                            marker=dict(size=2, color=colors), name="scene"),
                go.Scatter3d(x=h[:, 0], y=h[:, 1], z=h[:, 2], mode="markers+text",
                            marker=dict(size=6, color=["#13bd48"]*6+["#388aff"]*4+["#ed8733"]*6+["#be5af2"]*4),
                            text=[f"{side}{k}" for side in ("R", "L") for k in range(10)], name="hands")]

    fig = go.Figure(data=traces(0), frames=[go.Frame(data=traces(t), name=str(t)) for t in range(31)])
    lo = np.nanpercentile(xyz[0], 2, axis=0); hi = np.nanpercentile(xyz[0], 98, axis=0)
    fig.update_layout(title="EgoDex offline teacher: scene + semantic hands (meters, z-up)",
        scene=dict(xaxis=dict(title="x (m)", range=[float(lo[0]-.1), float(hi[0]+.1)]),
                   yaxis=dict(title="y (m)", range=[float(lo[1]-.1), float(hi[1]+.1)]),
                   zaxis=dict(title="z (m)", range=[float(lo[2]-.1), float(hi[2]+.1)]), aspectmode="data"),
        sliders=[dict(steps=[dict(method="animate", args=[[str(t)], {"mode":"immediate", "frame":{"duration":0,"redraw":True}}],
                                 label=f"{t/30:.2f}s") for t in range(31)])],
        updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, {"frame":{"duration":70,"redraw":True},"fromcurrent":True}])])])
    fig.write_html(destination / "scene_hand.html", include_plotlyjs=True, auto_play=False)
    still = plt.figure(figsize=(10, 7)); ax = still.add_subplot(111, projection="3d")
    good = valid[0]
    ax.scatter(*xyz[0, good].T, c=pack["scene_rgb"][show][good]/255, s=2, alpha=.6)
    for point in range(20):
        path = hand[:, point].copy();path[~hmask[:, point]] = np.nan
        ax.plot(*path.T, color="green" if point < 10 else "orange", lw=1)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Scene at t0 and 1-second hand trajectories")
    ax.set_box_aspect(np.maximum(hi-lo, .1));still.tight_layout()
    still.savefig(destination / "scene_hand.png", dpi=160);plt.close(still)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--palm-offsets", type=Path, required=True)
    parser.add_argument("--depth-vendor", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--tracker-vendor", type=Path, required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--query-batch", type=int, default=4096)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.clips <= 2 or not 256 <= args.query_batch <= 4096 or not 1 <= args.seconds <= 3600:
        parser.error("pilot bound: 1..2 clips, 256..4096 queries/batch, <=3600 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); deadline = start+args.seconds
    report = dict(status="RUNNING", run_id=args.output.name, engineering_only=True,
                  training_allowed=False, split="official_test", device="cuda:0",
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), pid=os.getpid(),
                  depth_backend="SpaTrackerV2-VGGT4Track (substitute; not VGGT-Omega)",
                  exact_paper_backend=False, strictly_causal_observation=False,
                  observation_provenance="offline full-clip teacher reconstruction",
                  camera_backend="native EgoDex camera-to-ARKit-origin",
                  query_stride_px=2, query_batch=args.query_batch, support_grid_size=6,
                  query_batch_note="query partitioning is declared; not bitwise author preprocessing",
                  result=None, clips=[])

    def progress(update):
        report.update(update, elapsed_s=time.monotonic()-start)
        write_json(args.output / "progress.json", report)
        print(json.dumps(update), flush=True)

    try:
        import torch
        torch.set_num_threads(2); cv2.setNumThreads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable; do not silently fall back to CPU")
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["branch"] = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        paths = [args.hdf5, args.video, args.palm_offsets, args.tracker_checkpoint,
                 Path(__file__), TASK / "src/oakink_wm/pointwam_data.py",
                 args.depth_checkpoint / "model.safetensors", args.depth_checkpoint / "config.json"]
        report["input_sha256"] = {str(p.resolve()): sha256(p) for p in paths}
        report["tracker_git_commit"] = subprocess.check_output(["git", "-C", str(args.tracker_vendor), "rev-parse", "HEAD"], text=True).strip()
        episode = load_episode(args.hdf5)
        offsets = json.loads(args.palm_offsets.read_text())
        report["palm_offsets_provenance"] = offsets
        selected, windows = candidate_clips(episode["joints"], episode["joint_valid"], args.clips)
        report["window_audit"] = windows
        if not selected:
            raise ValueError("no clip passes the predeclared 2cm wrist-motion filter")
        for index, ids in enumerate(selected):
            if time.monotonic() > deadline:
                raise TimeoutError("pilot deadline")
            destination = args.output / f"clip{index:02d}-f{ids[0]:04d}";destination.mkdir()
            rgb, original_wh = decode_clip(args.video, ids, len(episode["joints"]))
            intrinsic = resize_intrinsics(episode["intrinsics"], original_wh, (640, 360))
            camera = episode["camera_to_origin"][ids]
            joints = episode["joints"][ids]; jvalid = episode["joint_valid"][ids]
            hand, hvalid = palm_keypoints(joints, jvalid, np.asarray(offsets["offsets_m"]))
            progress(dict(stage="DEPTH", current_clip=index, source_start_frame=int(ids[0])))
            depth = depth_from_spatracker(rgb, args.depth_vendor, args.depth_checkpoint)
            calibration = recover_metric_scale(depth, joints, jvalid, intrinsic, camera)
            depth *= calibration["scale"]
            progress(dict(stage="TRACKING_START", depth_calibration=calibration,
                          gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
            uv, visibility = track_clip(rgb, args.tracker_vendor, args.tracker_checkpoint,
                                       args.query_batch, deadline, progress)
            world, svalid = lift_tracks(uv, visibility, depth, intrinsic, camera)
            y, x = np.rint(uv[0, :, 1]).astype(int), np.rint(uv[0, :, 0]).astype(int)
            color = rgb[0, np.clip(y, 0, 359), np.clip(x, 0, 639)]
            pack = pack_clip(world, svalid, hand, hvalid, color, ids, episode["language"])
            # Independent camera round-trip uses original tracks and native transforms.
            reprojected, _ = project_world(world[:, None], intrinsic, camera)
            pixel_errors = np.linalg.norm(reprojected[:, 0]-uv, axis=-1)[svalid]
            gates = dict(positions_31_targets_30=(pack["scene_xyz_m"].shape[0] == 31 and pack["scene_displacement_m"].shape[0] == 30),
                         initial_scene_points_ge256=len(pack["track_ids"]) >= 256,
                         one_complete_semantic_hand=bool(pack["hand_valid"].all((0, 2)).any()),
                         scale_relative_iqr_le035=calibration["relative_iqr"] <= .35,
                         camera_roundtrip_le001px=bool(len(pixel_errors) and pixel_errors.max() <= .01),
                         finite_valid_scene=bool(np.isfinite(pack["scene_xyz_m"][pack["scene_valid"]]).all()),
                         finite_valid_hand=bool(np.isfinite(pack["hand_xyz_m"][pack["hand_valid"]]).all()),
                         strictly_increasing_clock=bool(np.all(np.diff(pack["timestamps_s"]) > 0)))
            np.savez_compressed(destination / "pointwam_clip.npz", **pack)
            np.savez_compressed(destination / "teacher_frontend.npz", source_frame_ids=ids,
                                depth_m=depth, tracks_uv=uv, visibility=visibility,
                                intrinsic=intrinsic, camera_to_origin=camera,
                                raw_hand_xyz=hand, raw_hand_valid=hvalid)
            progress(dict(stage="VISUALIZING", current_clip=index))
            visualize(destination, rgb, hand, hvalid, intrinsic, camera, pack, depth)
            record = dict(path=str(destination.resolve()), source_frames=ids.tolist(),
                          scene_points=len(pack["track_ids"]),
                          scene_valid_fraction=float(pack["scene_valid"].mean()),
                          all_future_valid_tracks=int(pack["scene_valid"].all(0).sum()),
                          hand_valid_fraction=float(pack["hand_valid"].mean()),
                          calibration=calibration, gates=gates, engineering_gates_pass=all(gates.values()),
                          roundtrip_max_px=float(pixel_errors.max()),
                          pack_sha256=sha256(destination / "pointwam_clip.npz"))
            write_json(destination / "audit.json", record);report["clips"].append(record)
            progress(dict(stage="CLIP_COMPLETE", current_clip=index))
        report.update(status="COMPLETED", stage="COMPLETED",
                      engineering_gates_pass=all(c["engineering_gates_pass"] for c in report["clips"]),
                      gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      total_output_bytes=sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()))
    except Exception as exc:
        report.update(status="FAILED", stage="FAILED", error=repr(exc))
        raise
    finally:
        report["elapsed_s"] = time.monotonic()-start
        write_json(args.output / "manifest.json", report)
        write_json(args.output / "progress.json", report)
        print(json.dumps({k:report[k] for k in ("status", "stage", "elapsed_s")}), flush=True)


if __name__ == "__main__":
    main()
