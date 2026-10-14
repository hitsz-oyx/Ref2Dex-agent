#!/usr/bin/env python3
"""Audit real Open-AoE inputs and decode hand-only H4/K24 fixtures, not a corpus."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from video_data_preparation.geometry import check_transform
from video_data_preparation.native_hand import NativeEpisode

# Matches the existing three-domain provider, not an inferred point layout.
MANO_BASE = [0, 13, 1, 4, 10, 7]
TIPS = [744, 320, 443, 554, 671]
OPENPOSE_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def probe_video(path):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,duration:packet=pts_time",
        "-show_packets", "-of", "json", str(path)], text=True, timeout=30))
    stream = result["streams"][0]
    times = np.sort([float(p["pts_time"]) for p in result["packets"] if "pts_time" in p])
    stream["packet_count"] = len(times)
    stream["max_30hz_step_error_s"] = float(np.max(np.abs(np.diff(times)-1/30)))
    stream["duplicate_pts"] = int(np.sum(np.diff(times) <= 0))
    return stream


def decode_hands(data, model_root, device):
    # Old licensed MANO pickles import chumpy; aliases affect only this process.
    for name, value in [("bool", bool), ("int", int), ("float", float),
                        ("complex", complex), ("object", object), ("str", str), ("unicode", str)]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import torch
    from smplx import MANO
    torch.set_num_threads(2)
    output, full_output = [], []
    start = time.monotonic()
    with torch.no_grad():
        for index, side in [(1, "right"), (0, "left")]:
            model = MANO(str(model_root), is_rhand=side == "right", use_pca=False,
                         flat_hand_mean=True).to(device)
            # AoE's official layer applies this fix to the affected left model.
            if side == "left":
                model.shapedirs[:, 0, :] *= -1
            selected, all_joints = [], []
            for lo in range(0, data["pred_valid"].shape[1], 128):
                if time.monotonic()-start > 60:
                    raise TimeoutError("bounded MANO decode exceeded 60 seconds")
                fields = {name: torch.as_tensor(data[key][index, lo:lo+128],
                                               dtype=torch.float32, device=device)
                          for name, key in [("betas", "pred_betas"), ("global_orient", "pred_rot"),
                                            ("hand_pose", "pred_hand_pose"), ("transl", "pred_trans")]}
                res = model(**fields)
                tips = res.vertices[:, TIPS]
                selected.append(torch.cat((res.joints[:, MANO_BASE], tips), 1).cpu().numpy())
                all_joints.append(torch.cat((res.joints[:, :16], tips), 1)[:, OPENPOSE_ORDER].cpu().numpy())
            output.append(np.concatenate(selected))
            full_output.append(np.concatenate(all_joints))
    return np.stack(output, 1), np.stack(full_output, 1), time.monotonic()-start


def projection(points, k):
    image = points @ k.T
    return image[..., :2] / image[..., 2:3]


def render_sheet(video, camera_hands, valid, k, ticks, path, title):
    capture = cv2.VideoCapture(str(video))
    tiles = []
    for tick in ticks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(tick))
        ok, frame = capture.read()
        if not ok:
            raise ValueError("failed to decode requested real video frame")
        for side, color in [(0, (255, 190, 60)), (1, (60, 180, 255))]:
            if not valid[tick, side, 0]:
                continue
            hand = camera_hands[tick, side]
            uv = projection(hand, k)
            keep = np.isfinite(uv).all(1) & (hand[:, 2] > 0)
            keep &= (uv[:, 0] >= 0) & (uv[:, 0] < frame.shape[1])
            keep &= (uv[:, 1] >= 0) & (uv[:, 1] < frame.shape[0])
            for finger in range(5):
                ids = [0] + list(range(1+4*finger, 5+4*finger))
                for x, y in zip(ids, ids[1:]):
                    if keep[x] and keep[y]:
                        cv2.line(frame, tuple(uv[x].astype(int)), tuple(uv[y].astype(int)), color, 3, cv2.LINE_AA)
            for i in np.flatnonzero(keep):
                cv2.circle(frame, tuple(uv[i].astype(int)), 5, color, -1, cv2.LINE_AA)
            if keep[0]:
                cv2.putText(frame, "R" if side == 0 else "L", tuple(uv[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        tile = cv2.resize(frame, (960, 540))
        cv2.putText(tile, "%s frame %d / %.2fs" % (title, tick, tick/30), (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        tiles.append(tile)
    capture.release()
    if not cv2.imwrite(str(path), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])):
        raise IOError("contact sheet write failed")


def audit(sample, dest, model_root, device):
    dest.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(sample / "ego_process/ego_hands_reconstruction/hands.npz", allow_pickle=False))
    camera_data = dict(np.load(sample / "ego_process/ego_hands_reconstruction/camera_traj.npz", allow_pickle=False))
    metadata = json.loads((sample / "ego_process/ego_undistorted_video/undistorted_video_info.json").read_text())
    annotations = json.loads((sample / "ego_annotation/ego_action_annotation.json").read_text())
    video = sample / "ego_process/ego_undistorted_video/raw_video_undistorted.mp4"
    stream = probe_video(video)
    raw_stream = probe_video(sample / "raw_video.mp4")
    frames = int(stream["nb_frames"])
    expected = dict(R_w2c=(frames, 3, 3), t_w2c=(frames, 3), R_c2w=(frames, 3, 3),
                    t_c2w=(frames, 3), pred_trans=(2, frames, 3), pred_rot=(2, frames, 3),
                    pred_trans_cam=(2, frames, 3), pred_rot_cam=(2, frames, 3),
                    pred_hand_pose=(2, frames, 45), pred_betas=(2, frames, 10), pred_valid=(2, frames))
    shape_ok = all(data[key].shape == shape for key, shape in expected.items())
    finite = all(np.isfinite(v).all() for v in data.values())
    c2w = check_transform(camera_data["cam_c2w"])
    if not shape_ok or not finite or c2w.shape != (frames, 4, 4):
        raise ValueError("input hand/camera array shape or finite check failed")
    if not np.isin(data["pred_valid"], [0, 1]).all():
        raise ValueError("unexpected per-hand validity encoding")
    pars = metadata["cameraParams"]
    k_json = np.array([[pars["fx_pixels"], 0, pars["cx_pixels"]],
                       [0, pars["fy_pixels"], pars["cy_pixels"]], [0, 0, 1.]])
    k = np.asarray(camera_data["intrinsic"])
    pose_error = max(float(np.max(abs(c2w[:, :3, :3]-data["R_c2w"]))),
                     float(np.max(abs(c2w[:, :3, 3]-data["t_c2w"]))))
    clock_ok = (stream["packet_count"] == frames == int(raw_stream["nb_frames"])
                and Fraction(stream["avg_frame_rate"]) == 30
                and stream["max_30hz_step_error_s"] < 1e-5
                and stream["duplicate_pts"] == 0)
    if not clock_ok or pars["resolution"] != "%dx%d" % (stream["width"], stream["height"]):
        raise ValueError("actual video clock or intrinsic resolution mismatch")
    if pose_error > 1e-4 or np.max(abs(k-k_json)) > 1e-4:
        raise ValueError("duplicate published camera inputs disagree")
    hand, full, decode_s = decode_hands(data, model_root, device)
    side_valid = (data["pred_valid"][[1, 0]] > .5).T
    valid = np.broadcast_to(side_valid[..., None], hand.shape[:-1]).copy()
    hand[~valid] = 0.
    camera_full = np.einsum("tij,thkj->thki", data["R_w2c"], full) + data["t_w2c"][:, None, None]
    camera_full[~side_valid] = 0.
    camera_hand = np.einsum("tij,thkj->thki", data["R_w2c"], hand) + data["t_w2c"][:, None, None]
    uv = projection(camera_hand, k)
    in_view = valid & (camera_hand[..., 2] > 0) & np.isfinite(uv).all(-1)
    in_view &= (uv[..., 0] >= 0) & (uv[..., 0] < stream["width"])
    in_view &= (uv[..., 1] >= 0) & (uv[..., 1] < stream["height"])
    anchors = np.arange(3, frames-24, 8)
    h4 = np.stack([side_valid[t-3:t+1].all(0) for t in anchors])
    stable_future = np.stack([side_valid[t-3:t+25].all(0) for t in anchors])
    roots = data["pred_trans"]
    computed_cam = np.einsum("tij,htj->hti", data["R_w2c"], roots) + data["t_w2c"][None]
    native = NativeEpisode(hand, valid, c2w, k, sample.name, "Open-AoE segment annotations")
    tick = int(anchors[np.flatnonzero(h4.any(1))[0]])
    fixture = native.window(tick, np.linalg.inv(c2w[tick]))
    np.savez_compressed(dest / "hand-only-window.npz", **fixture, tick=tick, training_allowed=False)
    np.savez_compressed(dest / "decoded-hands.npz", hand_world=hand, hand_valid=valid,
                        camera_c2w=c2w, intrinsic=k, training_allowed=False)
    ticks = np.minimum(np.array([60, 270, 570, 2100]), frames-1).astype(int)
    render_sheet(video, camera_full, valid, k, ticks, dest / "hand-overlay.jpg", "published K")
    k_reference = np.array([[float(data["focal"]), 0, stream["width"]/2],
                            [0, float(data["focal"]), stream["height"]/2], [0, 0, 1.]])
    render_sheet(video, camera_full, valid, k_reference, ticks,
                 dest / "hand-overlay-reference-K.jpg", "official viewer K")
    verbs = sorted({row["verb"] for segment in annotations for row in segment["atomic_action"]})
    result = dict(sample=sample.name, video=stream, raw_video=raw_stream, clock_pass=clock_ok,
                  camera_pose_disagreement_max=pose_error,
                  camera_inverse_rotation_error_max=float(np.max(abs(data["R_w2c"]@data["R_c2w"]-np.eye(3)))),
                  hand_root_camera_transform_error_max_m=float(np.max(abs(computed_cam-data["pred_trans_cam"]))),
                  intrinsic=k.tolist(), reference_viewer_intrinsic=k_reference.tolist(),
                  mano_decode_s=decode_s, device=device, hand_order=["right", "left"],
                  valid_hand_fraction=side_valid.mean(0).tolist(),
                  projected_in_frame_fraction=float(in_view.sum()/valid.sum()),
                  hand_span_median_m=float(np.median(np.linalg.norm(hand[:,:,6:]-hand[:,:,0,None],axis=-1)[side_valid])),
                  attempted_hand_windows=len(anchors), h4_one_hand=int(h4.any(1).sum()),
                  h4_both_hands=int(h4.all(1).sum()),
                  complete_28_frames_one_hand=int(stable_future.any(1).sum()),
                  complete_28_frames_both_hands=int(stable_future.all(1).sum()),
                  fixture_shapes={key:list(value.shape) for key,value in fixture.items()},
                  annotation_segments=len(annotations), verbs=verbs,
                  annotation_last_frame=max(int(s["end_frame"]) for s in annotations),
                  training_allowed=False, scope="hand/camera engineering admission only",
                  remaining=["current object identity/mask", "current and label metric depth",
                             "fixed query object tracks and endpoint", "visual hand review",
                             "offline reconstructed observation provenance", "gravity/world-axis audit"],
                  thumb_note="MANO joint13 matches existing native provider; anatomical equivalence to EgoDex ThumbKnuckle needs review")
    (dest / "audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--mano-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    a = p.parse_args()
    manifest = json.loads((a.subset / "download-manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise ValueError("complete downloaded inputs required")
    start = time.monotonic()
    results = []
    for sample in manifest["samples"]:
        print("Audit", sample, flush=True)
        results.append(audit(a.subset/"raw"/sample, a.subset/"audit"/sample, a.mano_root, a.device))
    summary = dict(status="COMPLETE", samples=results, elapsed_s=time.monotonic()-start,
                   training_allowed=False,
                   code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    (a.subset / "audit-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
