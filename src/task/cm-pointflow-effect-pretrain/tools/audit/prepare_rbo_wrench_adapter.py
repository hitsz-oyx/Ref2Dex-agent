#!/usr/bin/env python3
"""Build a bounded, timestamp-aligned RBO RGB-D/FT adapter.

The adapter intentionally stops at a geometric object-frame wrench.  The RBO
archive does not contain payload mass/COM/inertia, so it must not manufacture a
payload-compensated object-COM net wrench.  All source files remain read-only;
only a small manifest and compressed audit arrays are written to ``--output``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


RGB_RE = re.compile(r"-(?P<ts>[0-9]+(?:\.[0-9]+)?)\.png$")
DEPTH_RE = re.compile(r"-(?P<ts>[0-9]+(?:\.[0-9]+)?)\.txt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float(value: str) -> float:
    return float(value.strip())


def nearest(values: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest row indices and absolute time errors for sorted values."""
    positions = np.searchsorted(values, query, side="left")
    left = np.clip(positions - 1, 0, len(values) - 1)
    right = np.clip(positions, 0, len(values) - 1)
    left_error = np.abs(values[left] - query)
    right_error = np.abs(values[right] - query)
    choose_right = right_error < left_error
    indices = np.where(choose_right, right, left)
    return indices.astype(np.int64), np.minimum(left_error, right_error)


def read_csv_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header: %s" % path)
        rows = list(reader)
    if not rows:
        raise ValueError("empty CSV: %s" % path)
    return list(reader.fieldnames), rows


def timestamp_column(rows: List[dict], preferred: str) -> np.ndarray:
    if preferred not in rows[0]:
        raise ValueError("missing timestamp column %s" % preferred)
    result = np.asarray([parse_float(row[preferred]) / 1e9 for row in rows], dtype=np.float64)
    if not np.isfinite(result).all() or not (np.diff(result) >= 0).all():
        raise ValueError("timestamp stream is not finite and monotonic")
    return result


def xyz(rows: List[dict], prefix: str) -> np.ndarray:
    return np.asarray(
        [[parse_float(row[prefix + axis]) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    )


def quat_xyzw(rows: List[dict], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [
                parse_float(row[prefix + axis])
                for axis in ("x", "y", "z", "w")
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(translation: np.ndarray, quat: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quat_to_matrix(quat)
    result[:3, 3] = translation
    return result


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def urdf_ft0_to_my(path: Path) -> np.ndarray:
    root = ET.parse(str(path)).getroot()
    transforms = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is None:
            continue
        xyz_text = origin.attrib.get("xyz", "0 0 0").split()
        rpy_text = origin.attrib.get("rpy", "0 0 0").split()
        translation = np.asarray([float(x) for x in xyz_text], dtype=np.float64)
        rpy = [float(x) for x in rpy_text]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rpy_matrix(*rpy)
        transform[:3, 3] = translation
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        transforms[(parent, child)] = transform
    first = transforms[("ft0", "ftModel")]
    second = transforms[("ftModel", "myframe")]
    return first @ second


def frame_wrench_to_parent(
    force_child: np.ndarray, torque_child: np.ndarray, transform_parent_child: np.ndarray
) -> np.ndarray:
    """Move a wrench at child origin into the parent frame/origin."""
    rotation = transform_parent_child[:3, :3]
    lever = transform_parent_child[:3, 3]
    force_parent = rotation @ force_child
    torque_parent = rotation @ torque_child + np.cross(lever, force_parent)
    return np.concatenate([force_parent, torque_parent])


def read_image_times(directory: Path, pattern: re.Pattern) -> Tuple[np.ndarray, List[str]]:
    paths = []
    times = []
    for path in directory.iterdir():
        match = pattern.search(path.name)
        if match:
            paths.append(path)
            times.append(float(match.group("ts")))
    if not paths:
        raise ValueError("no timestamped files in %s" % directory)
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], [str(paths[i]) for i in order]


def read_camera_info(path: Path) -> dict:
    _, rows = read_csv_rows(path)
    row = rows[0]
    keys = lambda prefix: [prefix + str(i) for i in range(9)]
    k = np.asarray([parse_float(row[key]) for key in keys("field.K")], dtype=np.float64).reshape(3, 3)
    d = np.asarray([parse_float(row["field.D" + str(i)]) for i in range(5)], dtype=np.float64)
    return {
        "width": int(float(row["field.width"])),
        "height": int(float(row["field.height"])),
        "distortion_model": row["field.distortion_model"],
        "K": k,
        "D": d,
        "frame_id": row["field.header.frame_id"],
    }


def sample_depth_world(
    path: Path,
    camera_to_world: np.ndarray,
    camera_info: dict,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.loadtxt(path, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth is not a matrix: %s" % path)
    rows = np.arange(0, depth.shape[0], max(1, stride), dtype=np.int64)
    cols = np.arange(0, depth.shape[1], max(1, stride), dtype=np.int64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    values = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    fx, fy = camera_info["K"][0, 0], camera_info["K"][1, 1]
    cx, cy = camera_info["K"][0, 2], camera_info["K"][1, 2]
    z = np.where(valid, values, 0.0)
    points_camera = np.stack(
        [(xx - cx) * z / fx, (yy - cy) * z / fy, z], axis=-1
    ).reshape(-1, 3)
    points_world = (camera_to_world[:3, :3] @ points_camera.T).T + camera_to_world[:3, 3]
    return points_world.astype(np.float32), valid.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction", type=Path, required=True)
    parser.add_argument("--ft-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-stride", type=int, default=32)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--match-tolerance-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists: %s" % args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    root = args.interaction
    rgb_times, rgb_paths = read_image_times(root / "camera_rgb", RGB_RE)
    depth_times, depth_paths = read_image_times(root / "camera_depth_registered", DEPTH_RE)
    if args.max_anchors > 0:
        rgb_times, rgb_paths = rgb_times[: args.max_anchors], rgb_paths[: args.max_anchors]
    anchor_count = len(rgb_times)
    if anchor_count < 4:
        raise ValueError("too few RGB anchors")

    _, ft_rows = read_csv_rows(root / "ft_sensor_netft_data.csv")
    _, joint_rows = read_csv_rows(root / "cardboardbox_joint_states.csv")
    _, rb_rows = read_csv_rows(root / "rb_poses_array.csv")
    _, tf_rows = read_csv_rows(root / "tf.csv")
    ft_times = timestamp_column(ft_rows, "field.header.stamp")
    joint_times = timestamp_column(joint_rows, "field.header.stamp")
    rb_times = timestamp_column(rb_rows, "field.markers0.header.stamp")

    def select_tf(child: str) -> Tuple[np.ndarray, np.ndarray]:
        wanted = child.lstrip("/")
        selected = [
            row
            for row in tf_rows
            if row["field.transforms0.child_frame_id"].lstrip("/") == wanted
        ]
        if not selected:
            raise ValueError("missing TF child %s" % child)
        return timestamp_column(selected, "field.transforms0.header.stamp"), selected

    # The archive records the moving camera pose as AsusXtionCameraFrame at
    # roughly 100 Hz.  The optical-frame TF rows are only a 10 Hz static-style
    # stream and would violate the 20 ms anchor contract.  We keep the camera
    # info frame and this pose-frame distinction explicit in the manifest.
    camera_tf_times, camera_tf_rows = select_tf("AsusXtionCameraFrame")
    object_tf_times, object_tf_rows = select_tf("TransparentBoxWith4Markers")
    ft_tf_times, ft_tf_rows = select_tf("ftSensor")

    ft_idx, ft_error = nearest(ft_times, rgb_times)
    joint_idx, joint_error = nearest(joint_times, rgb_times)
    rb_idx, rb_error = nearest(rb_times, rgb_times)
    camera_idx, camera_error = nearest(camera_tf_times, rgb_times)
    object_idx, object_error = nearest(object_tf_times, rgb_times)
    ft_tf_idx, ft_tf_error = nearest(ft_tf_times, rgb_times)
    depth_idx, depth_error = nearest(depth_times, rgb_times)
    tolerance = args.match_tolerance_ms / 1000.0
    all_errors = np.concatenate(
        [ft_error, joint_error, rb_error, camera_error, object_error, ft_tf_error, depth_error]
    )
    if float(all_errors.max()) > tolerance:
        raise ValueError("timestamp match exceeds tolerance: %.6f s" % float(all_errors.max()))

    ft_values = np.asarray(
        [
            [
                parse_float(row["field.wrench.force." + axis])
                for axis in ("x", "y", "z")
            ]
            + [
                parse_float(row["field.wrench.torque." + axis])
                for axis in ("x", "y", "z")
            ]
            for row in ft_rows
        ],
        dtype=np.float64,
    )
    ft0_to_my = urdf_ft0_to_my(args.ft_urdf)
    wrench_object = []
    camera_to_world = []
    object_pose = []
    joint_position = []
    rb0_pose = []
    point_rows = []
    point_valid_rows = []
    for i in range(anchor_count):
        camera_row = camera_tf_rows[int(camera_idx[i])]
        object_row = object_tf_rows[int(object_idx[i])]
        ft_row = ft_tf_rows[int(ft_tf_idx[i])]
        t_map_camera = pose_matrix(
            xyz([camera_row], "field.transforms0.transform.translation." )[0],
            quat_xyzw([camera_row], "field.transforms0.transform.rotation.")[0],
        )
        t_map_object = pose_matrix(
            xyz([object_row], "field.transforms0.transform.translation.")[0],
            quat_xyzw([object_row], "field.transforms0.transform.rotation.")[0],
        )
        t_map_ft = pose_matrix(
            xyz([ft_row], "field.transforms0.transform.translation.")[0],
            quat_xyzw([ft_row], "field.transforms0.transform.rotation.")[0],
        )
        t_object_my = np.linalg.inv(t_map_object) @ t_map_ft @ ft0_to_my
        wrench_object.append(
            frame_wrench_to_parent(ft_values[int(ft_idx[i]), :3], ft_values[int(ft_idx[i]), 3:], t_object_my)
        )
        camera_to_world.append(t_map_camera)
        object_pose.append(t_map_object)
        joint_position.append(parse_float(joint_rows[int(joint_idx[i])]["field.position0"]))
        rb_row = rb_rows[int(rb_idx[i])]
        rb0_pose.append(
            pose_matrix(
                xyz([rb_row], "field.markers0.pose.position.")[0],
                quat_xyzw([rb_row], "field.markers0.pose.orientation.")[0],
            )
        )
        points, valid = sample_depth_world(
            Path(depth_paths[int(depth_idx[i])]),
            t_map_camera,
            read_camera_info(root / "camera_depth_registered_camera_info.csv"),
            args.point_stride,
        )
        point_rows.append(points)
        point_valid_rows.append(valid)

    output_npz = args.output / "rbo_aligned.npz"
    np.savez_compressed(
        output_npz,
        anchor_timestamps=rgb_times.astype(np.float64),
        camera_to_world=np.asarray(camera_to_world, dtype=np.float32),
        object_pose=np.asarray(object_pose, dtype=np.float32),
        rb0_pose=np.asarray(rb0_pose, dtype=np.float32),
        joint_position=np.asarray(joint_position, dtype=np.float32),
        wrench_sensor_myframe=ft_values[ft_idx].astype(np.float32),
        wrench_object_geometric=np.asarray(wrench_object, dtype=np.float32),
        world_points=np.asarray(point_rows, dtype=np.float32),
        point_valid=np.asarray(point_valid_rows, dtype=bool),
    )
    source_files = [
        "ft_sensor_netft_data.csv",
        "cardboardbox_joint_states.csv",
        "rb_poses_array.csv",
        "tf.csv",
        "camera_rgb_camera_info.csv",
        "camera_depth_registered_camera_info.csv",
    ]
    source_hashes = {name: sha256(root / name) for name in source_files}
    source_hashes["ftSensor.urdf"] = sha256(args.ft_urdf)
    manifest = {
        "schema": "ref2dex.rbo-wrench-adapter.v1",
        "status": "COMPLETED_ALIGNMENT_SMOKE",
        "source": "rbo_articulated_objects",
        "zenodo_record": "1036660",
        "interaction": root.name,
        "anchor_count": anchor_count,
        "anchor_hz_nominal": 30,
        "point_stride": args.point_stride,
        "match_tolerance_ms": args.match_tolerance_ms,
        "max_match_error_ms": float(all_errors.max() * 1000.0),
        "median_match_error_ms": float(np.median(all_errors) * 1000.0),
        "streams": ["rgb", "registered_depth", "camera_info", "tf", "rb_pose", "joint", "ft_wrench"],
        "world_frame": "/map",
        "object_base_frame": "/TransparentBoxWith4Markers",
        "ft_sensor_frame": "ftSensor",
        "wrench_measurement_frame": "myframe",
        "position_unit_status": "inferred_meters",
        "force_unit_source": "visualizer+ROS_Wrench; preserve_raw_values",
        "payload_compensation": False,
        "wrench_object_semantics": "geometric frame/lever-arm transform only; not object-COM net wrench",
        "depth_unit_status": "inferred_meters",
        "camera_pose_frame": "AsusXtionCameraFrame",
        "point_contract": "registered depth sampled in camera optical frame and transformed with the recorded AsusXtionCameraFrame pose; optical-to-pose calibration status preserved",
        "source_file_hashes": source_hashes,
        "output_file": output_npz.name,
        "output_sha256": sha256(output_npz),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
