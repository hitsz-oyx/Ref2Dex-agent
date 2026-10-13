#!/usr/bin/env python3
"""Bounded engineering checks; never fabricate robot labels for human video."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from einops import rearrange


def module_attempt(name):
    try:
        module = importlib.import_module(name)
        return dict(module=name, status='IMPORT_OK', version=getattr(module, '__version__', None))
    except Exception as error:
        return dict(module=name, status='IMPORT_FAILED', error=type(error).__name__ + ': ' + str(error))


def isolated_static_helpers(source, namespace):
    """Execute unchanged upstream methods without faking their missing imports."""
    parsed = ast.parse(source.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef)
               and n.name == 'DiffusionFlowBCCloseLoopDataset')
    names = {'normalize_points', 'process_point_tracking_data'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = []; cls.keywords = []
    tree = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    return namespace[cls.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--im2flow-root', type=Path, required=True)
    parser.add_argument('--egodex-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.im2flow_root.resolve()))
    report = dict(schema='ref2dex.egodex-upstream-engineering-audit.v1',
                  scope='CPU import, unmodified helper and real-data coordinate checks only; no policy/learning claim',
                  training_allowed=False, fabricated_robot_labels=False)
    report['imports'] = [module_attempt(n) for n in [
        'im2flow2act.common.pointcloud',
        'im2flow2act.common.utility.arr',
        'im2flow2act.diffusion_policy.dataloader.diffusion_flow_bc_dataset',
        'im2flow2act.flow_generation.dataloader.animateflow_dataset']]
    from im2flow2act.common.pointcloud import get_pointcloud
    from im2flow2act.common.utility.arr import complete_random_sampling
    source = args.im2flow_root / 'im2flow2act/diffusion_policy/dataloader/diffusion_flow_bc_dataset.py'
    helper = isolated_static_helpers(source, dict(np=np, rearrange=rearrange,
                                                complete_random_sampling=complete_random_sampling))
    report['helper_execution'] = 'Unmodified AST-extracted normalize_points/process_point_tracking_data; not full loader import'
    with np.load(args.egodex_audit / 'inputs/r03/clip.npz') as file:
        sample = {k: file[k] for k in file.files}
    with np.load(args.egodex_audit / 'inputs/r03/projection.npz') as file:
        projection = {k: file[k] for k in file.files}
    with np.load(args.egodex_audit / 'goals/r03.npz') as file:
        goal = {k: file[k] for k in file.files}
    report['real_sample'] = dict(id='r03', fields={k: list(v.shape) for k, v in sample.items()},
                                native_robot_action_available='action' in sample,
                                native_robot_proprioception_available='proprioception' in sample,
                                original_object_goal_available=bool(sample['object_goal_valid'].any()))
    selected = goal['current_object_valid']
    xyz = np.concatenate([goal['start_object_points'][None], goal['desired_object_points']], axis=0)[:, selected]
    valid = np.concatenate([goal['current_object_valid'][None], goal['desired_object_valid']], axis=0)[:, selected]
    intrinsic = projection['intrinsic']
    pix = xyz @ intrinsic.T
    uv = pix[..., :2] / np.where(valid, pix[..., 2], 1)[..., None]
    # Explicit reference-camera 2D projection. Do not silently use future camera poses.
    tracks = np.concatenate([uv * np.array([256/640, 256/360]), valid[..., None]], -1).astype(np.float32)
    np.random.seed(0)
    packed = helper.process_point_tracking_data(
        tracks.copy(), np.ones(len(xyz[0]), bool), num_points_to_sample=32,
        point_tracking_img_size=[256, 256], robot_mask=np.zeros(len(xyz[0]), bool),
        object_sampling=True, herustic_filter=[])
    report['im2flow_point_helper'] = dict(input_points=len(xyz[0]), output_shape=list(packed.shape),
        finite=bool(np.isfinite(packed).all()), original_mask_preserved=bool(np.all(packed[..., 2] == 1)),
        output_unique_current_points=int(len(np.unique(packed[0, :, :2], axis=0))),
        coordinate_frame='virtual fixed current camera; not a complete moving-camera policy adaptation')
    assert packed.shape == (4, 32, 3) and np.isfinite(packed).all()
    try:
        import torch
        from im2flow2act.common.flow_network import FlowPos2dConcat
        torch.set_num_threads(1)
        encoder = FlowPos2dConcat(embed_dim=16).eval()
        flows = torch.from_numpy(packed)[None]
        descriptors = torch.zeros((*flows.shape[:-1], 16))
        hidden = flows.clone(); hidden[..., 2] = 0
        moved_hidden = hidden.clone(); moved_hidden[..., :2] = 0
        with torch.no_grad():
            observed_features = encoder(descriptors, flows)
            hidden_features = encoder(descriptors, hidden)
            moved_features = encoder(descriptors, moved_hidden)
        report['im2flow_default_visibility_encoding'] = dict(
            status='UPSTREAM_MODULE_EXECUTED', output_shape=list(observed_features.shape),
            finite=bool(torch.isfinite(observed_features).all()),
            max_diff_visibility_toggle=float((observed_features-hidden_features).abs().max()),
            max_diff_invalid_coordinate_edit=float((hidden_features-moved_features).abs().max()),
            conclusion='Default pre/post_zero_out False ignores the visibility flag; invalid coordinates still encode a position',
            scope='Deterministic positional encoder only; zero descriptor fixture, not a robot state or action')
    except Exception as error:
        report['im2flow_default_visibility_encoding'] = dict(status='IMPORT_OR_EXECUTION_FAILED',
            error=type(error).__name__+': '+str(error))
    try:
        helper.process_point_tracking_data(tracks.copy(), np.zeros(len(xyz[0]), bool),
                                           num_points_to_sample=32, point_tracking_img_size=[256, 256])
        report['empty_target_mask'] = dict(status='UNEXPECTED_ACCEPTANCE')
    except ValueError as error:
        report['empty_target_mask'] = dict(status='EXPECTED_REJECTION', error=str(error))
    rectangular = np.array([[[320., 180., 1.]]], np.float32)
    received = helper.normalize_points(rectangular.copy(), 360, 640)
    report['rectangular_normalization'] = dict(input_xy=[320, 180], width=640, height=360,
        upstream_xy=received[0, 0, :2].tolist(), expected_xy=[.5, .5],
        conclusion='Default square resize is required, or x/W y/H normalization must be adapted')
    # Lift a sparse depth assembled from EXISTING pseudo-label z. This checks geometry
    # compatibility, never asserts independent depth reconstruction accuracy.
    pixel = np.rint(uv[0]).astype(int)
    depth = np.zeros((360, 640), np.float32)
    depth[pixel[:, 1], pixel[:, 0]] = xyz[0, :, 2]
    points, _, _ = get_pointcloud(depth, None, None, intrinsic, points=pixel)
    lift_error = np.linalg.norm(points - xyz[0], axis=-1)*1000
    report['upstream_backprojection'] = dict(points=len(pixel), median_mm=float(np.median(lift_error)),
        max_mm=float(lift_error.max()), depth_source='existing EgoDex pseudo-label z; integer-pixel quantization',
        independent_depth_quality_verified=False)
    camera_errors = []
    manifest = json.loads((args.egodex_audit / 'inputs/manifest.json').read_text())
    for row in manifest['records']:
        base = args.egodex_audit / 'inputs' / row['id']
        with np.load(base / 'clip.npz') as s, np.load(base / 'projection.npz') as p:
            current_future = s['point_target'][23]
            current_to_future_camera = np.linalg.inv(p['camera_to_origin'][3]) @ p['camera_to_origin'][0]
            future_camera = current_future @ current_to_future_camera[:3, :3].T + current_to_future_camera[:3, 3]
            common = s['point_target_valid'][23]
            error = np.linalg.norm(future_camera[common]-current_future[common], axis=-1)*1000
            camera_errors.append(dict(id=row['id'], points=int(common.sum()),
                                      median_mm=float(np.median(error)), p95_mm=float(np.percentile(error, 95))))
    report['omitting_time_varying_camera_error'] = camera_errors
    report['camera_error_mean_median_mm'] = float(np.mean([r['median_mm'] for r in camera_errors]))
    report['camera_error_max_p95_mm'] = float(max(r['p95_mm'] for r in camera_errors))
    np.savez_compressed(args.output / 'im2flow-helper-r03.npz', tracks=packed, reference_current=xyz[0])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
