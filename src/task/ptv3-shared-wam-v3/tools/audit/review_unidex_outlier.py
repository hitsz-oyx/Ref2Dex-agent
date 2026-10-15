"""Independently replay one UniDexGrasp simulator outlier without model execution.

Reads immutable source records and writes a bounded forensic receipt. It does
not admit/reject samples, modify a recipe, or decide physical plausibility.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK.parent / '3d-data-preparation' / 'src'))
from pose_mesh_data.windows import pose_mesh_sample, xyzquat_to_matrix


def read_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--row', type=int, required=True)
    parser.add_argument('--tick', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('forensic receipts are immutable')
    root = args.root.resolve()
    meta = json.loads((root / 'manifest.json').read_text())
    row = meta['sequences'][args.row]
    if row['source_id'] != 4 or row['format'] != 'unidexgrasp-compact':
        raise ValueError('this diagnostic requires actual UniDexGrasp compact data')
    compact = read_npz(root / row['path'])
    parent_manifest = Path(row['parent_source_manifest'])
    parent = json.loads(parent_manifest.read_text())
    identity = next(item for item in parent['trajectories']
                    if item['path'] == row['source_sequence'])
    raw_root = parent_manifest.parent
    result = json.loads((raw_root / 'result.json').read_text())
    environment = result['environment_objects'][identity['env']]
    if environment['object_code'] != row['object_code'] or environment['scale'] != row['scale']:
        raise ValueError('simulator environment object identity mismatch')
    raw_path = raw_root / result['episodes'][identity['episode']]['path']
    env, tick = identity['env'], args.tick
    # Load only fields needed to distinguish clock/conversion errors from raw
    # simulator dynamics; no policy inference and no GPU context is created.
    with np.load(raw_path, allow_pickle=False) as raw:
        raw_time = raw['time'].copy()
        raw_object = raw['object'][:, env].copy()
        raw_hand = raw['hand'][:, env].copy()
        raw_obs_velocity = raw['observation'][:, env, 198:201].copy()
        raw_reset = raw['reset'][:, env].copy()
        raw_hand_root = raw['hand_root'][:, env].copy()
    selected = np.arange(0, len(raw_time), 2)
    exact = dict(time=np.array_equal(compact['time'], raw_time[selected]),
                 object=np.array_equal(compact['object'], raw_object[selected]),
                 hand=np.array_equal(compact['hand'], raw_hand[selected]))
    if not all(exact.values()):
        raise ValueError('compact payload differs from original observed 60Hz record')
    if not np.allclose(compact['time'], compact['action_time'][::2], atol=1e-12, rtol=0):
        raise ValueError('compact frame/action clock mismatch')
    poses = xyzquat_to_matrix(compact['object'])
    count = len(poses)
    hands = np.zeros((count, 2, 11, 3), np.float32)
    hands[:, 0] = compact['hand']
    hand_valid = np.zeros((count, 2, 11), bool)
    hand_valid[:, 0] = True
    sequence = dict(time=compact['time'], hand=hands, hand_valid=hand_valid,
                    object_poses=poses[:, None], object_pose_valid=np.ones((count, 1), bool),
                    segment=np.zeros(count, np.int64))
    geometry_row = next(item for item in meta['objects'] if item['key'] == row['objects'][0])
    geometry = read_npz(root / geometry_row['path'])
    query = geometry['query'].astype(np.float64)
    query_world = np.einsum('tij,pj->tpi', poses[:, :3, :3], query) + poses[:, None, :3, 3]
    sample = pose_mesh_sample(sequence, [geometry], tick, 0, 4, [args.row, tick], row['profile'])
    world_delta = query_world[tick+1:tick+25] - query_world[tick]
    expected = world_delta @ poses[tick, :3, :3]
    label_error = float(np.abs(expected - sample['scene_target'][0]).max())
    if label_error > 1e-5:
        raise ValueError('independent world-space reconstruction differs from training labels')
    centre = compact['object'][:, :3].astype(np.float64)
    speed = np.linalg.norm(np.diff(centre, axis=0) / np.diff(compact['time'])[:, None], axis=-1)
    raw_velocity = np.diff(raw_object[:, :3].astype(np.float64), axis=0) / np.diff(raw_time)[:, None]
    raw_speed = np.linalg.norm(raw_velocity, axis=-1)
    acceleration = np.diff(raw_velocity, axis=0) / np.diff(raw_time)[1:, None]
    distances = np.linalg.norm(compact['hand'].astype(np.float64) - centre[:, None], axis=-1)
    with np.load(root / meta['window_index'], allow_pickle=False) as indices:
        ticks = indices['verified_train'][indices['verified_train'][:, 0] == args.row, 1]
    windows = []
    for anchor in ticks:
        displacement = query_world[anchor+1:anchor+25] - query_world[anchor]
        norms = np.linalg.norm(displacement, axis=-1)
        windows.append(dict(tick=int(anchor), start_time_s=float(compact['time'][anchor]),
                            max_query_displacement_m=float(norms.max()),
                            endpoint_max_query_displacement_m=float(norms[-1].max())))
    trace = []
    for frame in range(max(0, 2*tick-4), min(len(raw_time), 2*tick+15)):
        trace.append(dict(frame_60hz=frame, time_s=float(raw_time[frame]),
                          object_xyz_m=raw_object[frame, :3].tolist(),
                          next_step_speed_m_s=float(raw_speed[frame]) if frame < len(raw_speed) else None,
                          recorded_speed_m_s=float(np.linalg.norm(raw_obs_velocity[frame])),
                          hand_root_xyz_m=raw_hand_root[frame, :3].tolist()))
    norms = np.linalg.norm(world_delta, axis=-1)
    max_frame, max_query = np.unravel_index(norms.argmax(), norms.shape)
    report = dict(
        scope='CPU independent source replay; no model inference, training change or data exclusion',
        source_name='UniDexGrasp++ official state policy, locally recorded simulator trajectories',
        units=dict(position='m', time='s', speed='m/s', acceleration='m/s^2'),
        provenance=dict(root=str(root), row=args.row, tick=tick, source_sequence=row['source_sequence'],
                        raw_episode=str(raw_path), env=env, object_code=row['object_code'],
                        scale=row['scale'], geometry=str(root / geometry_row['path'])),
        checks=dict(compact_equals_original_raw_every_second_frame=exact,
                    action_clock_matches_observations=True,
                    source_interval_s=float(np.diff(raw_time).mean()),
                    selected_interval_s=float(np.diff(compact['time']).mean()),
                    horizon_duration_s=float(compact['time'][tick+24] - compact['time'][tick]),
                    raw_reset_before_terminal=bool(raw_reset[:-1].any()),
                    world_reconstruction_max_abs_label_error_m=label_error),
        geometry=dict(cad_aabb_extent_m=np.ptp(geometry['surface'], axis=0).tolist(),
                      maximum_local_point_radius_m=float(np.linalg.norm(geometry['surface'], axis=1).max())),
        window=dict(anchor_object_xyz_m=centre[tick].tolist(), endpoint_object_xyz_m=centre[tick+24].tolist(),
                    centre_endpoint_displacement_m=float(np.linalg.norm(centre[tick+24]-centre[tick])),
                    maximum_query_displacement_m=float(norms.max()),
                    endpoint_max_query_displacement_m=float(norms[-1].max()),
                    query_rms_displacement_m=float(np.sqrt(np.square(norms).mean())),
                    max_at_future_frame=int(max_frame+1), max_query_id=int(max_query),
                    hand_landmark_to_object_centre_at_anchor_min_max_m=[float(distances[tick].min()), float(distances[tick].max())],
                    hand_landmark_to_object_centre_at_endpoint_min_max_m=[float(distances[tick+24].min()), float(distances[tick+24].max())]),
        sequence=dict(frames_30hz=count, maximum_speed_30hz_m_s=float(speed.max()),
                      maximum_speed_60hz_m_s=float(raw_speed.max()),
                      maximum_recorded_object_speed_m_s=float(np.linalg.norm(raw_obs_velocity, axis=1).max()),
                      maximum_fd_acceleration_60hz_m_s2=float(np.linalg.norm(acceleration, axis=1).max()),
                      maximum_hand_root_speed_m_s=float(np.linalg.norm(np.diff(raw_hand_root[:, :3], axis=0) / np.diff(raw_time)[:, None], axis=1).max()),
                      retained_train_windows=windows),
        onset_trace_60hz=trace,
        assessment=dict(conversion='Original 60Hz object states already contain the large excursion; clock, identity, CAD metre scale and label transform are consistent.',
                        physical_quality='Abnormal high-speed ejection is present in the simulator record. The record alone cannot distinguish actual contact response from simulator contact instability or bad initialization.',
                        limitation='No original render/contact impulse trace or deterministic simulator rerun inspected; numerical validity does not establish physical quality.',
                        next='Expose full-scene and trajectory view plus source-wise speed/acceleration/contact review; keep quality flags separate from filtering, and do not silently select future-successful windows.'))
    args.output.mkdir(parents=True)
    (args.output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('checks', 'geometry', 'window', 'sequence')}, indent=2))


if __name__ == '__main__':
    main()
