"""Independently replay GT-hand inputs, relative targets and native dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.retargeter import (SCHEMA, HandTrajectoryRetargeter,
    Standardizer, commanded_targets, absolute_targets, trajectory_input, native_control)
from consequence_evaluator.contracts import is_within


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path, gpu):
    with path.open('rb') as stream:
        packet = pickle.load(stream)
    if (packet.get('schema') != 'ref2dex.gt-hand-retarget-behavior.v1'
            or packet.get('training_allowed') is not False or len(packet['actions']) != 542
            or packet['done'][:-1].any()):
        raise ValueError('full bounded engineering packet required')
    source_path = Path(packet['gt_source'])
    checkpoint = Path(packet['checkpoint'])
    if sha(source_path) != packet['gt_source_sha256'] or sha(checkpoint) != packet['checkpoint_sha256']:
        raise ValueError('source/checkpoint hash drift')
    with source_path.open('rb') as stream:
        source = pickle.load(stream)
    ticks = np.arange(0, 542, 24)
    np.testing.assert_array_equal(packet['retarget_query_ticks'], ticks)
    np.testing.assert_array_equal(packet['actions'], packet['requested_controls'])
    np.testing.assert_array_equal(packet['desired_pd_targets'][:, 1], packet['desired_pd_targets'][:, 2])
    expected_gt = commanded_targets(source['dof_position'][:-1, 0], source['actions'][:, 0])
    np.testing.assert_array_equal(packet['desired_pd_targets'][:, 0], expected_gt)
    hands, states = [], []
    for index, tick in enumerate(ticks):
        value = packet['retarget_inputs'][index]
        assert set(value) == {'current_hand', 'future_hand', 'state'}
        future = source['hand_keypoints'][np.minimum(tick+np.arange(1, 25), 542), 0]
        np.testing.assert_array_equal(value['future_hand'], future)
        np.testing.assert_array_equal(value['current_hand'], packet['hand_keypoints'][tick, 2])
        state = np.r_[packet['dof_position'][tick, 2], packet['dof_velocity'][tick, 2]]
        np.testing.assert_array_equal(value['state'], state)
        hands.append(trajectory_input(value['current_hand'][None], future[None])[0])
        states.append(state)
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
    torch.set_num_threads(2)
    device = 'cuda:%d' % gpu
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['schema'] == SCHEMA
    model = HandTrajectoryRetargeter(payload['width']).to(device).eval()
    model.load_state_dict(payload['state_dict'], strict=True)
    stats = {key: Standardizer(**value) for key, value in payload['statistics'].items()}
    with torch.no_grad():
        prediction = stats['target'].decode(model(
            torch.as_tensor(stats['hand'].encode(hands), device=device),
            torch.as_tensor(stats['state'].encode(states), device=device)).cpu().numpy())
    replay_error = float(np.max(np.abs(prediction-packet['retarget_relative_chunks'])))
    if replay_error > 2e-5:
        raise ValueError('frozen learned inference replay mismatch: %s' % replay_error)
    absolute = absolute_targets(packet['dof_position'][ticks, 2], packet['retarget_relative_chunks'])
    np.testing.assert_allclose(absolute, packet['retarget_absolute_chunks'], atol=2e-6)
    for tick in range(542):
        for role in (1, 2):
            np.testing.assert_array_equal(packet['desired_pd_targets'][tick, role],
                packet['retarget_absolute_chunks'][tick//24, tick % 24])
    raw = native_control(packet['desired_pd_targets'], packet['dof_position'][:-1, 1:])
    np.testing.assert_allclose(np.clip(raw, -1, 1), packet['actions'][:, 1:], atol=2e-6)
    np.testing.assert_array_equal(np.sum(raw != np.clip(raw, -1, 1), -1), packet['clipped_coordinate_counts'])
    desired_error = np.max(np.abs(commanded_targets(packet['dof_position'][:-1, 1:],
        packet['actions'][:, 1:])-packet['desired_pd_targets']), -1)
    np.testing.assert_allclose(desired_error, packet['commanded_target_max_abs_error'], atol=2e-6)
    metrics = {}
    for role, name in enumerate(packet['role_names']):
        delta = packet['hand_keypoints'][1:, role]-source['hand_keypoints'][1:, 0]
        rmse = float(np.sqrt(np.mean(delta**2))*1000)
        assert abs(rmse-packet['metrics'][name]['hand_coordinate_rmse_mm']) < 1e-5
        metrics[name] = dict(hand_coordinate_rmse_mm=rmse,
            approach_rmse_mm=float(np.sqrt(np.mean(delta[:120]**2))*1000),
            post_approach_rmse_mm=float(np.sqrt(np.mean(delta[120:]**2))*1000))
    return dict(status='PASS', packet_sha256=sha(path), query_count=len(ticks),
        requested_native_max_abs=0., replay_relative_max_abs=replay_error,
        target_conversion_max_abs=desired_error.max(0).tolist(),
        clipping_counts=packet['clipped_coordinate_counts'].sum(0).tolist(),
        metrics=metrics, gate=packet['gate'], source_sha256=sha(source_path), checkpoint_sha256=sha(checkpoint),
        claim='wiring/causality audit; no task-success or deployability claim')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    a = p.parse_args()
    if (a.output.exists() or not is_within(a.output, ROOT / 'outputs/consequence-evaluator')
            or not is_within(a.packet, ROOT / 'outputs/consequence-evaluator')):
        p.error('task-owned packet and fresh output required')
    result = audit(a.packet.resolve(), a.gpu)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
