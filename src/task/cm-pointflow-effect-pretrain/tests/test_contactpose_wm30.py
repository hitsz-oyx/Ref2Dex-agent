"""Physical transforms, real clocks, and gap boundaries in native ContactPose."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.contactpose import (OPENPOSE_11, contactpose_windows, native_trajectory,
                                  resample_segments, subject_split, window_rows)


def payload(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return dict(rotation=q[[3, 0, 1, 2]].tolist(), translation=T[:3, 3].tolist())


def annotation(n=60, moving=False):
    joints = np.arange(63).reshape(21, 3) / 1000
    hands = [dict(valid=False, moving=False, joints=np.zeros((21, 3)).tolist()),
             dict(valid=True, moving=moving, joints=joints.tolist())]
    frames = []
    for i in range(n):
        world = np.eye(4)
        world[:3, :3] = Rotation.from_euler('z', i/300).as_matrix()
        world[:3, 3] = [i/300, 1., 2.]
        ns = int(round(i/30 * 1e9))
        frame = dict(oTw=payload(np.linalg.inv(world)), time={'kinect2_middle': dict(sec=1000+ns//10**9, nsec=ns%10**9)})
        if moving:
            hTo = np.eye(4); hTo[1, 3] = i/1000
            frame['hTo'] = [payload(np.eye(4)), payload(hTo)]
        frames.append(frame)
    return dict(hands=hands, cameras={'kinect2_middle': dict(valid=True)}, frames=frames)


def test_oTw_inverse_joint_order_and_missing_hand_mask():
    a = annotation()
    raw = native_trajectory(a)
    assert OPENPOSE_11 == [0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20]
    assert raw['timestamp_origin_ns'] == 1000 * 10**9
    assert np.allclose(raw['times'], np.arange(60)/30, atol=1e-9)
    assert np.allclose(raw['poses'][30, :3, 3], [.1, 1., 2.])
    expected = np.asarray(a['hands'][1]['joints'])[OPENPOSE_11]
    expected = expected @ raw['poses'][30, :3, :3].T + raw['poses'][30, :3, 3]
    assert np.allclose(raw['hand'][30, 0], expected)
    assert raw['hand_valid'][:, 0].all() and not raw['hand_valid'][:, 1].any()
    assert np.all(raw['hand'][:, 1] == 0)


def test_moving_hTo_is_inverted_in_object_frame():
    a = annotation(moving=True)
    raw = native_trajectory(a)
    joints = np.asarray(a['hands'][1]['joints'])[OPENPOSE_11]
    world = raw['poses'][30]
    expected = (joints - [0, .03, 0]) @ world[:3, :3].T + world[:3, 3]
    assert np.allclose(raw['hand'][30, 0], expected)
    packs, audit = resample_segments(raw)
    assert len(packs) == 1 and audit['gaps'] == 0
    # Hand/world-object coupling remains rigid during rotation interpolation.
    pack = packs[0]
    local = np.einsum('tji,tnj->tni', pack['poses'][:, 0, :3, :3],
                      pack['hand'][:, 0] - pack['poses'][:, 0, None, :3, 3])
    assert np.allclose(local[:, 0, 1], joints[0, 1] - pack['timestamps'] * .03, atol=1e-6)


def test_long_timestamp_gap_is_split_and_windows_never_cross_it():
    a = annotation(100)
    for frame in a['frames'][40:]:
        frame['time']['kinect2_middle']['sec'] += 2
    raw = native_trajectory(a)
    packs, audit = resample_segments(raw)
    assert audit['gaps'] == 1 and len(packs) == 2
    for pack in packs:
        assert np.allclose(np.diff(pack['timestamps']), 1/30, atol=1e-7)
        rows = window_rows(pack, np.ones((len(pack['hand']), 1), bool))
        for _, tick, _ in rows:
            source = pack['source_frame_ids'][tick-3:tick+25]
            assert (source < 40).all() or (source >= 40).all()
    assert packs[0]['raw_range'] == [0, 40] and packs[1]['raw_range'] == [40, 100]


def test_nonmonotonic_clock_rejected_and_missing_moving_pose_masked():
    a = annotation(moving=True)
    a['frames'][20].pop('hTo')
    raw = native_trajectory(a)
    assert not raw['hand_valid'][20, 0]
    packs, audit = resample_segments(raw)
    assert audit['invalid_raw_frames'] == 1
    assert all(not (p['raw_range'][0] <= 20 < p['raw_range'][1]) for p in packs)
    a['frames'][12]['time'] = a['frames'][11]['time']
    with pytest.raises(ValueError, match='strictly increasing'):
        native_trajectory(a)


def test_current_proximity_only_and_invalid_future_pose_rejected():
    packs, _ = resample_segments(native_trajectory(annotation()))
    p = packs[0]
    near = np.zeros((len(p['hand']), 1), bool)
    near[15:] = True
    rows = window_rows(p, near)
    assert (rows[:, 1] >= 15).all()  # no selection from future contact
    p['pose_valid'][25] = False
    assert not any(t-3 <= 25 < t+25 for _, t, _ in window_rows(p, near))


def test_subject_split_reuses_group_and_adapter_retains_source_ids(tmp_path):
    assert subject_split('full11') == 'train'
    assert subject_split('full6') == 'val'
    assert subject_split('full10') == 'test'
    packs, _ = resample_segments(native_trajectory(annotation()))
    p = packs[0]
    root = tmp_path / 'pack'
    dest = root / 'processed/sequences/contactpose_fixture'
    dest.mkdir(parents=True)
    p.update(program=np.zeros((len(p['hand']), 1), bool), near=np.ones((len(p['hand']), 1), bool), centers=np.zeros((1, 3)))
    for k, v in p.items():
        if k != 'raw_range': np.save(dest / (k+'.npy'), v)
    rows = window_rows(p, p['near'])
    np.save(root / 'processed/index_train.npy', np.column_stack((np.zeros(len(rows), int), rows)))
    (dest/'meta.json').write_text(json.dumps(dict(objects=['contactpose_fixture'])))
    (root/'processed/manifest.json').write_text(json.dumps(dict(status='COMPLETED', fps=30, sequences=['contactpose_fixture'])))
    can = root/'processed/canonical';can.mkdir()
    points = np.zeros((512, 3)); points[:, 0] = np.linspace(-.04, .04, 512)
    np.savez(can/'contactpose_fixture.npz', points=points, normals=np.ones_like(points), radius=.04, center=np.zeros(3))
    before = np.load(dest/'frame_ids.npy').copy()
    dataset = contactpose_windows(root, 'train')
    sample = dataset[0]
    assert sample['points'].shape == (1, 512, 3) and sample['effect'].shape == (1, 24, 4, 4)
    assert sample['action'].shape == (24, 22, 9)
    assert sample['action_valid'][:, :11].all() and not sample['action_valid'][:, 11:].any()
    assert np.array_equal(np.load(dest/'frame_ids.npy'), before)
    assert np.isfinite(sample['action']).all() and np.isfinite(sample['effect']).all()
