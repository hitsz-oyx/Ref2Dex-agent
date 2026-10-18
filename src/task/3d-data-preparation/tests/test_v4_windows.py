import numpy as np

from pose_mesh_data.v4_windows import v4_pose_mesh_sample


def test_v4_sample_is_target_only_and_has_fixed_contract():
    n = 40
    hand = np.zeros((n, 2, 11, 3), np.float32)
    hand[:, 0, :, 0] = .02
    valid = np.ones((n, 2, 11), bool)
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 2, 1, 1))
    poses[:, 0, 2, 3] = np.arange(n, dtype=np.float32) * .001
    seq = dict(time=np.arange(n) / 30., hand=hand, hand_valid=valid,
               object_poses=poses, object_pose_valid=np.ones((n, 2), bool),
               segment=np.zeros(n, np.int64))
    surface = np.stack((np.linspace(-.1, .1, 512), np.zeros(512), np.zeros(512)), -1).astype(np.float32)
    geometry = dict(surface=surface, normals=np.tile([0., 1., 0.], (512, 1)).astype(np.float32))
    sample = v4_pose_mesh_sample(seq, [geometry, geometry], 3, 0, 8, [0, 3], 'test')
    assert sample['target_object_only']
    assert sample['xyz'].shape == (512, 3)
    assert sample['scene_points'].shape == (1, 32, 3)
    assert sample['interaction_points'].shape == (1, 128, 3)
    assert sample['interaction_target'].shape == (1, 24, 128, 7)
    assert sample['object_valid'].tolist() == [True]
    assert np.isfinite(sample['interaction_target']).all()
    # Same target/static contract as main3 and the online runtime.
    assert np.all(sample['features'][:, 12] == 1.)
    assert np.all(sample['features'][:, 13] == 0.)  # object moves 1 mm/frame in H4
    assert sample['interaction_origin_semantics'] == 'current-interaction-min-minus-5cm.v1'
    assert sample['schema'] == 'ref2dex.wam-v4-target-object.v4'
    seq['object_pose_valid'][8, 0] = False
    missing = v4_pose_mesh_sample(seq, [geometry, geometry], 3, 0, 8, [0, 3], 'test')
    assert not missing['interaction_valid'][:, 4].any()
    assert not missing['interaction_target'][:, 4].any()
