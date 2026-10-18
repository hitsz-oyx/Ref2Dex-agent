import numpy as np
import torch

from pose_mesh_data.v4_data import _main3_target_sample, v4_collate


def sample():
    points = np.linspace(-.1, .1, 6 * 512 * 3, dtype=np.float32).reshape(6, 512, 3)
    effects = np.tile(np.eye(4, dtype=np.float32), (6, 24, 1, 1))
    xyz = points.reshape(-1, 3)
    features = np.zeros((len(xyz), 18), np.float32)
    scene_object = np.repeat(np.arange(6), 512)
    object_features = np.zeros((6, 15), np.float32); object_features[2, 13] = 1
    return dict(points=points, effect=effects, xyz=xyz, features=features, scene_object=scene_object,
                object_features=object_features, source_id=np.int64(0), sample_id=np.array([0, 0, 3]),
                start_hand=np.zeros((2, 11, 3), np.float32), start_hand_history=np.zeros((2, 11, 9), np.float32),
                start_hand_valid=np.ones((2, 11), bool), trajectory_target=np.zeros((24, 2, 11, 3), np.float32),
                trajectory_valid=np.ones((24, 2, 11), bool))


def test_main3_view_and_collate_keep_dense_geometry_contract():
    s = _main3_target_sample(sample(), np.tile([0., 0., 1.], (512, 1)))
    b = v4_collate([s])
    assert b["xyz"].shape == (1, 512, 3)
    assert b["object_surface_normals"].shape == (1, 1, 512, 3)
    assert b["interaction_target"].shape == (1, 1, 24, 128, 7)
    assert b["scene_points"].shape == (1, 1, 32, 3)
    assert torch.isfinite(b["interaction_target"]).all()
    assert torch.all(b["features"][..., 12] == 1)
    assert torch.all(b["features"][..., 13] == 1)  # no H4 displacement in fixture
    moving = sample()
    moving["features"][:, 9:12] = np.array([.01, 0., 0.], np.float32)
    moved = _main3_target_sample(moving, np.tile([0., 0., 1.], (512, 1)))
    assert np.all(moved["features"][:, 13] == 0.)
