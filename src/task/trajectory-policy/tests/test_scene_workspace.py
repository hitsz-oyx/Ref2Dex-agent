import torch

from trajectory_policy.scene_workspace import mask_scene_outliers


def test_one_boundary_point_is_dropped_and_a_shifted_cloud_still_fails():
    origin = (-2., -2., -2.)
    xyz = torch.tensor([[
        [0., 0., 0.],
        [-2.005, 0., 0.],
    ]])
    batch = dict(xyz=xyz, point_valid=torch.ones(1, 2, dtype=torch.bool))
    assert mask_scene_outliers(batch, origin, 0.01) == 1
    assert batch['point_valid'].tolist() == [[True, False]]
    shifted = dict(xyz=xyz-3, point_valid=torch.ones(1, 2, dtype=torch.bool))
    try:
        mask_scene_outliers(shifted, origin, 0.01)
    except ValueError as error:
        assert 'exceeds' in str(error)
    else:
        raise AssertionError('a fully shifted cloud must still fail')
    nine = dict(xyz=torch.cat((xyz[:, :1], xyz[:, 1:].expand(1, 9, 3)), 1),
                point_valid=torch.ones(1, 10, dtype=torch.bool))
    assert mask_scene_outliers(nine, origin, 0.01, limit=64) == 9


def test_outlier_limit_is_per_environment_row():
    origin = (-2., -2., -2.)
    row = torch.tensor([[0., 0., 0.], [-2.005, 0., 0.]])
    mild = dict(xyz=row[None].repeat(4, 1, 1), point_valid=torch.ones(4, 2, dtype=torch.bool))
    assert mask_scene_outliers(mild, origin, 0.01, limit=1) == 4
    assert mild['point_valid'][:, 1].tolist() == [False, False, False, False]
    cloud = torch.zeros(4, 3, 3)
    cloud[:, 2] = torch.tensor([-2.005, 0., 0.])
    cloud[1, 1] = torch.tensor([-2.005, 0., 0.])
    bad = dict(xyz=cloud, point_valid=torch.ones(4, 3, dtype=torch.bool))
    try:
        mask_scene_outliers(bad, origin, 0.01, limit=1)
    except ValueError as error:
        assert 'exceeds' in str(error)
    else:
        raise AssertionError('one row over the limit must fail the batch')
