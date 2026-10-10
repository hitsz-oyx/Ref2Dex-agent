import numpy as np

from trajectory_policy.prefix_fitting import temporal_maps, future_xyz_velocity, fit_prefix


def test_projection_recovers_in_basis_curves_and_actual_future_only_velocity():
    p, v, f = temporal_maps(1/30)
    nodes = np.arange(24).reshape(2, 4, 3)/100
    xyz = np.einsum('tk,nkd->ntd', p, nodes)
    np.testing.assert_allclose(future_xyz_velocity(xyz, 1/30), np.einsum('tk,nkd->ntd', v, nodes), atol=1e-14)
    np.testing.assert_allclose(xyz+.1*future_xyz_velocity(xyz, 1/30), np.einsum('tk,nkd->ntd', f, nodes), atol=1e-14)
    fit = fit_prefix(xyz)
    np.testing.assert_allclose(fit['nodes'], nodes, atol=1e-13)
    assert max(fit['normal_errors'].values()) < 1e-12


def test_prefix_projection_cannot_remove_off_basis_curvature():
    xyz = np.zeros((1, 24, 3))
    xyz[0, :8, 0] = np.arange(8)**2/100
    fit = fit_prefix(xyz)
    assert np.sqrt((fit['position_bound_error']**2).sum(-1).mean()) > .01
    assert fit['normal_errors']['position'] < 1e-12
