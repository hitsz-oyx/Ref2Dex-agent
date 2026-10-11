from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from trajectory_policy.metric_decoder import coordinates, MetricDecoder, SCHEMA, fit_basis
from consequence_evaluator.retargeter import wrist_rotation

URDF = 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'


def test_dense_physical_coordinates_retain_pose_and_latent_changes_intent():
    folder = Path('tmp/trajectory-policy-tests')
    folder.mkdir(parents=True, exist_ok=True)
    current = np.zeros((1, 18), np.float32)
    current[0, 3] = 3.13
    future = np.repeat(current[:, None], 24, axis=1)
    future[0, :, 0] = np.linspace(.01, .4, 24)
    future[0, :, 3] = np.linspace(3.14, 3.4, 24)
    future[0, :, [6, 8, 10, 12, 14, 15]] = np.asarray([.8, .9, 1., 1.1, .5, .2])[:, None]
    obj = np.eye(4, dtype=np.float32)[None]
    obj[0, :3, :3] = Rotation.from_euler('z', .6).as_matrix()
    component = np.eye(288, dtype=np.float32)[:48]
    basis = folder/'metric-roundtrip-basis.npz'
    np.savez(basis, schema=np.asarray(SCHEMA), mean=coordinates(future, current, obj)[0],
        components=component, encoder_components=component.T, latent_scale=np.ones(48, np.float32))
    decoder = MetricDecoder(URDF, 'cpu', basis)
    c = decoder.encode(future, current, obj)
    q = decoder.decode(c, current, obj, 1/30)['q'].numpy()[:, 1:]
    np.testing.assert_allclose(q[:, :, :3], future[:, :, :3], atol=2e-7)
    np.testing.assert_allclose(wrist_rotation(q), wrist_rotation(future), atol=5e-7)
    c[0, 0] += .3
    changed = decoder.decode(c, current, obj, 1/30)['q'].numpy()[:, 1:]
    assert np.linalg.norm(changed[0, 0, :3]-q[0, 0, :3]) > .0029
    np.testing.assert_allclose(changed[:, 1:, :3], q[:, 1:, :3], atol=2e-7)


def test_weighted_basis_projection_is_dual_and_reproduces_training_span():
    # Tiny mathematical smoke on CPU; actual543x288 fit uses GPU.
    rng = np.random.default_rng(5)
    values = rng.normal(size=(60, 288))@np.diag(np.linspace(.01, .1, 288))
    values[10:] = values[:10][np.arange(50)%10]
    basis = fit_basis(values, 'cpu')
    scores = (values-basis['mean'])@basis['encoder_components']
    reconstruction = basis['mean']+scores@basis['components']
    np.testing.assert_allclose(reconstruction, values, atol=1e-6)
    assert float(basis['dual_error']) < 1e-10
