import numpy as np
from scipy.spatial.transform import Rotation

from trajectory_policy.dense_decoder import DenseDecoder
from consequence_evaluator.reference_tracking import apply_coupling, future_reference_velocity
import torch

URDF = 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'


def test_dense_action_retains_native_branch_coupling_and_feedforward():
    # Tiny CPU interface check, including crossing the principal Euler branch.
    current = np.zeros((1, 18), np.float32)
    current[0, 3] = 3.13
    future = np.repeat(current[:, None], 24, axis=1)
    future[0, :, 0] = np.linspace(.01, .4, 24)
    future[0, :, 3] = np.linspace(3.14, 3.4, 24)
    future[0, :, [6, 8, 10, 12, 14, 15]] = np.asarray([.8, .9, 1., 1.1, .5, .2])[:, None]
    future = apply_coupling(torch.as_tensor(future)).numpy()
    obj = np.eye(4, dtype=np.float32)[None]
    obj[0, :3, :3] = Rotation.from_euler('xyz', [.2, -.4, .6]).as_matrix()
    decoder = DenseDecoder(URDF, 'cpu')
    c = decoder.encode(future, current, obj)
    decoded = decoder.decode(c, current, obj, 1/30)
    expected = np.concatenate((current[:, None], future), 1)
    np.testing.assert_allclose(decoded['q'].numpy(), expected, atol=5e-7)
    np.testing.assert_allclose(decoded['velocity'][0].numpy(),
                               future_reference_velocity(torch.as_tensor(expected[0]), 1/30).numpy(), atol=1e-5)
    c[0, 0] += .3
    changed = decoder.decode(c, current, obj, 1/30)['q'].numpy()
    assert np.linalg.norm(changed[0, 1, :3]-expected[0, 1, :3]) > .0029
    np.testing.assert_allclose(changed[:, 2:], expected[:, 2:], atol=5e-7)
