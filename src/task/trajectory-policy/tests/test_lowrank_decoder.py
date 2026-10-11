from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from trajectory_policy.lowrank_decoder import LowrankDecoder, coordinates, SCHEMA
from consequence_evaluator.retargeter import wrist_rotation

URDF = 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'


def test_dense_coordinate_inverse_with_rotated_live_frame_and_euler_branch():
    # Project-specific scratch, never system /tmp.
    folder = Path('tmp/trajectory-policy-tests')
    folder.mkdir(parents=True, exist_ok=True)
    basis = folder/'coordinate-roundtrip-basis.npz'
    current = np.zeros((1, 18), np.float32)
    current[0, 3] = 3.13
    future = np.repeat(current[:, None], 24, axis=1)
    future[0, :, 0] = np.linspace(.01, .4, 24)
    future[0, :, 3] = np.linspace(3.14, 3.4, 24)
    future[0, :, [6, 8, 10, 12, 14, 15]] = np.asarray([.8, .9, 1., 1.1, .5, .2])[:, None]
    obj = np.eye(4, dtype=np.float32)[None]
    obj[0, :3, :3] = Rotation.from_euler('z', .6).as_matrix()
    mean = coordinates(future, current, obj)[0]
    np.savez(basis, schema=np.asarray(SCHEMA), mean=mean, components=np.eye(288, dtype=np.float32)[:48], latent_scale=np.ones(48, np.float32))
    decoder = LowrankDecoder(URDF, 'cpu', basis)
    c = decoder.encode(future, current, obj)
    decoded = decoder.decode(c, current, obj, 1/30)['q'].numpy()[:, 1:]
    np.testing.assert_allclose(decoded[:, :, :3], future[:, :, :3], atol=2e-7)
    np.testing.assert_allclose(wrist_rotation(decoded), wrist_rotation(future), atol=5e-7)
    np.testing.assert_allclose(decoded[:, :, [6, 8, 10, 12, 14, 15]], future[:, :, [6, 8, 10, 12, 14, 15]], atol=2e-7)
