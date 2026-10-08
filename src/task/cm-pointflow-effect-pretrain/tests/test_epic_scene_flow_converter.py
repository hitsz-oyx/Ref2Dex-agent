"""Small deterministic contracts for the EPIC scene-flow conversion probe."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/convert_epic_scene_flow.py'
spec = importlib.util.spec_from_file_location('epic_scene_flow_converter', SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_fit_similarity_recovers_unordered_metric_transform():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(180, 3)) * np.array([.3, .2, .02])
    rotation = Rotation.from_euler('xyz', [20, -15, 35], degrees=True).as_matrix()
    scale = 1.7
    translation = np.array([.2, -.1, .8])
    target = scale * source @ rotation + translation
    # Reordering exercises the point-cloud (rather than vertex-correspondence)
    # path used for Contact mesh to depth-mask fitting.
    target = target[rng.permutation(len(target))]
    fit = converter.fit_similarity(source, target, iterations=10)
    assert fit is not None
    assert fit['rms'] < 1e-5
    np.testing.assert_allclose(fit['scale'], scale, rtol=1e-4, atol=1e-6)


def test_camera_world_roundtrip_uses_row_points():
    w2c = np.eye(4)
    w2c[:3, :3] = Rotation.from_euler('z', 25, degrees=True).as_matrix()
    w2c[:3, 3] = [.4, -.2, 1.1]
    points = np.array([[.1, .2, .7], [-.3, .5, 1.2]])
    np.testing.assert_allclose(
        converter.camera_to_world(converter.world_to_camera(points, w2c), w2c),
        points, atol=1e-7)
