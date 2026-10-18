import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from wam_rl_policy.wam_v4_runtime import (
    V4_SCHEMA, apply_dense_overlay, collate_v4, load_geometry,
    v4_online_samples,
)
from pose_mesh_data.v4_windows import prepare_v4_geometry


def _geometry(offset=0.):
    # 512 distinct surface points plus the 32 original flow identities.
    surface = np.stack((np.linspace(-.1, .1, 512), np.zeros(512),
                        np.full(512, offset)), -1).astype(np.float32)
    normals = np.tile([0., 1., 0.], (512, 1)).astype(np.float32)
    query = surface[np.linspace(0, 511, 32, dtype=np.int64)].copy()
    return dict(surface=surface, normals=normals, query=query)


def _poses(batch=1):
    rng = np.random.default_rng(7)
    out = np.zeros((batch, 4, 7), np.float64)
    out[..., :3] = rng.normal(0, .02, (batch, 4, 3))
    out[..., 3:] = Rotation.random(batch * 4, random_state=rng).as_quat().reshape(batch, 4, 4)
    return out


def test_online_v4_batch_has_explicit_512_32_128_contract():
    geoms = [prepare_v4_geometry(_geometry()) for _ in range(2)]
    objects = _poses(2)
    hands = np.zeros((2, 4, 11, 3), np.float32)
    hands[..., 0] = .01
    goal_world = objects[:, -1, :3] + .03
    samples = v4_online_samples(objects, hands, geoms, goal_world=goal_world)
    assert len(samples) == 2
    for sample in samples:
        assert sample['xyz'].shape == (512, 3)
        assert sample['features'].shape == (512, 18)
        assert sample['scene_points'].shape == (1, 32, 3)
        assert sample['interaction_points'].shape == (1, 128, 3)
        assert sample['interaction_normals'].shape == (1, 128, 3)
        assert sample['interaction_query_valid'].shape == (1, 128)
        assert sample['goal_object_delta'].shape == (32, 3)
        assert sample['start_hand'].shape == (2, 11, 3)
        assert sample['start_hand_history'].shape == (2, 11, 9)
        assert sample['start_hand_valid'][0].all() and not sample['start_hand_valid'][1].any()
        assert np.all(sample['features'][:, 12] == 1.)
        np.testing.assert_array_equal(
            sample['features'][:, 13],
            np.linalg.norm(sample['features'][:, 9:12], axis=-1) < .002)
    batch = collate_v4(samples)
    assert batch['xyz'].shape == (2, 512, 3)
    assert batch['interaction_query_valid'].shape == (2, 1, 128)


def test_online_goal_delta_is_preserved_not_recomputed_from_world_position():
    geoms = [prepare_v4_geometry(_geometry()) for _ in range(1)]
    goal = np.zeros((1, 32, 3), np.float32)
    goal[0, :, 0] = np.linspace(-.02, .02, 32)
    samples = v4_online_samples(_poses(1), np.zeros((1, 4, 11, 3)), geoms, goal_delta=goal)
    np.testing.assert_allclose(samples[0]['goal_object_delta'], goal[0], atol=1e-6)


def test_geometry_cache_and_v1_dense_overlay_are_bound_to_identity(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    np.savez(cache / 'object-000.npz', **_geometry())
    (cache / 'manifest.json').write_text(json.dumps(dict(records=[dict(
        object_code='core/bottle', scale=.08, geometry=str(cache / 'object-000.npz'))])))
    dense = tmp_path / 'dense'
    dense.mkdir()
    replacement = _geometry(offset=1.)
    np.savez(dense / 'object.npz', **replacement)
    (dense / 'surface-manifest.json').write_text(json.dumps(dict(
        schema='ref2dex.wam-v4-dense-surface.v1', status='COMPLETED',
        objects=[dict(key='core/bottle@0.08', path='object.npz')])))
    prepared = load_geometry(cache, [dict(object_code='core/bottle', scale=.08)],
                             dense / 'surface-manifest.json')
    assert len(prepared) == 1
    surface = prepared[0][0]
    np.testing.assert_allclose(surface[:, 2], 1., atol=1e-6)
    with np.testing.assert_raises(ValueError):
        load_geometry(cache, [dict(object_code='core/bottle', scale=.08)],
                      dense / 'surface-manifest.json', dense_sha256='0' * 64)


def test_apply_dense_overlay_rejects_unknown_payload_identity(tmp_path):
    dense = tmp_path / 'dense'
    dense.mkdir()
    np.savez(dense / 'object.npz', surface=np.zeros((512, 3), np.float32),
             normals=np.tile([0., 1., 0.], (512, 1)).astype(np.float32))
    (dense / 'surface-manifest.json').write_text(json.dumps(dict(
        schema='ref2dex.wam-v4-dense-surface.v1', status='COMPLETED',
        objects=[dict(key='core/bottle@0.08', path='object.npz')])))
    with np.testing.assert_raises(ValueError):
        apply_dense_overlay({'core/bottle@0.08': _geometry()}, dense / 'surface-manifest.json')


class FakeModel(torch.nn.Module):
    def __init__(self, stats, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch, return_shared=False):
        b = batch['xyz'].shape[0]
        pred = dict(
            trajectory=torch.zeros(b, 24, 2, 11, 3),
            scene=torch.zeros(b, 1, 24, 32, 3),
            interaction=torch.zeros(b, 1, 24, 128, 7),
            shared_tokens=torch.zeros(b, 34, 768),
            shared_valid=torch.ones(b, 34, dtype=torch.bool),
            scene_token_count=4,
        )
        return pred

    def encode(self, batch):
        b = batch['xyz'].shape[0]
        scene = dict(tokens=torch.zeros(b, 4, 768))
        return torch.zeros(b, 34, 768), torch.ones(b, 34, dtype=torch.bool), scene


def test_v4_runtime_loads_checkpoint_and_exposes_tau_c_and_summary(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    np.savez(cache / 'object-000.npz', **_geometry())
    (cache / 'manifest.json').write_text(json.dumps(dict(records=[dict(
        object_code='core/bottle', scale=.08, geometry=str(cache / 'object-000.npz'))])))
    checkpoint = tmp_path / 'v4.pt'
    config = dict(schema=V4_SCHEMA)
    stats = {'own': 'v4'}
    torch.save(dict(schema=V4_SCHEMA, config=config, stats=stats, step=7,
                    model={'weight': torch.tensor([2.])}), checkpoint)
    from wam_rl_policy.wam_v4_runtime import V4Runtime
    runtime = V4Runtime(checkpoint, [dict(object_code='core/bottle', scale=.08)], cache,
                        device='cpu', model_factory=FakeModel)
    batch = runtime.batch(_poses(1), np.zeros((1, 4, 11, 3)),
                          goal_world=_poses(1)[:, -1, :3] + .02)
    pred = runtime.predict(batch)
    assert pred['trajectory'].shape == (1, 24, 2, 11, 3)
    assert pred['interaction'].shape == (1, 1, 24, 128, 7)
    summary = runtime.summary(batch)
    assert summary.shape == (1, 2304)


def test_v4_runtime_rejects_non_v4_checkpoint(tmp_path):
    checkpoint = tmp_path / 'v3.pt'
    torch.save(dict(schema='ref2dex.ptv3-shared-wam.v3', config={}, stats={}, model={}), checkpoint)
    from wam_rl_policy.wam_v4_runtime import V4Runtime
    with np.testing.assert_raises(ValueError):
        V4Runtime(checkpoint, [], tmp_path, device='cpu', model_factory=FakeModel)
