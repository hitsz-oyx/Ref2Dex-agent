import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
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
    replacement['query'] = _geometry()['query']
    replacement['surface'][:32] = replacement['query']
    np.savez(dense / 'object.npz', **replacement)
    (dense / 'surface-manifest.json').write_text(json.dumps(dict(
        schema='ref2dex.wam-v4-dense-surface.v1', status='COMPLETED',
        objects=[dict(key='core/bottle@0.08', path='object.npz')])))
    prepared = load_geometry(cache, [dict(object_code='core/bottle', scale=.08)],
                             dense / 'surface-manifest.json')
    assert len(prepared) == 1
    surface = prepared[0][0]
    np.testing.assert_allclose(surface[32:, 2], 1., atol=1e-6)
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


def test_runtime_cache_requires_explicit_original_flow_queries(tmp_path):
    geometry = _geometry()
    del geometry['query']
    np.savez(tmp_path / 'object.npz', **geometry)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(records=[dict(
        object_code='core/bottle', scale=.08, geometry=str(tmp_path / 'object.npz'))])))
    with pytest.raises(ValueError, match='original 32 flow queries'):
        load_geometry(tmp_path, [dict(object_code='core/bottle', scale=.08)])


@pytest.mark.parametrize('invalid', ['changed_query', 'missing_query', 'missing_original'])
def test_dense_overlay_preserves_original_flow_queries(tmp_path, invalid):
    original, replacement = _geometry(), _geometry()
    if invalid == 'changed_query':
        replacement['query'] += 1.
    elif invalid == 'missing_query':
        del replacement['query']
    else:
        del original['query']
    payload = tmp_path / 'object.npz'
    np.savez(payload, **replacement)
    manifest = tmp_path / 'surface-manifest.json'
    manifest.write_text(json.dumps(dict(
        schema='ref2dex.wam-v4-dense-surface.v2', status='COMPLETED',
        objects=[dict(key='core/bottle@0.08', path=payload.name,
                      sha256=hashlib.sha256(payload.read_bytes()).hexdigest())])))
    with pytest.raises(ValueError, match='flow query'):
        apply_dense_overlay({'core/bottle@0.08': original}, manifest,
                            hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_v4_runtime_enforces_checkpoint_dense_geometry_binding(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    np.savez(cache / 'object.npz', **_geometry())
    (cache / 'manifest.json').write_text(json.dumps(dict(records=[dict(
        object_code='core/bottle', scale=.08, geometry=str(cache / 'object.npz'))])))
    payload = tmp_path / 'dense.npz'
    dense = _geometry(offset=.02)
    dense['query'] = _geometry()['query']
    dense['surface'][:32] = dense['query']
    np.savez(payload, **dense)
    manifest = tmp_path / 'surface-manifest.json'
    manifest.write_text(json.dumps(dict(
        schema='ref2dex.wam-v4-dense-surface.v2', status='COMPLETED',
        objects=[dict(key='core/bottle@0.08', path=payload.name,
                      sha256=hashlib.sha256(payload.read_bytes()).hexdigest())])))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint = tmp_path / 'v4.pt'
    torch.save(dict(schema=V4_SCHEMA, config={}, stats={'recipe': dict(
        surface_geometry_manifest=str(manifest), surface_geometry_sha256=digest)},
        model={'weight': torch.tensor([2.])}), checkpoint)
    from wam_rl_policy.wam_v4_runtime import V4Runtime
    # An omitted CLI override still uses the checkpoint's bound overlay.
    runtime = V4Runtime(checkpoint, [dict(object_code='core/bottle', scale=.08)], cache,
                        device='cpu', model_factory=FakeModel)
    np.testing.assert_array_equal(runtime.geometry[0][0], dense['surface'])
    relocated = tmp_path / 'relocated-manifest.json'
    relocated.write_bytes(manifest.read_bytes())
    runtime = V4Runtime(checkpoint, [dict(object_code='core/bottle', scale=.08)], cache,
                        dense_manifest=relocated, device='cpu', model_factory=FakeModel)
    np.testing.assert_array_equal(runtime.geometry[0][0], dense['surface'])
    # A different valid manifest cannot override the checkpoint identity.
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest SHA-256 mismatch'):
        V4Runtime(checkpoint, [], cache, dense_manifest=manifest,
                  device='cpu', model_factory=FakeModel)
    with pytest.raises(ValueError, match='checkpoint.*dense'):
        V4Runtime(checkpoint, [], cache, dense_manifest=manifest,
                  dense_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  device='cpu', model_factory=FakeModel)


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
    # The fake forward has no interaction_prediction_valid; add it to test the
    # explicit executor-facing accessor without a real model.
    pred['interaction_prediction_valid'] = torch.ones(1, 1, 24, 128, dtype=torch.bool)
    interaction, valid = runtime.interaction_for_executor(pred)
    assert interaction.shape == (1, 24, 128, 7) and valid.shape == (1, 24, 128)
    # v4 tau -> existing student clock/frame is part of the runtime contract.
    const = torch.zeros(1, 24, 2, 11, 3)
    const[0, :, 0, :, 0] = (torch.arange(1, 25) * .12 / 30)[:, None]
    tau = runtime.tau_for_student({'trajectory': const}, np.eye(4), frame='world')
    assert tau.shape == (8, 11, 3)
    np.testing.assert_allclose(tau[-1, :, 0], .12 * 8 / 60, atol=1e-6)


def test_v4_runtime_rejects_non_v4_checkpoint(tmp_path):
    checkpoint = tmp_path / 'v3.pt'
    torch.save(dict(schema='ref2dex.ptv3-shared-wam.v3', config={}, stats={}, model={}), checkpoint)
    from wam_rl_policy.wam_v4_runtime import V4Runtime
    with np.testing.assert_raises(ValueError):
        V4Runtime(checkpoint, [], tmp_path, device='cpu', model_factory=FakeModel)
