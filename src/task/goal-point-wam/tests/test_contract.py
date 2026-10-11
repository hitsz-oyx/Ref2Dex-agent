import sys
from pathlib import Path

import numpy as np
import pytest
import torch


TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

from goal_point_wam.data import (  # noqa: E402
    GoalWindows,
    _current_hand,
    goal_collate,
)
from goal_point_wam.model import ExecutedActionAdapter, GoalEncoder, WorldDecoder  # noqa: E402


def _sample(offset=0.0, missing_left=False):
    objects = 1
    object_points = np.zeros((objects, 512, 3), np.float32)
    object_points[0, :, 0] = np.linspace(0.1, 0.2, 512) + offset
    effect = np.tile(np.eye(4, dtype=np.float32), (objects, 24, 1, 1))
    effect[:, :, 0, 3] = np.linspace(0.0, .02, 24)
    hand_rows = []
    hand_features = []
    for hand in range(2):
        if hand == 1 and missing_left:
            continue
        for keypoint in range(11):
            hand_rows.append([offset, .01 * hand, .001 * keypoint])
            hand_features.append([0., 0., 1., float(hand), keypoint / 10., 0.])
    object_xyz = object_points[0]
    object_features = np.zeros((objects, 15), np.float32)
    object_features[:, 0] = offset
    hand_feature_array = np.concatenate((np.zeros((len(hand_features), 12), np.float32),
                                         np.asarray(hand_features, np.float32)), axis=1)
    object_feature_array = np.concatenate((
        np.zeros((len(object_xyz), 12), np.float32),
        np.tile([1., 0., 0., 0., 0., 0.], (len(object_xyz), 1)),
    ), axis=1)
    return dict(
        xyz=np.concatenate((object_xyz, np.asarray(hand_rows, np.float32))),
        features=np.concatenate((object_feature_array, hand_feature_array)),
        scene_object=np.concatenate((np.zeros(len(object_xyz), np.int64),
                                     -np.ones(len(hand_rows), np.int64))),
        action=np.zeros((24, 22, 9), np.float32),
        action_valid=np.zeros((24, 22), bool),
        points=object_points,
        radius=np.ones(objects, np.float32),
        object_features=object_features,
        effect=effect.transpose(0, 1, 2, 3),
        category=np.int64(0),
        sample_id=np.asarray([0, 0, 0], np.int64),
        hand_presence=np.asarray([not missing_left, not missing_left], bool),
    )


def test_current_hand_uses_semantic_metadata_and_mask():
    sample = _sample(missing_left=True)
    points, valid = _current_hand(sample)
    assert points.shape == (2, 11, 3)
    assert valid[0].all()
    assert not valid[1].any()
    assert np.all(points[1] == 0)


class _Base:
    meta = {"sources": [{"name": "oakink2"}, {"name": "grab"}, {"name": "arctic"}]}
    offsets = np.asarray([0, 1, 2, 3])

    def __len__(self):
        return 2

    def __getitem__(self, index):
        sample = _sample(offset=float(index), missing_left=index == 1)
        # Future positions are held in the first three action channels; the
        # second sample intentionally lacks one hand at every horizon.
        sample["action"][:, :11, :3] = sample["xyz"][-11:]
        sample["action"][:, 11:, :3] = sample["xyz"][-11:]
        sample["action_valid"][:, :11] = True
        if index == 0:
            sample["action_valid"][:, 11:] = True
        return sample


def test_goal_collate_preserves_masks_and_shapes():
    dataset = GoalWindows(_Base())
    batch = goal_collate([dataset[0], dataset[1]])
    assert batch["start_hand"].shape == (2, 2, 11, 3)
    assert batch["goal_hand"].shape == (2, 2, 11, 3)
    assert batch["trajectory_target"].shape == (2, 24, 2, 11, 3)
    assert batch["trajectory_valid"].shape == (2, 24, 2, 11)
    assert batch["trajectory_valid"][0].all()
    assert not batch["trajectory_valid"][1, :, 1].any()
    assert batch["scene_object"].shape == batch["point_valid"].shape


def test_goal_and_world_components_are_finite_with_missing_goal_points():
    stats = {
        "action_mean": [0.] * 9,
        "action_std": [1.] * 9,
        "flow_mean": [[0., 0., 0.]] * 24,
        "flow_std": [[1., 1., 1.]] * 24,
        "translation_mean": [[0., 0., 0.]] * 24,
        "translation_std": [[1., 1., 1.]] * 24,
        "rotation_scale": [1.] * 24,
    }
    goal = GoalEncoder()
    start = torch.zeros(2, 2, 11, 3)
    end = torch.ones_like(start)
    valid = torch.ones(2, 4, 11, dtype=torch.bool)
    valid[1, 1:] = False
    tokens, mask = goal(start, end, valid)
    assert tokens.shape == (2, 45, 128)
    assert mask[:, -1].all()
    assert torch.isfinite(tokens).all()

    action = ExecutedActionAdapter(stats)
    encoded, action_mask = action(torch.zeros(2, 24, 2, 11, 9), valid[:, :2].unsqueeze(1).expand(2, 24, 2, 11))
    assert encoded.shape == (2, 528, 128)
    assert action_mask.shape == (2, 528)
    world = WorldDecoder(stats)
    output = world(torch.zeros(2, 8, 128), torch.ones(2, 8, dtype=torch.bool),
                   torch.zeros(2, 1, 128), torch.ones(2, 1, dtype=torch.bool),
                   encoded, action_mask)
    assert output["translation"].shape == (2, 1, 24, 3)
    assert output["rotation"].shape == (2, 1, 24, 3, 3)
    assert torch.isfinite(output["translation"]).all()
    assert torch.isfinite(output["rotation"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full sparse scene isolation uses CUDA")
def test_full_model_world_isolation_smoke():
    # The GPU engineering entry point performs this audit against a real
    # three-source batch; this test is intentionally only an opt-in CUDA gate.
    pytest.skip("covered by train_three_domain.py GPU engineering audit")


def test_world_preserves_keypoint_assignment_and_masks_missing_actions():
    # The former per-time mean was invariant to this permutation. World memory
    # must retain semantic assignment, and invalid payloads must be irrelevant.
    torch.set_num_threads(2)
    torch.manual_seed(9)
    stats = dict(action_mean=[0.] * 9, action_std=[1.] * 9,
                 flow_mean=[[0.] * 3] * 24, flow_std=[[1.] * 3] * 24,
                 translation_mean=[[0.] * 3] * 24, translation_std=[[1.] * 3] * 24,
                 rotation_scale=[1.] * 24)
    adapter, world = ExecutedActionAdapter(stats), WorldDecoder(stats)
    torch.nn.init.normal_(world.head[-1].weight, std=.02)
    action = torch.randn(1, 24, 2, 11, 9)
    valid = torch.ones(1, 24, 2, 11, dtype=torch.bool)
    valid[:, :, 1] = False
    scene = torch.randn(1, 8, 128)
    scene_mask = torch.ones(1, 8, dtype=torch.bool)
    objects = torch.randn(1, 1, 128)
    object_mask = torch.ones(1, 1, dtype=torch.bool)
    def predict(payload):
        tokens, mask = adapter(payload, valid)
        return world(scene, scene_mask, objects, object_mask, tokens, mask)['translation']
    with torch.no_grad():
        original = predict(action)
        corrupted = action.clone()
        corrupted[:, :, 1] = float('nan')
        assert torch.equal(original, predict(corrupted))
        permuted = action.flip(3)
        assert (original - predict(permuted)).abs().max() > 1e-5


def test_pretrained_transfer_is_strict_and_leaves_new_projection_untouched(monkeypatch):
    from goal_point_wam import pretrained
    class Scene(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.scene_proj = torch.nn.Linear(2, 2)
            self.object_proj = torch.nn.Linear(2, 2)
            self.register_buffer('scene_mean', torch.zeros(2))
            self.register_buffer('scene_std', torch.ones(2))
    scene = Scene()
    monkeypatch.setattr(pretrained, 'file_digest', lambda path: 'vendor')
    config = dict(hidden=128, patch_size=128, fps=30, history=4, horizon=24,
                  voxel_m=.01, voxel_origin_m=[-2., -2., -2.])
    source = {}
    for key, value in scene.state_dict().items():
        if key.startswith('object_proj.'): continue
        name = 'backbone.core.' + key[len('backbone.'):] if key.startswith('backbone.') else key
        source[name] = value.clone() if key.startswith('scene_') and '.' not in key else torch.full_like(value, .3)
    state = dict(checkpoint_kind='pointworld-temporal.ddp.v1', config=config, step=50000,
                 identity=dict(stats_sha256='stats', git_commit='parent',
                               vendor_sources={'ptv3/ptv3.py': 'vendor', 'ptv3/ptv3_arch.yaml': 'vendor'}),
                 model=source)
    before = {key: value.clone() for key, value in scene.state_dict().items()}
    source['scene_std'] = torch.full((2,), 2.)
    with pytest.raises(ValueError, match='normalization buffer'): pretrained.transfer_scene(scene, state, config, 'stats')
    assert all(torch.equal(before[k], v) for k, v in scene.state_dict().items())
    source['scene_std'] = torch.ones(2)
    report = pretrained.transfer_scene(scene, state, config, 'stats')
    assert report['transferred_parameters'] == 12
    assert torch.equal(scene.backbone.weight, source['backbone.core.weight'])
    assert torch.equal(scene.object_proj.weight, before['object_proj.weight'])
    with pytest.raises(ValueError, match='normalization hash'): pretrained.transfer_scene(scene, state, config, 'wrong')
    with pytest.raises(ValueError, match='config mismatch'): pretrained.transfer_scene(scene, state, {**config, 'patch_size': 64}, 'stats')
