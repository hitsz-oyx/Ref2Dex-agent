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
    object_features[:, 13] = 1.0
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
    assert batch["start_hand_history"].shape == (2, 2, 11, 9)
    assert batch["goal_hand"].shape == (2, 2, 11, 3)
    assert batch["trajectory_target"].shape == (2, 24, 2, 11, 3)
    assert batch["trajectory_valid"].shape == (2, 24, 2, 11)
    assert batch["trajectory_valid"][0].all()
    assert not batch["trajectory_valid"][1, :, 1].any()
    assert batch["scene_object"].shape == batch["point_valid"].shape


def test_observed_hand_history_is_semantic_current_only_and_missing_safe():
    from goal_point_wam.data import _current_hand_history
    sample = _sample(missing_left=True)
    rows = np.flatnonzero(sample['scene_object'] < 0)
    sample['features'][rows, 3:12] = np.arange(len(rows))[:, None] + np.arange(9)[None] / 10.
    expected = _current_hand_history(sample)
    np.testing.assert_allclose(expected[0, :, 0], np.arange(11))
    assert not expected[1].any()
    order = np.arange(len(sample['xyz']))[::-1]
    for key in ('xyz', 'features', 'scene_object'): sample[key] = sample[key][order].copy()
    sample['action'][:] = 9876.
    np.testing.assert_array_equal(_current_hand_history(sample), expected)
    handrow = np.flatnonzero(sample['scene_object'] < 0)[0]
    sample['features'][handrow, 3] = np.nan
    with pytest.raises(ValueError, match='observed hand history'):
        _current_hand_history(sample)


def test_object_goal_uses_target_flag_and_rigid_current_frame_without_hand_goal():
    from goal_point_wam.data import _object_goal
    sample = _sample()
    second = sample["points"] + np.array([.3, .1, 0.], np.float32)
    sample["points"] = np.concatenate((sample["points"], second))
    sample["object_features"] = np.zeros((2, 15), np.float32)
    sample["object_features"][1, 13] = 1.
    effect = np.tile(np.eye(4, dtype=np.float32), (2, 24, 1, 1))
    effect[1, -1, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    effect[1, -1, :3, 3] = [.04, -.02, .01]
    sample["effect"] = effect
    result = _object_goal(sample, 32)
    ids = np.linspace(0, 511, 32, dtype=np.int64)
    expected_start = second[0, ids]
    expected_goal = np.stack((-expected_start[:, 1], expected_start[:, 0],
                              expected_start[:, 2]), axis=-1) + [.04, -.02, .01]
    np.testing.assert_allclose(result["start_object_points"], expected_start)
    np.testing.assert_allclose(result["goal_object_points"], expected_goal, atol=1e-7)
    assert result["goal_object_slot"] == 1
    sample["action"][:] = 12345.
    np.testing.assert_array_equal(_object_goal(sample, 32)["goal_object_points"], result["goal_object_points"])
    sample["object_features"][0, 13] = 1.
    with pytest.raises(ValueError, match="exactly one"):
        _object_goal(sample, 32)
    batch = goal_collate([GoalWindows(_Base(), goal_type="object_points")[i] for i in range(2)])
    assert batch["goal_object_points"].shape == (2, 32, 3)
    assert "goal_hand" not in batch and "goal_hand_valid" not in batch
    assert not batch["trajectory_valid"][1, :, 1].any()


def test_object_goal_is_a_masked_point_set_and_coordinate_gradients_exist():
    from goal_point_wam.model import ObjectGoalEncoder, TrajectoryDecoder
    torch.set_num_threads(2)
    torch.manual_seed(5)
    goal, trajectory = ObjectGoalEncoder(), TrajectoryDecoder()
    torch.nn.init.normal_(trajectory.head[-1].weight, std=.02)
    start, end = torch.randn(2, 7, 3), torch.randn(2, 7, 3, requires_grad=True)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1] = False
    def predict(a, b, mask):
        tokens, token_mask = goal(a, b, mask)
        return trajectory(torch.zeros(2, 3, 128), torch.ones(2, 3, dtype=torch.bool), tokens, token_mask)
    original = predict(start, end, valid)
    gradient, = torch.autograd.grad(original[0].square().sum(), end)
    assert torch.isfinite(gradient).all() and gradient[0].abs().max() > 0
    assert gradient[1].abs().max() == 0
    with torch.no_grad():
        order = torch.tensor([3, 0, 6, 2, 1, 5, 4])
        torch.testing.assert_close(original, predict(start[:, order], end[:, order], valid[:, order]), atol=2e-6, rtol=2e-5)
        invalid_end, invalid_start = end.detach().clone(), start.clone()
        invalid_end[1] = invalid_start[1] = float('nan')
        torch.testing.assert_close(original, predict(invalid_start, invalid_end, valid), atol=0, rtol=0)


def test_object_goal_scale_is_a_unit_conversion_with_identical_weights():
    from goal_point_wam.model import ObjectGoalEncoder
    torch.set_num_threads(2)
    raw = ObjectGoalEncoder(coordinate_scale_m=1.).eval()
    scaled = ObjectGoalEncoder(coordinate_scale_m=.05).eval()
    scaled.load_state_dict(raw.state_dict(), strict=True)
    start, end = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        a, am = raw(start / .05, end / .05, valid)
        b, bm = scaled(start, end, valid)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(am, bm)
    for bad in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite and positive'):
            ObjectGoalEncoder(coordinate_scale_m=bad)


def test_relative_object_goal_uses_paired_displacement_and_masks_nan():
    from goal_point_wam.model import ObjectGoalEncoder
    torch.set_num_threads(2)
    relative = ObjectGoalEncoder(coordinate_scale_m=.05, encoding='start_delta').eval()
    absolute = ObjectGoalEncoder(coordinate_scale_m=.05).eval()
    absolute.load_state_dict(relative.state_dict(), strict=True)
    start, end = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool); valid[1] = False
    start[1] = end[1] = float('nan')
    with torch.no_grad():
        a, am = relative(start, end, valid)
        b, bm = absolute(start, end - start, valid)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(am, bm) and torch.isfinite(a).all()


def test_controlled_teacher_preserves_world_and_masks_artificial_labels():
    sys.path.insert(0, str(TASK / 'tools/audit'))
    from probe_controlled_goal import controlled_batch
    batch = goal_collate([GoalWindows(_Base(), goal_type='object_points')[i] for i in range(2)])
    shift = torch.tensor([[.1, -.02, .03], [-.05, .04, -.01]])
    changed = controlled_batch(batch, shift)
    torch.testing.assert_close(changed['goal_object_points'] - batch['goal_object_points'],
                               shift[:, None].expand(-1, 32, -1), atol=1e-7, rtol=1e-6)
    valid = batch['trajectory_valid'][:, -1]
    expected = shift[:, None, None].expand(-1, 2, 11, -1)
    torch.testing.assert_close((changed['trajectory_target'] - batch['trajectory_target'])[:, -1][valid], expected[valid])
    assert torch.count_nonzero(changed['trajectory_target'][~batch['trajectory_valid']]) == 0
    for key in ('action', 'effect', 'start_object_points', 'start_hand', 'features'):
        assert changed[key] is batch[key]
    torch.testing.assert_close(controlled_batch(batch, torch.zeros_like(shift))['trajectory_target'], batch['trajectory_target'])


def test_geometric_reference_recovers_goal_rigid_motion_without_future_hands():
    sys.path.insert(0, str(TASK / 'tools/audit'))
    from evaluate_object_goal import geometric_baselines
    torch.manual_seed(13)
    points = torch.randn(2, 8, 3)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = torch.tensor([.02, -.03, .04])
    hand = torch.randn(2, 2, 11, 3)
    goal = points @ rotation.T + shift
    valid = torch.ones(2, 8, dtype=torch.bool); valid[:, -1] = False
    points[:, -1] = goal[:, -1] = float('nan')
    batch = dict(start_object_points=points, goal_object_points=goal, object_goal_valid=valid, start_hand=hand)
    predicted = geometric_baselines(batch)
    endpoint = hand @ rotation.T + shift - hand
    torch.testing.assert_close(predicted['object_rigid_follow'][:, -1], endpoint, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(predicted['object_rigid_follow'][:, 0], endpoint / 24, atol=2e-6, rtol=2e-6)
    assert all(torch.isfinite(value).all() for value in predicted.values())


def test_compact_inputs_are_observed_only_and_goal_rigid_features_are_correct():
    sys.path.insert(0, str(TASK / 'tools/audit'))
    from probe_compact_forecast import compact_features, CompactForecast
    from goal_point_wam.data import _current_hand_history
    sample = _sample(missing_left=True)
    start, valid = _current_hand(sample)
    sample.update(start_hand=start, start_hand_valid=valid, start_hand_history=_current_hand_history(sample))
    rng = np.random.default_rng(14)
    points = rng.normal(size=(8, 3)).astype(np.float32)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.float32)
    sample.update(start_object_points=points, goal_object_points=points @ rotation.T + [.02, -.03, .04],
                  object_goal_valid=np.ones(8, bool))
    stats = dict(scene_mean=[0.] * 18, scene_std=[1.] * 18)
    hand, geometry, goal = compact_features(sample, stats)
    np.testing.assert_allclose(goal[3:], (rotation - np.eye(3))[:, :2].T.reshape(-1), atol=2e-6)
    assert not hand[1].any()
    sample['action'][:] = 12345.
    for a, b in zip((hand, geometry, goal), compact_features(sample, stats)):
        np.testing.assert_array_equal(a, b)
    batch = dict(hand=torch.from_numpy(hand)[None], geometry=torch.from_numpy(geometry)[None],
                 goal=torch.from_numpy(goal)[None].requires_grad_())
    model = CompactForecast()
    torch.testing.assert_close(model(batch, True), model(batch, False), atol=0, rtol=0)
    torch.nn.init.normal_(model.head[-1].weight, std=.01)
    prediction = model(batch, True)
    gradient, = torch.autograd.grad(prediction.sum(), batch['goal'])
    assert prediction.shape == (1, 24, 2, 11, 3) and torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_goal_injection_routes_match_weights_and_remove_encoded_null_leakage():
    from goal_point_wam.model import TrajectoryDecoder
    torch.set_num_threads(2)
    arms = {}
    for route in ('memory', 'query', 'none'):
        torch.manual_seed(27)
        arms[route] = TrajectoryDecoder(layers=1, goal_injection=route, goal_query_adapter=True).eval()
        # Zero output initialization otherwise hides the conditional path.
        torch.nn.init.normal_(arms[route].head[-1].weight, std=.01)
    state = arms['memory'].state_dict()
    assert all(torch.equal(state[k], arm.state_dict()[k]) for arm in arms.values() for k in state)
    scene = torch.randn(1, 4, 128)
    scene_mask = torch.ones(1, 4, dtype=torch.bool)
    goal = torch.randn(1, 9, 128, requires_grad=True)
    valid = torch.ones(1, 9, dtype=torch.bool)
    altered = goal.detach().clone()
    altered[:, 4:] += torch.randn_like(altered[:, 4:])  # Includes goal-encoded null.
    for route, arm in arms.items():
        pred = arm(scene, scene_mask, goal, valid)
        other = arm(scene, scene_mask, altered, valid)
        gradient, = torch.autograd.grad(pred.square().sum(), goal, allow_unused=True)
        if route == 'none':
            assert torch.equal(pred, other) and gradient is None
        else:
            assert (pred - other).abs().max() > 1e-5
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
        if route == 'query':
            assert torch.count_nonzero(gradient[:, -1]) == 0
    valid[:, 1:4] = False
    poisoned = goal.detach().clone(); poisoned[:, 1:4] = float('nan')
    query = arms['query']
    torch.testing.assert_close(query(scene, scene_mask, poisoned, valid),
                               query(scene, scene_mask, goal, valid), atol=0, rtol=0)
    valid[:, :-1] = False
    assert torch.count_nonzero(query._query_condition(poisoned, valid)) == 0
    with pytest.raises(ValueError, match='requires'):
        TrajectoryDecoder(goal_injection='query')


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
