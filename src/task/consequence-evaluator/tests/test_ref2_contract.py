"""Decision-known plans and physical futures; tiny CPU engineering fixtures."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_consequence_evaluator import fixture, change_arrays
from test_collection import root_state
from test_supervision import packet, collection_fixture
from consequence_evaluator.collection import Episode, Perturbations
from consequence_evaluator.contracts import ARMS, FUTURE_DIM, K, future_mode
from consequence_evaluator.data import Windows, interaction_future, sha
from consequence_evaluator.model import Evaluator
from consequence_evaluator.supervision import expert_anchor, local_preferences, physical_trace
from label_continuous import label
from prepare_windows import prepare


def test_requested_plan_precedes_execution_and_survives_clipping():
    controller = Perturbations(6, 7)
    controller.chunks[1] = .1
    before, known = controller.known_plan(0)
    assert known.tolist() == [True, False, False, False, False, False]
    assert not before.any()
    base = np.full((6, 18), .99)
    action, _, audit = controller.apply(base, np.zeros(6), np.zeros(6, bool),
        np.zeros(6), 1, np.full(6, 100), np.ones(6, bool))
    planned, known = controller.known_plan(1)
    assert known[1] and np.allclose(planned[1], .1)
    assert np.allclose(action[1] - base[1], .01)
    assert not np.allclose(planned[1, 0], audit['actual_residual'][1])
    snapshot = planned.copy()
    controller.apply(np.zeros_like(base), np.zeros(6), np.zeros(6, bool),
        np.zeros(6), 2, np.full(6, 99), np.ones(6, bool))
    assert np.array_equal(planned, snapshot)
    following, _ = controller.known_plan(2)
    assert np.array_equal(following[1, :-1], planned[1, 1:])
    assert not following[1, -1].any()


def test_window_action_is_requested_plan_never_executed_control(tmp_path):
    source = tmp_path / 'source'
    def mutation(a, split, quality):
        a['action'][:] = .9
        a['residual_plan'][:] = .08
    annotations = fixture(source, mutation)
    out = tmp_path / 'windows'
    prepare(source, annotations, out)
    data = Windows(out)
    assert np.allclose(data.arrays['action'], .08)
    assert data.batch([0], 'cpu')[0]['future'].shape == (1, K, FUTURE_DIM)


@pytest.mark.parametrize('problem', ['old_schema', 'old_actions', 'unknown_plan'])
def test_old_or_decision_unknown_data_is_rejected(tmp_path, problem):
    source = tmp_path / 'source'
    annotations = fixture(source)
    path = source / 'manifest.json'
    m = json.loads(path.read_text())
    if problem == 'old_schema':
        m['schema'] = 'ref2dex.consequence-evaluator.episodes.v1'
    elif problem == 'old_actions':
        m['action_semantics'] = 'executed_reactive_native_control'
    else:
        record = m['episodes'][0]
        data = source / record['path']
        with np.load(data) as p:
            a = {k: p[k].copy() for k in p.files}
        a['plan_known'][0] = False
        np.savez_compressed(data, **a)
        record['sha256'] = sha(data)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        prepare(source, annotations, tmp_path / 'out')


def test_three_arms_equal_capacity_and_future_masks_after_normalization():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    initial = Evaluator(6, width=8, layers=1)
    models = {arm: copy.deepcopy(initial) for arm in ARMS}
    assert len({sum(p.numel() for p in m.parameters()) for m in models.values()}) == 1
    h, a, f = torch.randn(2, 6), torch.randn(2, K, 18), torch.randn(2, K, FUTURE_DIM)
    changed = f.clone()
    changed[..., 12:] += 20
    for arm in ARMS:
        model = models[arm]
        original = model(h, a, f, future_mode(arm))['score']
        new = model(h, a, changed, future_mode(arm))['score']
        if arm == 'oracle_interaction':
            assert not torch.allclose(original, new)
        else:
            assert torch.equal(original, new)
    assert torch.equal(initial(h, a, f, False)['score'], initial(h, a, f + 100, False)['score'])
    assert not torch.allclose(initial(h, a, f, 'object')['score'], initial(h, a, f + 1, 'object')['score'])


def test_interaction_future_is_relative_to_each_future_object_and_world_invariant():
    poses = np.broadcast_to(np.eye(4), (K+1, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(K+1) * .01
    local = np.random.default_rng(7).normal(size=(11, 3)) * .01
    hand = poses[:, None, :3, 3] + local
    expected = interaction_future(poses, hand)
    assert np.allclose(expected, local)
    world = np.eye(4)
    world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world[:3, 3] = [1, 2, 3]
    transformed = hand @ world[:3, :3].T + world[:3, 3]
    assert np.allclose(expected, interaction_future(world @ poses, transformed), atol=1e-6)


def test_force_proxy_without_near_hand_geometry_cannot_anchor_progress():
    arrays, audit = packet()
    audit['surface_gap'][:] = .05
    trace = physical_trace(arrays, audit)
    assert trace['force_proxy'][1:].all() and not trace['held'].any()
    assert expert_anchor(trace, dict(assigned_phase='clean', perturbation_tick=-1))[0] == 'failure'
    del audit['surface_gap']
    with pytest.raises(ValueError, match='surface gap'):
        physical_trace(arrays, audit)


@pytest.mark.parametrize('problem', ['expert', 'motion', 'object', 'hand'])
def test_preference_pairs_reject_current_state_and_controller_confounds(tmp_path, problem):
    source, out = tmp_path/'source', tmp_path/'out'
    annotations = fixture(source)
    prepare(source, annotations, out)
    def mutate(a):
        if problem in ('expert', 'motion'):
            a[problem][2] = 'changed'
        elif problem == 'object':
            a['current_object'][2, 0, 3] += .031
        else:
            a['current_hand'][2] += .021
    change_arrays(out, mutate)
    with pytest.raises(ValueError):
        Windows(out)


def test_long_native_episode_is_not_truncated_at_old600_step_cap():
    points = {'hand_keypoints': np.zeros((11, 3))}
    ep = Episode(np.zeros(6), root_state(), False, max_steps=1200, kinematics=points)
    for tick in range(650):
        ep.append(np.zeros(18), 'approach', np.zeros(6), root_state(), False,
                  np.zeros(18), np.zeros(18, bool), tick == 649, points,
                  plan=np.zeros((K, 18)), plan_known=True)
    assert ep.arrays()['history'].shape == (651, 6)
    assert ep.arrays()['residual_plan'].shape == (650, K, 18)


def test_geometry_groups_actual_objects_and_scales_around_root():
    from consequence_evaluator.physical_geometry import PhysicalGeometry
    class Surface:
        def __init__(self, offset): self.offset = offset
        def hand(self, links):
            return links[:, :1, :3, 3], None
        def object(self, pose):
            return pose[:, None, :3, 3] + self.offset, None
    geometry = PhysicalGeometry.__new__(PhysicalGeometry)
    geometry.query_ids, geometry.key_ids = [0], list(range(11))
    geometry.surfaces = dict(airplane=Surface(.1), mug=Surface(.4))
    bodies = torch.zeros(2, 11, 13)
    bodies[..., 6] = 1
    bodies[..., :3] = 1
    state = torch.zeros(2, 13)
    state[:, 6] = 1
    state[:, :3] = 1
    task = SimpleNamespace(num_envs=2, device='cpu', _rigid_body_state=bodies,
        _target_states=state, object_id=torch.tensor([1, 0]), data_id=torch.tensor([0, 1]),
        object_name=['airplane', 'mug'], ball_size=2.)
    points, gap = geometry.measure(task)
    assert torch.equal(points, bodies[..., :3])
    assert torch.allclose(gap, torch.tensor([.8, .2]) * np.sqrt(3), atol=1e-6)
