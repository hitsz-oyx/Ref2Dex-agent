import copy

import numpy as np
import pytest

from wam_rl_policy_v3.goal_channel import (
    ResetWaypointChannel, bind_source, digest, instruction_from_displacement,
    validate_instruction, validate_source,
)


def inputs():
    reset = dict(execution_id='new-execution', reset_id='reset-0', clock_id='collector-monotonic', at_ns=1000000000)
    def source(payload, role, kind='declared_task'):
        return bind_source(payload, source_role=role, source_kind=kind, source_id=role,
                           reset=reset, observed_at_ns=reset['at_ns'], available_at_ns=reset['at_ns'])
    pose = np.eye(4)
    query = np.random.default_rng(77).uniform(-.02, .02, (32, 3))
    anchor = source(pose, 'reset_observation', 'reset_measurement')
    rule = source(dict(rule='fixed_world_translation', translation_m=[0., 0., .2],
                       target_stage='task_completion'), 'task_instruction')
    query = source(query, 'reset_geometry')
    surface = source(np.zeros((16, 3)), 'reset_geometry')
    artifact = instruction_from_displacement(anchor, rule, query, surface, reset=reset,
        object_slot=0, hand_slot=0, goal_type='position_only')
    return artifact, reset, pose


def test_current_object_pose_is_legal_reset_input():
    artifact, reset, _ = inputs()
    validate_source(artifact['sources']['anchor'], reset, role='reset_observation')
    validate_instruction(artifact)


def test_explicit_current_pose_goal_is_valid_and_not_unknown():
    from ptv3_shared_wam_v6.goal_policy import UNKNOWN_GOAL_POLICY, apply_goal_input_policy
    artifact, reset, pose = inputs()
    sources = artifact['sources']
    rule = sources['rule'].copy()
    rule['payload'] = dict(rule='fixed_world_translation', translation_m=[0., 0., 0.],
                           target_stage='task_completion')
    rule['source_sha256'] = digest(rule['payload'])
    hold = instruction_from_displacement(sources['anchor'], rule, sources['query'], sources['surface'],
        reset=reset, object_slot=0, hand_slot=0, goal_type='position_only')
    channel = ResetWaypointChannel(hold, expected_reset=reset)
    tensors, _ = channel.update(pose, np.zeros((11, 3)), tick=0,
        observed_at_ns=reset['at_ns'], available_at_ns=reset['at_ns'], decision_at_ns=reset['at_ns'])
    assert not tensors['goal_object_delta'].any() and tensors['object_goal_valid'].all()
    assert tensors['goal_object_points'].any()
    assert not apply_goal_input_policy(tensors, UNKNOWN_GOAL_POLICY)['object_goal_valid'].any()


@pytest.mark.parametrize('attack', ['future_role', 'future_time', 'self_recording', 'observation_as_intent'])
def test_future_realized_pose_and_observation_as_intent_are_rejected(attack):
    artifact, reset, _ = inputs()
    source = copy.deepcopy(artifact['sources']['anchor'])
    if attack == 'future_role':
        source['source_role'] = 'future_realized_pose'
    elif attack == 'future_time':
        # Even a falsely retained legal role and True availability cannot
        # admit a t+24 acquisition timestamp.
        source['observed_at_ns'] += 800000000
        source['available_at_ns'] += 800000000
    elif attack == 'self_recording':
        source.update(source_kind='independent_recording', source_execution_id=reset['execution_id'])
    else:
        source['source_role'] = 'task_instruction'
    with pytest.raises(ValueError):
        validate_source(source, reset)


def test_future_path_substitution_is_rejected_even_with_recomputed_hash():
    artifact, _, _ = inputs()
    artifact['reference_poses'][1][0][3] = 123.
    artifact['instruction_sha256'] = digest({k: v for k, v in artifact.items() if k != 'instruction_sha256'})
    with pytest.raises(ValueError, match='not derived'):
        validate_instruction(artifact)


@pytest.mark.parametrize('rotated', [False, True])
def test_training_and_deployment_use_same_producer_and_match_online_tensor_boundary(rotated):
    from ptv3_shared_wam_v6.online_runtime import online
    from ptv3_shared_wam_v5.data import collate
    import torch
    artifact, reset, pose = inputs()
    if rotated:
        from scipy.spatial.transform import Rotation
        pose[:3, :3] = Rotation.from_rotvec([.2, .3, -.4]).as_matrix()
        pose[:3, 3] = [.11, -.21, .32]
    train = ResetWaypointChannel(artifact, expected_reset=reset)
    deploy = ResetWaypointChannel(artifact, expected_reset=reset)
    for tick in range(5):
        now = reset['at_ns'] + round(tick * 1e9 / 30)
        args = dict(tick=tick, observed_at_ns=now, available_at_ns=now, decision_at_ns=now)
        tensors, receipt = train.update(pose, np.zeros((11, 3)), **args)
        actual, online_receipt = deploy.update(pose, np.zeros((11, 3)), **args)
        assert receipt == online_receipt
        for key in tensors:
            assert tensors[key].dtype == actual[key].dtype
            assert tensors[key].tobytes() == actual[key].tobytes()
        geo = dict(query=train.scheduler.points, surface=np.zeros((2048, 3)), normals=np.zeros((2048, 3)))
        sample = online(geo, np.tile(pose, (4, 1, 1)), np.zeros((4, 2, 21, 3)),
            np.zeros((4, 2, 778, 3)), np.ones((2, 21), bool), np.zeros((2, 778), bool),
            np.zeros((2, 778), np.int64), receipt['goal_pose'], goal_semantics=receipt['goal_semantics'])
        for key in tensors:
            assert torch.equal(collate([tensors])[key], collate([{k: sample[k] for k in tensors}])[key])
        assert receipt['reference_endpoint_time_s'] is None


def test_cross_reset_missing_history_future_observation_and_modified_content_rejected():
    artifact, reset, pose = inputs()
    with pytest.raises(ValueError, match='match target'):
        ResetWaypointChannel(artifact, expected_reset=dict(reset, reset_id='other'))
    channel = ResetWaypointChannel(artifact, expected_reset=reset)
    with pytest.raises(ValueError, match='continuous'):
        channel.update(pose, np.zeros((11, 3)), tick=3, observed_at_ns=reset['at_ns'],
                       available_at_ns=reset['at_ns'], decision_at_ns=reset['at_ns'])
    with pytest.raises(ValueError, match='unavailable'):
        channel.update(pose, np.zeros((11, 3)), tick=0, observed_at_ns=reset['at_ns']+1,
                       available_at_ns=reset['at_ns']+1, decision_at_ns=reset['at_ns'])
    artifact['sources']['anchor']['payload'][0][3] = 1.
    with pytest.raises(ValueError, match='source hash'):
        validate_instruction(artifact)


def test_artifacts_preserve_sources_and_replay_without_overwrite(tmp_path):
    import json
    artifact, reset, pose = inputs()
    folder = tmp_path / 'episode'
    channel = ResetWaypointChannel(artifact, expected_reset=reset, artifact_dir=folder)
    channel.update(pose, np.zeros((11, 3)), tick=0, observed_at_ns=reset['at_ns'],
                   available_at_ns=reset['at_ns'], decision_at_ns=reset['at_ns'])
    saved = json.loads((folder / 'instruction.json').read_text())
    assert saved == artifact
    waypoint = json.loads((folder / 'waypoints.jsonl').read_text())
    assert waypoint['instruction_sha256'] == saved['instruction_sha256']
    assert waypoint['reset'] == reset
    with pytest.raises(FileExistsError):
        ResetWaypointChannel(artifact, expected_reset=reset, artifact_dir=folder)


def test_terminal_task_goal_is_not_the_local_planner_waypoint():
    artifact, reset, pose = inputs()
    channel = ResetWaypointChannel(artifact, expected_reset=reset)
    tensors, receipt = channel.update(pose, np.zeros((11, 3)), tick=0,
        observed_at_ns=reset['at_ns'], available_at_ns=reset['at_ns'], decision_at_ns=reset['at_ns'])
    np.testing.assert_array_equal(receipt['goal_pose'], artifact['task_goal_pose'])
    assert receipt['target_stage'] == 'task_completion'
    planner = np.asarray(receipt['planner_waypoint_diagnostic']['goal_pose'])
    assert np.linalg.norm(planner[:3, 3] - np.asarray(receipt['goal_pose'])[:3, 3]) > .1
    np.testing.assert_allclose(tensors['goal_object_delta'][:, 2], .2)
    moved = pose.copy(); moved[0, 3] = .01
    now = reset['at_ns'] + 33333333
    _, later = channel.update(moved, np.zeros((11, 3)), tick=1,
        observed_at_ns=now, available_at_ns=now, decision_at_ns=now)
    assert later['goal_pose'] == receipt['goal_pose']
    assert not later['live_observation_available_before_reset']


def test_unstated_task_stage_and_future_terminal_substitution_rejected():
    artifact, reset, _ = inputs()
    sources = copy.deepcopy(artifact['sources'])
    del sources['rule']['payload']['target_stage']
    sources['rule']['source_sha256'] = digest(sources['rule']['payload'])
    with pytest.raises(ValueError, match='task_completion'):
        instruction_from_displacement(sources['anchor'], sources['rule'], sources['query'], sources['surface'],
            reset=reset, object_slot=0, hand_slot=0, goal_type='position_only')
    artifact['task_goal_pose'][0][3] = 123.
    artifact['instruction_sha256'] = digest({k: v for k, v in artifact.items() if k != 'instruction_sha256'})
    with pytest.raises(ValueError, match='not derived'):
        validate_instruction(artifact)


def test_exact_retries_are_idempotent_and_conflicting_duplicates_rejected(tmp_path):
    artifact, reset, pose = inputs()
    channel = ResetWaypointChannel(artifact, expected_reset=reset, artifact_dir=tmp_path / 'episode')
    args = dict(tick=0, observed_at_ns=reset['at_ns'], available_at_ns=reset['at_ns'],
                decision_at_ns=reset['at_ns'])
    first, receipt = channel.update(pose, np.zeros((11, 3)), **args)
    size = channel.written_bytes
    first['goal_object_delta'][:] = 999  # caller mutation must not corrupt retries
    retry, retried_receipt = channel.update(pose, np.zeros((11, 3)), **args)
    assert retry['goal_object_delta'].max() < 1
    assert receipt == retried_receipt and channel.written_bytes == size
    assert len((tmp_path / 'episode/waypoints.jsonl').read_text().splitlines()) == 1
    with pytest.raises(ValueError, match='conflicting duplicate'):
        channel.update(pose, np.ones((11, 3)), **args)


def test_write_refusal_does_not_advance_state(tmp_path, monkeypatch):
    import wam_rl_policy_v3.goal_channel as module
    artifact, reset, pose = inputs()
    channel = ResetWaypointChannel(artifact, expected_reset=reset, artifact_dir=tmp_path / 'episode')
    args = dict(tick=0, observed_at_ns=reset['at_ns'], available_at_ns=reset['at_ns'],
                decision_at_ns=reset['at_ns'])
    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 0})())
        with pytest.raises(RuntimeError, match='20GiB'):
            channel.update(pose, np.zeros((11, 3)), **args)
    assert channel.last_tick == -1
    actual, receipt = channel.update(pose, np.zeros((11, 3)), **args)
    expected, fresh_receipt = ResetWaypointChannel(artifact, expected_reset=reset).update(
        pose, np.zeros((11, 3)), **args)
    assert receipt == fresh_receipt
    np.testing.assert_array_equal(actual['goal_object_delta'], expected['goal_object_delta'])
