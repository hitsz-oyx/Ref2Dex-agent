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
    rule = source(dict(rule='fixed_world_translation', translation_m=[0., 0., .02]), 'task_instruction')
    query = source(query, 'reset_geometry')
    surface = source(np.zeros((16, 3)), 'reset_geometry')
    artifact = instruction_from_displacement(anchor, rule, query, surface, reset=reset,
        object_slot=0, hand_slot=0, goal_type='position_only')
    return artifact, reset, pose


def test_current_object_pose_is_legal_reset_input():
    artifact, reset, _ = inputs()
    validate_source(artifact['sources']['anchor'], reset, role='reset_observation')
    validate_instruction(artifact)


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
            np.zeros((2, 778), np.int64), receipt['goal_pose'])
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
