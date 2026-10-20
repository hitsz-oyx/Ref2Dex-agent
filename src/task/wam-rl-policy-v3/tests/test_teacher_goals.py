import json

import numpy as np

from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
from wam_rl_policy_v3.teacher_quality import QUALITY_SCHEMA,TABLE_SUPPORT_CONTRACT
from wam_rl_policy_v3.task_annotations import GOAL_SOURCE
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.task_runtime import TaskController
from wam_rl_policy_v3.robot_adapter import InspireAdapter


def build_teacher_fixture(tmp_path):
    name = 'test_object_motion'
    source = tmp_path / 'batch/sequences' / name
    sidecar = tmp_path / 'audit/sequences' / name
    source.mkdir(parents=True); sidecar.mkdir(parents=True)
    n = 8
    poses = np.repeat(np.eye(4)[None], n + 1, axis=0)
    poses[:, 0, 3] = np.arange(n+1) * .01
    points = np.array([[x,y,z] for x in (-.01,.01) for y in (-.01,.01) for z in (-.01,.01)])
    hands = np.full((n+1,11,3), .5)
    spec = TaskSpec.from_reference(poses, source_fps=30, terminal_mode='HOLD', trim_interaction=False)
    packet = spec.packet(); packet['source_role'] = GOAL_SOURCE
    (sidecar/'task.json').write_text(json.dumps(packet))
    np.savez(source/'canonical.npz', points=points)
    np.savez(sidecar/'admission.npz', supported=np.ones(n+1, bool))
    valid = np.arange(n+1) > 0
    np.savez(source/'trajectory.npz', action=np.zeros((n,18)), q=np.zeros((n+1,18)),
        dq=np.zeros((n+1,18)), native_current_obs=np.zeros((n+1,721)), obj=poses,
        velocity=np.zeros((n+1,6)), dof_force_valid=valid, contact_force_valid=valid,
        dof_force=np.zeros((n+1,18)), body_contact_forces=np.zeros((n+1,5,3)), hand=hands)
    (source.parents[1]/'manifest.json').write_text(json.dumps(dict(observation_contract=dict(contact_body_ids=list(range(5))))))
    manifest = tmp_path/'audit/manifest.json'
    manifest.write_text(json.dumps(dict(schema=QUALITY_SCHEMA, status='COMPLETED', training_allowed=True,
        goal_source=GOAL_SOURCE, table_support_contract=TABLE_SUPPORT_CONTRACT, records=[dict(sequence=name, source=str(source), sidecar='sequences/'+name,
        training_prefix_commands=n, split='train', engineering='PASS')])))
    return manifest, packet, poses, points, hands, valid, n


def test_corpus_rejects_previous_support_contract_before_loading_rows(tmp_path):
    import pytest
    manifest,*_=build_teacher_fixture(tmp_path)
    packet=json.loads(manifest.read_text());packet['schema']='ref2dex.teacher-measured-rollout-admission.v1'
    packet.pop('table_support_contract');manifest.write_text(json.dumps(packet))
    with pytest.raises(ValueError,match='corrected table support'):
        TeacherRolloutCorpus(manifest,'train')


def test_cache_goal_replay_uses_the_identical_task_decisions_as_actor_inputs(tmp_path):
    manifest, packet, poses, points, hands, valid, n = build_teacher_fixture(tmp_path)
    episode = TeacherRolloutCorpus(manifest, 'train').episode(0)
    controller = TaskController(TaskSpec.from_packet(packet), points)
    for t in range(n):
        result = controller.observe(poses[t], InspireAdapter.canonical_hand(hands[t]), np.zeros(6), t/30,
            contact_forces=np.zeros((5,3)), force_valid=bool(valid[t]), supported=True)
        np.testing.assert_array_equal(episode['task'][t], result['task'])
        np.testing.assert_array_equal(episode['local_goal_pose'][t], result['decision'].local_goal_pose)
        # ACQUIRE without measured grasp evidence must not advance a second WAM scheduler.
        np.testing.assert_array_equal(episode['local_goal_pose'][t], poses[t])
