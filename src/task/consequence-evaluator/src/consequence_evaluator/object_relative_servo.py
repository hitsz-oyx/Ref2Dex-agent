"""Analytic GT wrist transport; oracle experiment, no learned inverse."""
import hashlib
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .retargeter import (DOF_NAMES, ACTIVE_FINGERS, FINGER_SCALE,
                        wrist_rotation, closest_euler, commanded_targets, native_control)
from .gate1 import episode_outcome


def transport_wrist(target_q, source_object, live_object, reference_q):
    """Left-multiply wrist by live_object @ inverse(source_object).

    Wrist link6 translation is native q[:3]; hand_base_link has a fixed
    orientation offset, which cancels under this left multiplication.
    Finger shape/PD preload is preserved exactly in native index order.
    """
    target_q, source_object, live_object = map(np.asarray, (target_q, source_object, live_object))
    rotation = live_object[..., :3, :3] @ np.swapaxes(source_object[..., :3, :3], -1, -2)
    position = target_q[..., :3]-source_object[..., :3, 3]
    result = target_q.copy()
    result[..., :3] = np.einsum('...ij,...j->...i', rotation, position)+live_object[..., :3, 3]
    result[..., 3:6] = closest_euler(rotation @ wrist_rotation(target_q), np.asarray(reference_q)[..., 3:6])
    return result.astype('float32')


def local_points(hand, object_pose):
    return np.einsum('...ji,...pj->...pi', object_pose[..., :3, :3],
                     hand-object_pose[..., None, :3, 3])


def bounded_transport(target, corrected, translation_cap=.02, rotation_cap=.15):
    """Bound feedback about the world nominal, retaining its planned motion."""
    result = corrected.copy()
    delta = corrected[:3]-target[:3]
    result[:3] = target[:3]+delta*min(1., translation_cap/max(float(np.linalg.norm(delta)), 1e-12))
    delta_rot = wrist_rotation(corrected) @ wrist_rotation(target).T
    rv = Rotation.from_matrix(delta_rot).as_rotvec()
    rv *= min(1., rotation_cap/max(float(np.linalg.norm(rv)), 1e-12))
    result[3:6] = closest_euler(Rotation.from_rotvec(rv).as_matrix() @ wrist_rotation(target), target[3:6])
    return result.astype('float32')


class ObjectRelativeGTExecution:
    roles = ['reactive_teacher', 'world_gt_command', 'object_gt_command', 'object_gt_measured_nextq']

    def __init__(self, source, task, player, identity, anchor='current', layout='command_vs_measured'):
        with source.open('rb') as stream:
            packet = pickle.load(stream)
        if (packet.get('engineering_only') is not True or task.num_envs != 4
                or packet.get('role_names', [None])[0] != 'reactive_teacher'
                or packet['dof_position'].shape != (543, 4, 18)
                or packet['actions'].shape != (542, 4, 18) or packet['done'][:-1, 0].any()):
            raise ValueError('full teacher source/four-role oracle required')
        for key in ('physics_hash', 'controller_hash', 'backend', 'environment', 'actor_execution'):
            if identity[key] != packet['replay_identity'][key]:
                raise ValueError('object relative source runtime identity mismatch: '+key)
        self.body_names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
        names = task.gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0])
        if tuple(names) != DOF_NAMES:
            raise ValueError('actual Gym native DOF order mismatch')
        scale = task._pd_action_scale.cpu().numpy()
        if (not np.allclose(scale[:6], [1, 1, 1, np.pi, np.pi, np.pi])
                or not np.allclose(scale[ACTIVE_FINGERS], FINGER_SCALE)
                or np.max(np.abs(task._pd_action_offset.cpu().numpy())) > 1e-6):
            raise ValueError('native PD mapping mismatch')
        self.source = source
        self.packet = packet
        self.source_pose = packet['object_pose'][:, 0]
        self.source_hand = packet['hand_keypoints'][:, 0]
        self.source_q = packet['dof_position'][:, 0]
        self.source_targets = commanded_targets(self.source_q[:-1], packet['actions'][:, 0])
        self.anchor = anchor
        self.layout = layout
        if layout == 'transport_ablation':
            self.roles = ['reactive_teacher', 'world_gt_command', 'translation_gt_command', 'bounded_se3_gt_command']
        elif layout == 'chunk_alignment':
            self.roles = ['reactive_teacher', 'world_gt_command', 'query24_se3_gt_command', 'query24_bounded_gt_command']
        self.device = player.device
        self.desired, self.clip_counts, self.errors, self.live_poses = [], [], [], []
        self.query_ticks, self.query_poses, self.query_chunks = [], [], []

    def control(self, tick, task, points, teacher_control):
        from .collection import pose_matrix
        q = task._dof_pos.detach().cpu().numpy().copy()
        live = pose_matrix(task._target_states.detach().cpu().numpy())
        source_pose = self.source_pose[tick+(self.anchor == 'future')]
        relative_command = transport_wrist(self.source_targets[tick], source_pose, live[2], q[2])
        relative_measured = transport_wrist(self.source_q[tick+1], source_pose, live[3], q[3])
        # Measured q may differ slightly from the native coupling because of
        # physical loads; independent joints determine actual PD targets.
        for dst, src, ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
            relative_measured[dst] = relative_measured[src]*ratio
        desired = np.stack((self.source_targets[tick], relative_command, relative_measured))
        if self.layout == 'transport_ablation':
            translation = self.source_targets[tick].copy()
            translation[:3] += live[2, :3, 3]-source_pose[:3, 3]
            corrected = transport_wrist(self.source_targets[tick], source_pose, live[3], q[3])
            bounded = bounded_transport(self.source_targets[tick], corrected)
            desired = np.stack((self.source_targets[tick], translation, bounded))
        elif self.layout == 'chunk_alignment':
            if tick % 24 == 0:
                chunks = []
                for role in (2, 3):
                    values = []
                    for future_tick in range(tick, min(tick+24, 542)):
                        nominal = self.source_targets[future_tick]
                        corrected = transport_wrist(nominal, source_pose, live[role], q[role])
                        values.append(corrected if role == 2 else bounded_transport(nominal, corrected))
                    chunks.append(np.stack(values))
                self.cached_chunks = chunks
                self.query_ticks.append(tick); self.query_poses.append(live.copy())
                self.query_chunks.append(chunks)
            desired = np.stack((self.source_targets[tick], self.cached_chunks[0][tick % 24],
                                self.cached_chunks[1][tick % 24]))
        raw = native_control(desired, q[1:])
        clipped = np.clip(raw, -1, 1)
        result = teacher_control.clone()
        result[1:] = torch.as_tensor(clipped, dtype=torch.float32, device=self.device)
        native = task._action_to_pd_targets(result.clone()).detach().cpu().numpy()
        rebuilt = commanded_targets(q[1:], clipped)
        if np.max(np.abs(native[1:]-rebuilt)) > 2e-5:
            raise ValueError('actual native PD target conversion mismatch')
        if not np.isfinite(desired).all():
            raise ValueError('nonfinite transported GT target')
        self.desired.append(desired.copy()); self.live_poses.append(live.copy())
        self.clip_counts.append(np.sum(raw != clipped, -1))
        self.errors.append(np.max(np.abs(rebuilt-desired), -1))
        return result

    def finish(self, packets, identity, requested, metadata):
        if len(packets['actions']) != 542:
            raise ValueError('object-relative gate requires full542 controls')
        source_local = local_points(self.source_hand, self.source_pose)
        outcomes, metrics = {}, {}
        for i, name in enumerate(self.roles):
            role = {key: packets[key][:, i] for key in ('object_pose', 'hand_keypoints',
                    'surface_gap', 'support_gap', 'table_footprint', 'object_velocity')}
            role['timestamps'] = packets['timestamps']
            outcome = episode_outcome(role); outcomes[name] = outcome
            world = role['hand_keypoints'][1:]-self.source_hand[1:]
            relative = local_points(role['hand_keypoints'], role['object_pose'])[1:]-source_local[1:]
            metrics[name] = dict(maximum_held_frames=outcome['maximum_held_frames'],
                full_task_success=outcome['success'], intermediate_loss_events=outcome['intermediate_loss_events'],
                hand_coordinate_rmse_mm=float(np.sqrt(np.mean(world**2))*1000),
                object_local_hand_coordinate_rmse_mm=float(np.sqrt(np.mean(relative**2))*1000),
                contact_stage_local_rmse_mm=float(np.sqrt(np.mean(relative[59:200]**2))*1000),
                max_lift_m=float(np.max(role['object_pose'][:, 2, 3]-role['object_pose'][0, 2, 3])))
        teacher = outcomes[self.roles[0]]['maximum_held_frames']
        passes = {name: bool(teacher >= 45 and metrics[name]['maximum_held_frames'] >= .9*teacher
                            and outcomes[name]['intermediate_loss_events'] == 0)
                  for name in self.roles[1:]}
        primary = {'transport_ablation':'bounded_se3_gt_command',
                   'chunk_alignment':'query24_bounded_gt_command'}.get(self.layout, 'object_gt_command')
        gate = dict(teacher_held=teacher, held_fraction=.9, role_passes=passes,
                    passed=passes[primary], primary_role=primary, engineering_only=True,
                    secondary_local_rmse_threshold_mm=40)
        extra = dict(schema='ref2dex.object-relative-gt-servo.v1', role_names=self.roles,
            group_mode='synchronous_object_relative_gt_servo', training_allowed=False, engineering_only=True,
            requested_controls=np.asarray(requested), desired_pd_targets=np.asarray(self.desired),
            live_object_query_poses=np.asarray(self.live_poses), object_anchor=self.anchor,
            servo_layout=self.layout, translation_feedback_cap_m=(.02 if self.layout == 'transport_ablation' else None),
            rotation_feedback_cap_rad=(.15 if self.layout == 'transport_ablation' else None),
            alignment_query_ticks=self.query_ticks, alignment_query_poses=self.query_poses,
            alignment_query_chunks=self.query_chunks,
            clipped_coordinate_counts=np.asarray(self.clip_counts), commanded_target_max_abs_error=np.asarray(self.errors),
            gt_source=str(self.source), gt_source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            privileged_future_geometry=True, native_body_names=self.body_names,
            source_outcome=self.packet['role_outcomes']['reactive_teacher'],
            target_semantics='every-step analytic object-conditioned wrist transport; independent source fingers',
            replay_identity=identity, metrics=metrics, role_outcomes=outcomes, gate=gate, **metadata)
        return dict(**packets, **extra)
