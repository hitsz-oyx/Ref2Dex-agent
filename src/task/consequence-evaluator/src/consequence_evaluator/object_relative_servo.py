"""Analytic GT wrist transport; oracle experiment, no learned inverse."""
import hashlib
import json
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


def recover_wrist(hand, root_template, reference_q):
    """Recover the native wrist from palm and five fixed finger-root points.

    Joint rotations do not move their proximal-link origins. A calibrated
    URDF rest template therefore determines the rigid wrist transform without
    reading future joint states. Euler branch selection uses a supplied prior.
    """
    indices = [0, 1, 3, 5, 7, 9]
    template = np.asarray(root_template)[indices]
    observed = np.asarray(hand)[...,indices,:]
    a = template-template.mean(0)
    b = observed-observed.mean(-2,keepdims=True)
    u, _, vh = np.linalg.svd(np.einsum('pi,...pj->...ij',a,b))
    v = np.swapaxes(vh,-1,-2)
    correction = np.broadcast_to(np.eye(3),v.shape).copy()
    correction[...,2,2] = np.linalg.det(v@np.swapaxes(u,-1,-2))
    rotation = v@correction@np.swapaxes(u,-1,-2)
    result = np.asarray(reference_q).copy()
    result[...,:3] = observed.mean(-2)-np.einsum('...ij,j->...i',rotation,template.mean(0))
    result[...,3:6] = closest_euler(rotation,result[...,3:6])
    return result.astype('float32')


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

    def __init__(self, source, task, player, identity, anchor='current', layout='command_vs_measured', preload_path=None, inverse_path=None):
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
        rest = np.einsum('ij,pj->pi',wrist_rotation(self.source_q[0]).T,
                         self.source_hand[0]-self.source_q[0,:3])
        # Root geometry is calibrated from the known reset, then every future
        # wrist comes only from its 11 recorded keypoints and continuity.
        reconstructed = [self.source_q[0].copy()]
        for hand in self.source_hand[1:]:
            reconstructed.append(recover_wrist(hand,rest,reconstructed[-1]))
        self.geometry_wrist_q = np.stack(reconstructed)
        self.source_targets = commanded_targets(self.source_q[:-1], packet['actions'][:, 0])
        self.anchor = anchor
        self.layout = layout
        if layout == 'transport_ablation':
            self.roles = ['reactive_teacher', 'world_gt_command', 'translation_gt_command', 'bounded_se3_gt_command']
        elif layout == 'chunk_alignment':
            self.roles = ['reactive_teacher', 'world_gt_command', 'query24_se3_gt_command', 'query24_bounded_gt_command']
        elif layout == 'finger_preload':
            self.roles = ['reactive_teacher', 'world_gt_command', 'measured_finger_no_preload', 'measured_finger_fixed_preload']
        elif layout == 'finger_preload_late':
            self.roles = ['reactive_teacher', 'world_gt_command', 'late_measured_finger_no_preload', 'late_measured_finger_fixed_preload']
        elif layout == 'wrist_geometry_late':
            self.roles = ['reactive_teacher', 'world_gt_command', 'late_geometry_wrist_no_ff', 'late_geometry_wrist_velocity_ff']
        elif layout == 'geometry_inverse_late':
            self.roles = ['reactive_teacher', 'world_gt_command', 'late_full_measured_geometry', 'late_full_geometry_inverse']
        elif layout == 'geometry_inverse_relative_late':
            self.roles = ['reactive_teacher', 'world_gt_command', 'late_inverse_world', 'late_inverse_bounded_se3']
        elif layout == 'geometry_inverse_repeat':
            self.roles = ['reactive_teacher', 'world_gt_command', 'late_inverse_world', 'late_inverse_world_replica']
        elif layout == 'geometry_load_delta_late':
            self.roles = ['reactive_teacher', 'world_gt_command', 'late_inverse_world', 'late_inverse_command_anchor']
        elif layout == 'geometry_pd':
            self.roles = ['reactive_teacher', 'world_gt_command', 'geometry_wrist_ff_source_fingers', 'geometry_wrist_ff_fixed_finger_preload']
        self.preload_path=preload_path
        self.preload=None
        if preload_path is not None:
            self.preload=json.loads(preload_path.read_text())
            if (self.preload.get('schema')!='ref2dex.gt-preload-statistics.v1'
                    or self.preload.get('active_fingers')!=ACTIVE_FINGERS.tolist()
                    or self.preload.get('statistics_scope')!='frozen_ref7_train_launches_only'):
                raise ValueError('train-only preload statistics mismatch')
            for record in self.preload['sources']:
                from pathlib import Path
                if hashlib.sha256(Path(record['source']).read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError('preload training source hash drift')
        self.inverse_path = inverse_path
        self.inverse_q = None
        if inverse_path is not None:
            manifest = json.loads((inverse_path/'manifest.json').read_text())
            inverse_file = inverse_path/'inverse.npz'
            if (manifest.get('schema') != 'ref2dex.hand-geometry-inverse.v1'
                    or manifest.get('training_allowed') is not False
                    or manifest.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest()
                    or manifest.get('inverse_sha256') != hashlib.sha256(inverse_file.read_bytes()).hexdigest()
                    or manifest.get('hand_sha256') != hashlib.sha256(self.source_hand.tobytes()).hexdigest()):
                raise ValueError('geometry inverse artifact provenance mismatch')
            with np.load(inverse_file) as data:
                self.inverse_q = data['q'].copy()
                np.testing.assert_array_equal(data['target_points'],self.source_hand)
                np.testing.assert_array_equal(data['reset_q'],self.source_q[0])
            if self.inverse_q.shape != (543,18) or not np.isfinite(self.inverse_q).all():
                raise ValueError('invalid geometry inverse q')
        self.device = player.device
        self.desired, self.clip_counts, self.errors, self.live_poses = [], [], [], []
        self.query_ticks, self.query_poses, self.query_chunks = [], [], []
        self.last_applied = None
        self.command_anchor_offset = None

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
        elif self.layout in ('geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late'):
            if self.inverse_q is None:
                raise ValueError('full geometry inverse requires provenance-checked artifact')
            measured = self.source_targets[tick].copy()
            inverse = measured.copy()
            if tick >= 120:
                velocity = (self.geometry_wrist_q[min(tick+2,542),:6]-self.geometry_wrist_q[tick+1,:6])*30
                if tick == 541:
                    velocity = (self.geometry_wrist_q[542,:6]-self.geometry_wrist_q[541,:6])*30
                for value, fingers in ((measured,self.source_q[tick+1]),(inverse,self.inverse_q[tick+1])):
                    value[:6] = self.geometry_wrist_q[tick+1,:6]+.1*velocity
                    value[ACTIVE_FINGERS] = fingers[ACTIVE_FINGERS]
                    for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
                        value[dst] = value[src]*ratio
                if self.layout == 'geometry_inverse_relative_late':
                    measured = inverse.copy()
                    corrected = transport_wrist(inverse,source_pose,live[3],q[3])
                    inverse = bounded_transport(inverse,corrected)
                elif self.layout == 'geometry_inverse_repeat':
                    measured = inverse.copy()
                elif self.layout == 'geometry_load_delta_late':
                    measured = inverse.copy()
                    if tick == 120:
                        equilibrium = self.inverse_q[tick].copy()
                        equilibrium[:6] = self.geometry_wrist_q[tick,:6]+.1*(
                            self.geometry_wrist_q[tick+1,:6]-self.geometry_wrist_q[tick,:6])*30
                        self.command_anchor_offset = self.last_applied[2]-equilibrium
                    inverse += self.command_anchor_offset
            desired = np.stack((self.source_targets[tick],measured,inverse))
        elif self.layout == 'wrist_geometry_late':
            measured = self.source_targets[tick].copy()
            compensated = measured.copy()
            if tick >= 120:
                measured[:6] = self.geometry_wrist_q[tick+1,:6]
                velocity = (self.geometry_wrist_q[min(tick+2,542),:6]-self.geometry_wrist_q[tick+1,:6])*30
                if tick == 541:
                    velocity = (self.geometry_wrist_q[542,:6]-self.geometry_wrist_q[541,:6])*30
                compensated[:6] = measured[:6]+.1*velocity
            desired = np.stack((self.source_targets[tick],measured,compensated))
        elif self.layout in ('finger_preload','finger_preload_late','geometry_pd'):
            if self.preload is None:
                raise ValueError('geometry/preload diagnostic requires frozen training statistics')
            no_preload=self.source_targets[tick].copy()
            no_preload[ACTIVE_FINGERS]=self.source_q[tick+1,ACTIVE_FINGERS]
            fixed=no_preload.copy()
            fixed[ACTIVE_FINGERS]+=np.asarray(self.preload['finger_preload_median_rad'],dtype='float32')
            if self.layout=='geometry_pd':
                forward = min(tick+2,542)
                velocity=(self.source_q[forward,:6]-self.source_q[tick+1,:6])*30
                if tick==541:velocity=(self.source_q[542,:6]-self.source_q[541,:6])*30
                wrist=self.source_q[tick+1,:6]+.1*velocity
                no_preload[:6]=wrist;fixed[:6]=wrist
                no_preload[ACTIVE_FINGERS]=self.source_targets[tick,ACTIVE_FINGERS]
            for value in (no_preload,fixed):
                for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
                    value[dst]=value[src]*ratio
            if self.layout == 'finger_preload_late' and tick < 120:
                no_preload = self.source_targets[tick].copy()
                fixed = no_preload.copy()
            desired=np.stack((self.source_targets[tick],no_preload,fixed))
        raw = native_control(desired, q[1:])
        clipped = np.clip(raw, -1, 1)
        result = teacher_control.clone()
        result[1:] = torch.as_tensor(clipped, dtype=torch.float32, device=self.device)
        native = task._action_to_pd_targets(result.clone()).detach().cpu().numpy()
        rebuilt = commanded_targets(q[1:], clipped)
        self.last_applied = rebuilt.copy()
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
        reference_held = max(teacher, self.packet['role_outcomes']['reactive_teacher']['maximum_held_frames'])
        passes = {name: bool(teacher >= 45 and metrics[name]['maximum_held_frames'] >= .9*reference_held
                            and outcomes[name]['intermediate_loss_events'] == 0)
                  for name in self.roles[1:]}
        primary = {'transport_ablation':'bounded_se3_gt_command',
                   'chunk_alignment':'query24_bounded_gt_command',
                   'finger_preload':'measured_finger_fixed_preload',
                   'finger_preload_late':'late_measured_finger_fixed_preload',
                   'wrist_geometry_late':'late_geometry_wrist_velocity_ff',
                   'geometry_inverse_late':'late_full_geometry_inverse',
                   'geometry_inverse_relative_late':'late_inverse_bounded_se3',
                   'geometry_inverse_repeat':'late_inverse_world',
                   'geometry_load_delta_late':'late_inverse_command_anchor',
                   'geometry_pd':'geometry_wrist_ff_fixed_finger_preload'}.get(self.layout, 'object_gt_command')
        gate = dict(teacher_held=teacher, source_held=self.packet['role_outcomes']['reactive_teacher']['maximum_held_frames'],
                    reference_held=reference_held, held_fraction=.9, role_passes=passes,
                    passed=passes[primary], primary_role=primary, engineering_only=True,
                    secondary_local_rmse_threshold_mm=40)
        extra = dict(schema='ref2dex.object-relative-gt-servo.v1', role_names=self.roles,
            group_mode='synchronous_object_relative_gt_servo', training_allowed=False, engineering_only=True,
            requested_controls=np.asarray(requested), desired_pd_targets=np.asarray(self.desired),
            live_object_query_poses=np.asarray(self.live_poses), object_anchor=self.anchor,
            servo_layout=self.layout, translation_feedback_cap_m=(.02 if self.layout in ('transport_ablation', 'chunk_alignment','geometry_inverse_relative_late') else None),
            geometry_switch_tick=(120 if self.layout in ('finger_preload_late','wrist_geometry_late','geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late') else None),
            wrist_inverse_source=('11-point fixed roots with reset calibration' if self.layout in ('wrist_geometry_late','geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late') else None),
            command_anchor_offset=self.command_anchor_offset,
            geometry_inverse_path=(str(self.inverse_path) if self.inverse_path is not None else None),
            geometry_inverse_sha256=(hashlib.sha256((self.inverse_path/'inverse.npz').read_bytes()).hexdigest() if self.inverse_path is not None else None),
            rotation_feedback_cap_rad=(.15 if self.layout in ('transport_ablation', 'chunk_alignment','geometry_inverse_relative_late') else None),
            alignment_query_ticks=self.query_ticks, alignment_query_poses=self.query_poses,
            alignment_query_chunks=self.query_chunks,
            preload_statistics_path=(str(self.preload_path) if self.preload_path is not None else None),
            preload_statistics_sha256=(hashlib.sha256(self.preload_path.read_bytes()).hexdigest() if self.preload_path is not None else None),
            clipped_coordinate_counts=np.asarray(self.clip_counts), commanded_target_max_abs_error=np.asarray(self.errors),
            gt_source=str(self.source), gt_source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            privileged_future_geometry=True, native_body_names=self.body_names,
            source_outcome=self.packet['role_outcomes']['reactive_teacher'],
            target_semantics=('query24 frozen analytic object-conditioned target chunks; current-q mechanical adapter each step'
                              if self.layout == 'chunk_alignment' else
                              'source commanded wrist; measured future finger geometry with train-only preload ablation'
                              if self.layout in ('finger_preload','finger_preload_late') else
                              '11-point wrist inverse after source-command grasp prefix; with/without velocity compensation'
                              if self.layout == 'wrist_geometry_late' else
                              'full measured-q / 11-point geometry inverse after source-command grasp prefix; geometry wrist velocity compensation'
                              if self.layout == 'geometry_inverse_late' else
                              '11-point inverse after grasp prefix; world nominal versus bounded live-object SE3 correction'
                              if self.layout == 'geometry_inverse_relative_late' else
                              'same full11-point inverse in two physical roles after grasp prefix; replicas are not independent seeds'
                              if self.layout == 'geometry_inverse_repeat' else
                              'geometry increments anchored to the controllers own last applied PD target at120; no future command/preload labels'
                              if self.layout == 'geometry_load_delta_late' else
                              'measured future wrist geometry plus forward velocity damping compensation; source/fixed-preload fingers'
                              if self.layout == 'geometry_pd' else
                              'every-step analytic object-conditioned wrist transport; independent source fingers'),
            replay_identity=identity, metrics=metrics, role_outcomes=outcomes, gate=gate, **metadata)
        return dict(**packets, **extra)
