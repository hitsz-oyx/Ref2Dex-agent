"""Privileged GT-hand engineering execution; import after native Isaac Gym."""
import hashlib
import pickle

import numpy as np
import torch

from .retargeter import (SCHEMA, DOF_NAMES, ACTIVE_FINGERS, FINGER_SCALE,
    HandTrajectoryRetargeter, Standardizer, commanded_targets, absolute_targets,
    native_control, trajectory_input)
from .gate1 import episode_outcome


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GTRetargetExecution:
    roles = ['reactive_teacher', 'direct_gt_target_servo', 'learned_retargeter', 'retarget_repeat']

    def __init__(self, checkpoint, source, task, player, identity):
        with source.open('rb') as stream:
            packet = pickle.load(stream)
        payload = torch.load(checkpoint, map_location='cpu')
        if (payload.get('schema') != SCHEMA or tuple(payload.get('dof_names', [])) != DOF_NAMES
                or payload.get('active_fingers') != ACTIVE_FINGERS.tolist()
                or task.num_envs != 4 or packet.get('engineering_only') is not True
                or packet.get('role_names', [None])[0] != 'reactive_teacher'
                or packet['dof_position'].shape != (543, 4, 18)
                or packet['hand_keypoints'].shape != (543, 4, 11, 3)
                or packet['actions'].shape != (542, 4, 18)
                or packet['done'][:-1, 0].any()):
            raise ValueError('GT-retarget checkpoint/source contract mismatch')
        expected_source = payload['manifest']['splits']['test']
        if len(expected_source) != 1 or expected_source[0]['sha256'] != digest(source):
            raise ValueError('GT source must be the frozen held-out full launch')
        for key in ('physics_hash', 'controller_hash', 'backend', 'environment', 'actor_execution'):
            if identity[key] != packet['replay_identity'][key]:
                raise ValueError('GT source runtime identity drift: '+key)
        handle = task.gym.get_actor_handle(task.envs[0], 0)
        if tuple(task.gym.get_actor_dof_names(task.envs[0], handle)) != DOF_NAMES:
            raise ValueError('native Inspire DOF order mismatch')
        scale = task._pd_action_scale.cpu().numpy()
        offset = task._pd_action_offset.cpu().numpy()
        if (not np.allclose(scale[:6], [1, 1, 1, np.pi, np.pi, np.pi])
                or not np.allclose(scale[ACTIVE_FINGERS], FINGER_SCALE)
                or np.max(np.abs(offset)) > 1e-6):
            raise ValueError('native PD target scale/offset mismatch')
        self.device = player.device
        self.model = HandTrajectoryRetargeter(payload['width']).to(self.device)
        self.model.load_state_dict(payload['state_dict'], strict=True); self.model.eval()
        self.stats = {key: Standardizer(**value) for key, value in payload['statistics'].items()}
        self.source_hand = packet['hand_keypoints'][:, 0].copy()
        self.source_targets = commanded_targets(packet['dof_position'][:-1, 0], packet['actions'][:, 0])
        self.source_outcome = packet['role_outcomes']['reactive_teacher']
        self.checkpoint, self.source = checkpoint, source
        self.query_ticks, self.inputs, self.chunks, self.relatives = [], [], [], []
        self.desired, self.clip_counts, self.target_errors = [], [], []

    def control(self, tick, task, points, teacher_control):
        q = task._dof_pos.detach().cpu().numpy().copy()
        dq = task._dof_vel.detach().cpu().numpy().copy()
        if tick % 24 == 0:
            future = self.source_hand[np.minimum(tick+np.arange(1, 25), 542)][None]
            current_hand = points.detach().cpu().numpy()[2:3].copy()
            state = np.concatenate((q[2:3], dq[2:3]), -1)
            hand = trajectory_input(current_hand, future)
            with torch.no_grad():
                relative = self.stats['target'].decode(self.model(
                    torch.as_tensor(self.stats['hand'].encode(hand), device=self.device),
                    torch.as_tensor(self.stats['state'].encode(state), device=self.device)).cpu().numpy())
            self.chunk = absolute_targets(q[2:3], relative)[0]
            if not np.isfinite(self.chunk).all():
                raise ValueError('nonfinite learned retarget targets')
            self.query_ticks.append(tick)
            self.inputs.append(dict(current_hand=current_hand[0], future_hand=future[0], state=state[0]))
            self.chunks.append(self.chunk.copy()); self.relatives.append(relative[0].copy())
        desired = np.stack((self.source_targets[tick], self.chunk[tick % 24], self.chunk[tick % 24]))
        raw = native_control(desired, q[1:])
        clipped = np.clip(raw, -1, 1)
        reconstructed = commanded_targets(q[1:], clipped)
        # Verify the native function itself, with a clone because it rescales
        # finger actions in-place. Couplings and wrist increment mapping must agree.
        native = task._action_to_pd_targets(torch.as_tensor(
            np.vstack((teacher_control[:1].cpu().numpy(), clipped)), device=self.device).clone()).cpu().numpy()
        if np.max(np.abs(native[1:]-reconstructed)) > 2e-5:
            raise ValueError('actual native PD mapping differs from retarget adapter')
        self.desired.append(desired.copy())
        self.clip_counts.append(np.sum(raw != clipped, -1))
        self.target_errors.append(np.max(np.abs(reconstructed-desired), -1))
        result = teacher_control.clone()
        result[1:] = torch.as_tensor(clipped, dtype=torch.float32, device=self.device)
        return result

    def finish(self, packets, identity, requested, metadata):
        if len(packets['actions']) != 542:
            raise ValueError('GT-hand gate requires full542 controls')
        metrics, outcomes = {}, {}
        for i, name in enumerate(self.roles):
            role = {key: value[:, i] for key, value in packets.items()
                    if key in ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
                               'table_footprint', 'dof_position', 'dof_velocity', 'object_velocity')}
            role['timestamps'] = packets['timestamps']
            outcomes[name] = episode_outcome(role)
            delta = packets['hand_keypoints'][1:, i]-self.source_hand[1:]
            finger_delta = ((packets['hand_keypoints'][1:, i, 1:]-packets['hand_keypoints'][1:, i, :1])
                            -(self.source_hand[1:, 1:]-self.source_hand[1:, :1]))
            metrics[name] = dict(hand_coordinate_rmse_mm=float(np.sqrt(np.mean(delta**2))*1000),
                point_distance_rmse_mm=float(np.sqrt(np.mean(np.sum(delta**2, -1)))*1000),
                wrist_coordinate_rmse_mm=float(np.sqrt(np.mean(delta[:, 0]**2))*1000),
                wrist_relative_finger_rmse_mm=float(np.sqrt(np.mean(finger_delta**2))*1000),
                maximum_held_frames=outcomes[name]['maximum_held_frames'],
                full_task_success=outcomes[name]['success'],
                max_lift_m=float(np.max(role['object_pose'][:, 2, 3]-role['object_pose'][0, 2, 3])))
        teacher = outcomes[self.roles[0]]['maximum_held_frames']
        passes = {name: bool(teacher >= 45 and metrics[name]['maximum_held_frames'] >= .9*teacher
                            and metrics[name]['hand_coordinate_rmse_mm'] < 40)
                  for name in self.roles[1:]}
        gate = dict(teacher_held=teacher, source_held=self.source_outcome['maximum_held_frames'],
                    held_fraction=.9, hand_coordinate_rmse_threshold_mm=40, role_passes=passes,
                    passed=bool(all(passes.values())), engineering_only=True)
        extra = dict(schema='ref2dex.gt-hand-retarget-behavior.v1', role_names=self.roles,
            group_mode='synchronous_same_process_gt_hand_retarget', engineering_only=True,
            training_allowed=False, requested_controls=np.asarray(requested), replay_identity=identity,
            retarget_query_ticks=np.asarray(self.query_ticks), retarget_inputs=self.inputs,
            retarget_absolute_chunks=np.asarray(self.chunks), retarget_relative_chunks=np.asarray(self.relatives),
            desired_pd_targets=np.asarray(self.desired), clipped_coordinate_counts=np.asarray(self.clip_counts),
            commanded_target_max_abs_error=np.asarray(self.target_errors),
            checkpoint=str(self.checkpoint), checkpoint_sha256=digest(self.checkpoint),
            gt_source=str(self.source), gt_source_sha256=digest(self.source), source_outcome=self.source_outcome,
            target_semantics='fixed absolute PD target sequence; current-q mechanical adapter each step; R query every24',
            privileged_future_geometry=True, unused_terminal_input_padding=10,
            metrics=metrics, role_outcomes=outcomes, gate=gate, **metadata)
        return dict(**packets, **extra)
