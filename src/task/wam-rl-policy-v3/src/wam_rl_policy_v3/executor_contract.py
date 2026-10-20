"""Named robot/task/tau fields shared by offline BC and online PPO."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .control_contracts import HAND_LAYOUT

TASK_DIM=10
TASK_SCALE=np.array([.05]*3+[.35]*3+[1.]*4,np.float32)
EXECUTOR_SCHEMA='ref2dex.task-executor-observation.v1'


def task_condition(pose,decision):
    pose=np.asarray(pose);goal=decision.local_goal_pose
    rotation=pose[:3,:3].T
    translation=rotation@(goal[:3,3]-pose[:3,3])
    angle=Rotation.from_matrix(rotation@goal[:3,:3]).as_rotvec()
    phase=np.eye(3,dtype=np.float32)[int(decision.phase)]
    result=np.r_[translation,angle,decision.path_progress,phase].astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('finite measured task condition required')
    return result


@dataclass(frozen=True)
class ExecutorLayout:
    robot_id:str
    robot_dim:int
    action_dim:int
    native_hz:int
    with_tau:bool=True

    def __post_init__(self):
        if min(self.robot_dim,self.action_dim,self.native_hz)<=0:raise ValueError('positive robot dimensions/frequency required')

    @property
    def blocks(self):
        return dict(robot=self.robot_dim,task=TASK_DIM,**(dict(tau=264,tau_valid=1) if self.with_tau else {}))

    @property
    def input_dim(self):return sum(self.blocks.values())

    def packet(self):
        return dict(schema=EXECUTOR_SCHEMA,robot_id=self.robot_id,fields=self.blocks,action_dim=self.action_dim,
            native_hz=self.native_hz,task_scale=TASK_SCALE.tolist(),
            tau_frame='world',tau_units='metres',tau_hand_slot='right',tau_hand_layout=HAND_LAYOUT,
            tau_representation='displacement_from_current_hand',
            tau_times_s=[i/self.native_hz for i in range(1,9)] if self.with_tau else [],
            task_layout=['local_object_translation3','local_object_rotvec3','confirmed_progress1','current_phase_onehot3'])

    def wam_tau(self,trajectory,object_rotation):
        """Both robots use canonical palm/bases/tips; only native clocks differ."""
        if not self.with_tau:raise ValueError('current BC layout does not consume tau')
        from .tau_contract import v4_trajectory_to_student
        return v4_trajectory_to_student(trajectory,object_rotation=object_rotation,native_fps=self.native_hz)

    def pack(self,robot,task,*,tau=None,tau_valid=False):
        robot=np.asarray(robot,np.float32);task=np.asarray(task,np.float32)
        if robot.shape[-1:]!=(self.robot_dim,) or task.shape!=robot.shape[:-1]+(10,):raise ValueError('robot/task dimensions do not match executor layout')
        if not np.allclose(task[...,-3:].sum(-1),1) or not np.isin(task[...,-3:],[0,1]).all():raise ValueError('one current execution phase required')
        parts=[robot,task]
        if self.with_tau:
            valid=np.broadcast_to(np.asarray(tau_valid,bool),robot.shape[:-1])
            if tau is None:
                if valid.any():raise ValueError('valid tau requires a prediction')
                values=np.zeros(robot.shape[:-1]+(264,),np.float32)
            else:
                tau=np.asarray(tau,np.float32)
                if tau.shape!=robot.shape[:-1]+(8,11,3):raise ValueError('eight native-time world displacement samples required')
                values=np.where(valid[...,None],tau.reshape(robot.shape[:-1]+(264,)),0)
            parts.extend((values,valid[...,None].astype(np.float32)))
        elif tau is not None or np.asarray(tau_valid).any():raise ValueError('current BC layout does not consume tau')
        result=np.concatenate(parts,-1)
        if not np.isfinite(result).all():raise ValueError('finite executor observation required')
        return result


def expand_named_model(parent,source_blocks,target_blocks,*,clip_dim,source_column_permutations=None):
    """Copy named input columns and normalization; new columns start at zero."""
    from .bc_mlp import NormalizedCommandMLP
    if sum(source_blocks.values())!=parent.input_dim or any(k not in target_blocks or target_blocks[k]<v for k,v in source_blocks.items()):
        raise ValueError('expansion must preserve every declared source column')
    if clip_dim!=parent.current_dim:raise ValueError('migration must preserve the original raw clipping prefix')
    permutations=source_column_permutations or {}
    if set(permutations)-set(source_blocks):raise ValueError('unknown source permutation field')
    result=NormalizedCommandMLP(sum(target_blocks.values()),clip_dim,parent.net[-1].out_features)
    target=result.state_dict();source=parent.state_dict()
    for key,value in source.items():
        if key not in ('net.0.weight','x_mean','x_scale'):target[key].copy_(value)
    target['net.0.weight'].zero_()
    src_offset=0;starts={};offset=0
    for name,width in target_blocks.items():starts[name]=offset;offset+=width
    if target_blocks.get('task')==TASK_DIM:
        target['x_scale'][starts['task']:starts['task']+TASK_DIM].copy_(target['x_scale'].new_tensor(TASK_SCALE))
    for name,width in source_blocks.items():
        permutation=np.asarray(permutations.get(name,np.arange(width)))
        if permutation.shape!=(width,) or permutation.dtype.kind not in 'iu' or not np.array_equal(np.sort(permutation),np.arange(width)):
            raise ValueError('source field permutation must preserve every column exactly once')
        dst=slice(starts[name],starts[name]+width);src=src_offset+permutation
        if np.any((starts[name]+np.arange(width)<clip_dim)!=(src<clip_dim)):
            raise ValueError('migration cannot move fields across the original clipping boundary')
        target['net.0.weight'][:,dst].copy_(source['net.0.weight'][:,src])
        for key in ('x_mean','x_scale'):target[key][dst].copy_(source[key][src])
        src_offset+=width
    result.load_state_dict(target)
    return result
