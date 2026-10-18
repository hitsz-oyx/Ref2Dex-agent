"""Single-environment scheduled v4 plans and BC-initialized PPO actor.

This boundary exposes predictions, not proof of physical feasibility. The
first executor consumes hand tau; C is returned separately for a later matched
conditioning experiment. Native actions are Shadow's 24 channels.
"""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn

from .student_inputs import student_features


class BCWarmStartActor(nn.Module):
    """Preserve BC mean/normalization; learn the entire actor during PPO.

PPO must store the sampled, unclipped action and its Normal log probability.
Clipping to the native bounds is an execution transform, not a replacement
for the sampled action when computing probability ratios.
    """
    action_kind = 'independent-gaussian'

    def __init__(self, packet, initial_std=.05):
        super().__init__()
        from .bc_mlp import CurrentMLP,SCHEDULED_SCHEMA,SCHEDULED_DIM
        self.current_model = packet.get('schema') == SCHEDULED_SCHEMA
        if self.current_model:
            from .control_contracts import scheduled_conditioning_contract
            declared=packet.get('config',{}).get('conditioning')
            if declared is not None and declared!=scheduled_conditioning_contract():raise ValueError('BC conditioning contract mismatch')
            if not np.isfinite(initial_std) or initial_std <= 0:
                raise ValueError('positive finite exploration std required')
            self.actor=CurrentMLP(SCHEDULED_DIM)
            self.actor.load_state_dict(packet['model'],strict=True)
            self.log_std=nn.Parameter(torch.full((24,),float(np.log(initial_std))))
            return
        if packet['variant'] != 'p2_tau' or packet['input_dim'] != 403:
            raise ValueError('requires p2_tau BC: 139 state + 264 world tau')
        if not np.isfinite(initial_std) or initial_std <= 0:
            raise ValueError('positive finite exploration std required')
        hidden = int(packet['hidden'])
        self.actor = nn.Sequential(nn.Linear(403, hidden), nn.ELU(),
                                   nn.Linear(hidden, hidden), nn.ELU(),
                                   nn.Linear(hidden, 24))
        self.actor.load_state_dict(packet['state_dict'], strict=True)
        mean = torch.as_tensor(packet['mean'], dtype=torch.float32).clone()
        std = torch.as_tensor(packet['std'], dtype=torch.float32).clone()
        if (mean.shape != (403,) or std.shape != (403,) or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or (std <= 0).any()):
            raise ValueError('invalid frozen BC normalization')
        self.register_buffer('input_mean', mean)
        self.register_buffer('input_std', std)
        self.log_std = nn.Parameter(torch.full((24,), float(np.log(initial_std))))

    def forward(self, observation):
        if self.current_model:
            if not torch.isfinite(observation).all():raise ValueError('finite BC input required')
            return self.actor(observation)
        if observation.shape[-1] != 403 or not torch.isfinite(observation).all():
            raise ValueError('finite raw 403D observation required')
        return self.actor((observation-self.input_mean)/self.input_std)

    def distribution(self, observation):
        mean = self(observation)
        return torch.distributions.Normal(mean, self.log_std.clamp(-8, 1).exp())

    def act_inference(self, observation):
        return self(observation).clamp(-1, 1)


class ScheduledV4ExecutorInputs:
    """Caller owns measured 60Hz history and must reset at each episode.

Use seven consecutive observations, never a recorded future, for H4. Startup
history must be acquired by the environment before calling this boundary.
The scheduler reference is a task instruction supplied at reset, not the
teacher's future measured path. Rebuild the scheduler for a new task goal.
    """
    def __init__(self, runtime, scheduler):
        if len(runtime.geometry) != 1:
            raise ValueError('initial integration supports one environment')
        if not np.array_equal(runtime.geometry[0][2], scheduler.points):
            raise ValueError('scheduler must use the original WAM flow queries')
        self.runtime, self.scheduler = runtime, scheduler

    def reset(self):
        self.scheduler.reset()

    def observation(self, objects, hands, times, current_pose, p2):
        objects, hands, times = map(np.asarray, (objects, hands, times))
        if (objects.shape != (7, 7) or hands.shape != (7, 11, 3)
                or times.shape != (7,) or not np.isfinite(times).all()
                or not np.isfinite(objects).all() or not np.isfinite(hands).all()
                or not np.allclose(np.diff(times), 1/60, atol=1e-6)):
            raise ValueError('requires seven consecutive 60Hz object/hand observations')
        if np.asarray(p2).shape != (139,) or not np.isfinite(p2).all():
            raise ValueError('single environment finite 139D state required')
        if not np.allclose(np.linalg.norm(objects[:, 3:7], axis=-1), 1., atol=1e-4):
            raise ValueError('unit object quaternion required')
        # Object records use xyz + xyzw, as does the native snapshot contract.
        from scipy.spatial.transform import Rotation
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(objects[-1, 3:7]).as_matrix()
        pose[:3, 3] = objects[-1, :3]
        if not np.allclose(pose, current_pose, atol=1e-6):
            raise ValueError('current pose does not match the last measured object')
        decision = self.scheduler.update(pose, float(times[-1]), hand_points=hands[-1])
        batch = self.runtime.batch(objects[None, ::2], hands[None, ::2],
                                   goal_delta=decision['goal_delta'][None])
        precision = (torch.autocast('cuda', dtype=torch.bfloat16)
                     if self.runtime.device.type == 'cuda' else nullcontext())
        with precision:
            prediction = self.runtime.predict(batch)
        tau = self.runtime.tau_for_student(prediction, pose)
        interaction, valid = self.runtime.interaction_for_executor(prediction)
        return dict(observation=student_features(p2, 'p2_tau', tau=tau),
                    tau=tau, interaction=interaction, interaction_valid=valid,
                    scheduler=decision)


class ScheduledCurrentExecutorInputs:
    """Full-current BC input at every measured tick, including startup.

    One instance per environment. Reset on every episode; no simulator state
    or teacher futures are substituted for the seven-frame measured history.
    """
    def __init__(self,runtime,scheduler):
        from collections import deque
        if len(runtime.geometry)!=1 or not np.array_equal(runtime.geometry[0][2],scheduler.points):
            raise ValueError('one environment with identical flow queries required')
        self.runtime,self.scheduler=runtime,scheduler
        self.history=deque(maxlen=7)

    def reset(self):
        self.history.clear();self.scheduler.reset()

    def observation(self,current,pose,hand,time_s,*,execution_ready=None,lost_control=False):
        from scipy.spatial.transform import Rotation
        current,pose,hand=map(np.asarray,(current,pose,hand))
        if (current.shape!=(300,) or pose.shape!=(4,4) or hand.shape!=(11,3)
                or not all(np.isfinite(v).all() for v in (current,pose,hand)) or not np.isfinite(time_s)):
            raise ValueError('finite current300, pose4x4, hand11x3 and time required')
        if self.history and not np.isclose(time_s-self.history[-1][0],1/60,atol=1e-6,rtol=0):
            raise ValueError('consecutive 60Hz measurements required; reset on new episode')
        decision=self.scheduler.update(pose,float(time_s),hand_points=hand,
            execution_ready=execution_ready,lost_control=lost_control)
        obj=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3].copy()).as_quat()]
        self.history.append((float(time_s),obj,hand.copy()))
        tau=np.zeros((8,11,3),np.float32);interaction=interaction_valid=None
        valid=len(self.history)==7
        if valid:
            rows=list(self.history)[::2]
            batch=self.runtime.batch(np.stack([r[1] for r in rows])[None],
                np.stack([r[2] for r in rows])[None],goal_delta=decision['goal_delta'][None])
            precision=torch.autocast('cuda',dtype=torch.bfloat16) if self.runtime.device.type=='cuda' else nullcontext()
            with precision:prediction=self.runtime.predict(batch)
            tau=self.runtime.tau_for_student(prediction,pose)
            interaction,interaction_valid=self.runtime.interaction_for_executor(prediction)
        return dict(observation=np.r_[current,tau.reshape(-1),float(valid)].astype(np.float32),
            tau=tau,history_valid=valid,scheduler=decision,interaction=interaction,interaction_valid=interaction_valid)
