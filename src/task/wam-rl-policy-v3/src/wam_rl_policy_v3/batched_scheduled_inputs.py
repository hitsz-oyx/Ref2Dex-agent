"""Independent measured histories/schedulers, one frozen WAM forward per tick."""
import os
from collections import deque
from contextlib import nullcontext
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .wam_v4_runtime import v4_online_samples, collate_v4
from .tau_contract import v4_trajectory_to_student
from .batched_geometry import native_tau,online_batch,workspace_valid
from .goal_scheduler import update_schedulers,OrderedGoalScheduler


class BatchedScheduledInputs:
    def __init__(self, runtime, schedulers):
        if len(runtime.geometry) != len(schedulers) or not schedulers:
            raise ValueError('one scheduler per runtime geometry required')
        for geometry, scheduler in zip(runtime.geometry, schedulers):
            if not np.array_equal(geometry[2], scheduler.points):
                raise ValueError('flow query identities differ')
        self.runtime, self.schedulers = runtime, schedulers
        self.pipeline=os.environ.get('REF2DEX_BATCH_PIPELINE','legacy')
        if self.pipeline not in ('legacy','tau','inputs','scheduler','all'):raise ValueError('unknown batch pipeline')
        self.histories = [deque(maxlen=7) for _ in schedulers]

    def reset(self, indices=None):
        for i in range(len(self.schedulers)) if indices is None else indices:
            self.histories[i].clear()
            self.schedulers[i].reset()

    def observation(self, current, poses, hands, times):
        n = len(self.schedulers)
        device_output=self.pipeline in ('tau','inputs','all')
        current=torch.as_tensor(current,device=self.runtime.device) if device_output else np.asarray(current)
        current_finite=bool(torch.isfinite(current).all()) if device_output else np.isfinite(current).all()
        poses,hands,times=map(np.asarray,(poses,hands,times))
        if (current.shape != (n,300) or poses.shape != (n,4,4) or hands.shape != (n,11,3)
                or times.shape != (n,) or not current_finite or not all(np.isfinite(x).all() for x in (poses,hands,times))):
            raise ValueError('finite batched measured states required')
        # Validate the whole batch before advancing any scheduler/history.
        for history, now in zip(self.histories,times):
            if history and not np.isclose(now-history[-1][0],1/60,atol=1e-6,rtol=0):
                raise ValueError('consecutive 60Hz history required; reset the affected environment')
        fast_scheduler=self.pipeline in ('scheduler','all') and all(isinstance(s,OrderedGoalScheduler) for s in self.schedulers)
        decisions=update_schedulers(self.schedulers,poses,times,hands,fast_geometry=os.environ.get('REF2DEX_DEEP_OPT') in ('cpu','all')) if fast_scheduler else []
        objects_now=np.concatenate((poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3].copy()).as_quat()),axis=-1)
        for i, scheduler in enumerate(self.schedulers):
            pose=poses[i]
            if not fast_scheduler:decisions.append(scheduler.update(pose,float(times[i]),hand_points=hands[i]))
            obj=objects_now[i]
            self.histories[i].append((float(times[i]),obj,hands[i].copy()))
        ready=[i for i,h in enumerate(self.histories) if len(h)==7]
        tau=torch.zeros(n,8,11,3,device=self.runtime.device) if device_output else np.zeros((n,8,11,3),np.float32)
        valid=torch.zeros(n,device=self.runtime.device) if device_output else np.zeros(n,np.float32)
        workspace_invalid=np.zeros(n,bool)
        if ready:
            rows=[list(self.histories[i])[::2] for i in ready]
            objects=np.array([[r[1] for r in h] for h in rows]);hand_history=np.array([[r[2] for r in h] for h in rows])
            geometry=[self.runtime.geometry[i] for i in ready];goals=np.stack([decisions[i]['goal_delta'] for i in ready])
            if self.pipeline in ('inputs','all'):
                batch=online_batch(objects,hand_history,geometry,goals,self.runtime.device)
                keep=workspace_valid(batch,self.runtime.config)
                host_keep=keep.cpu().numpy()
                workspace_invalid[np.asarray(ready)[~host_keep]]=True
                ready=[i for i,ok in zip(ready,host_keep) if ok]
                batch={k:v[keep] for k,v in batch.items()}
            else:
                samples=v4_online_samples(objects,hand_history,geometry,goal_delta=goals)
                if hasattr(self.runtime,'workspace_valid'):
                    keep=self.runtime.workspace_valid(samples)
                    workspace_invalid[np.asarray(ready)[~keep]]=True
                    samples=[sample for sample,ok in zip(samples,keep) if ok]
                    ready=[i for i,ok in zip(ready,keep) if ok]
                if ready:batch={k:v.to(self.runtime.device) for k,v in collate_v4(samples).items()}
        if ready:
            precision=torch.autocast('cuda',dtype=torch.bfloat16) if self.runtime.device.type=='cuda' else nullcontext()
            with torch.inference_mode(), precision:
                trajectory=self.runtime.predict(batch)['trajectory']
            if device_output:
                tau[ready]=native_tau(trajectory,torch.as_tensor(poses[ready,:3,:3],device=self.runtime.device))
                valid[ready]=1
            else:
                trajectory=trajectory.float().cpu().numpy()
                for row,i in enumerate(ready):
                    tau[i]=v4_trajectory_to_student(trajectory[row],object_rotation=poses[i,:3,:3])
                    valid[i]=1
        observation=torch.cat((current.float(),tau.flatten(1),valid[:,None]),1) if device_output else np.concatenate((current,tau.reshape(n,-1),valid[:,None]),axis=1).astype(np.float32)
        return dict(observation=observation,tau=tau,history_valid=valid,scheduler=decisions,workspace_invalid=workspace_invalid)
