"""Equivalent BC inference without unused future readouts; batched CPU H4 transforms."""
import numpy as np
import torch
from shadow_finetune.execution import Controller
from shadow_finetune.data import collate
from video_data_preparation.unidexgrasp_data import object_poses

# Must match pose_mesh_data.v4_interaction.STATIC_THRESHOLD_M.  The online
# module cannot import the data-preparation task at simulator runtime, so the
# contract is asserted by a cross-task test instead.
V4_STATIC_THRESHOLD_M = .002


def type_summaries(shared,valid,scene_count,hand_count):
    """Keep scene/hand/known-goal summaries separate; never include future readouts."""
    groups=((0,scene_count),(scene_count,scene_count+hand_count),(scene_count+hand_count,shared.shape[1]))
    if shared.shape[-1]!=768 or any(stop<=start for start,stop in groups):raise ValueError('WAM scene/hand/goal token contract')
    summaries=[]
    for start,stop in groups:
        mask=valid[:,start:stop,None]
        value=torch.where(mask,shared[:,start:stop],0.).sum(1)/mask.sum(1).clamp_min(1)
        summaries.append(value)
    return torch.cat(summaries,-1).detach()


def batched_inputs(objects,hands,goals,geometries):
    # objects B,H,7 and hands B,H,11,3; retain original float64 pose arithmetic.
    poses=object_poses(objects);now=poses[:,-1];r=now[:,:3,:3];center=now[:,:3,3]
    local_hand=np.einsum('bhki,bij->bhkj',hands-center[:,None,None],r)
    relative=np.linalg.inv(now)[:,None]@poses
    dynamics=lambda x:np.concatenate(((x[:,-1]-x[:,-2])*30,(x[:,-1]-2*x[:,-2]+x[:,-3])*900,x[:,-1]-x[:,0]),-1)
    hand_dynamics=dynamics(local_hand);deltas=np.einsum('bi,bij->bj',goals-center,r)
    result=[]
    for env,geometry in enumerate(geometries):
        points=geometry['surface'];normals=geometry['normals']
        moved=np.einsum('hij,pj->hpi',relative[env,:,:3,:3],points)+relative[env,:,None,:3,3]
        features=np.zeros((len(points),18),np.float32);features[:,:3]=normals;features[:,3:12]=dynamics(moved[None])[0]
        features[:,12]=1
        features[:,13]=np.linalg.norm(features[:,9:12],axis=-1)<V4_STATIC_THRESHOLD_M
        start=np.zeros((2,11,3),np.float32);start[0]=local_hand[env,-1]
        history=np.zeros((2,11,9),np.float32);history[0]=hand_dynamics[env]
        valid=np.zeros((2,11),bool);valid[0]=True
        query=points[np.linspace(0,len(points)-1,32,dtype=int)].astype(np.float32)
        result.append(dict(xyz=points.astype(np.float32),features=features,start_hand=start,start_hand_history=history,
            start_hand_valid=valid,object_valid=np.ones(1,bool),scene_points=query[None],start_object_points=query,
            goal_object_delta=np.broadcast_to(deltas[env],query.shape).astype(np.float32).copy(),object_goal_valid=np.ones(32,bool)))
    return result


class FastController(Controller):
    build_inputs = staticmethod(batched_inputs)
    collate_fn = staticmethod(collate)

    def _ensure_histories(self, count):
        """Create one short history per simulator environment.

        The old implementation kept one batch-level history.  That lets a
        single terminated environment overwrite the temporal context of every
        other environment in the batch.  Keeping the histories separately
        also makes the reset boundary explicit for callers that run a vector
        environment.
        """
        if not hasattr(self, 'histories') or len(self.histories) != count:
            self.histories = [[] for _ in range(count)]

    def reset_envs(self, env_ids):
        """Drop temporal context for the listed environments before next act."""
        if not hasattr(self, 'histories'):
            return
        for env_id in env_ids:
            env_id = int(env_id)
            if 0 <= env_id < len(self.histories):
                self.histories[env_id] = []
                self.invalid.discard(env_id)

    def mark_reset_envs(self, env_ids):
        """Schedule a reset after the terminal observation has been encoded."""
        pending = getattr(self, '_pending_reset_envs', set())
        pending.update(int(env_id) for env_id in env_ids)
        self._pending_reset_envs = pending

    def act(self,snapshot,observation,tick):
        count = len(snapshot['object'])
        self._ensure_histories(count)
        if tick == 1:
            self.histories = [[] for _ in range(count)]
            self.invalid = set()
        pending = getattr(self, '_pending_reset_envs', set())
        if pending:
            self.reset_envs(pending)
            self._pending_reset_envs = set()
        for env_id in range(count):
            self.histories[env_id].append({key: value[env_id:env_id+1]
                                           for key, value in snapshot.items()})
            self.histories[env_id] = self.histories[env_id][-7:]
        offsets = (-6, -4, -2, 0)
        past = [[max(len(history)-1+offset, 0) for offset in offsets]
                for history in self.histories]
        objects=np.stack([np.stack([self.histories[e][i]['object'][0] for i in past[e]], 0)
                          for e in range(count)], 0)
        hands=np.stack([np.stack([self.histories[e][i]['hand'][0] for i in past[e]], 0)
                        for e in range(count)], 0)
        samples=self.build_inputs(objects,hands,snapshot['goal'],self.geometry)
        admitted=[];indices=[]
        for env,sample in enumerate(samples):
            if not np.isfinite(sample['start_hand']).all():raise FloatingPointError('Nonfinite native BC landmarks')
            if (sample['start_hand'][0]<-2).any():self.invalid.add(env)
            if env in self.invalid:continue
            sample.update(policy_state=np.clip(snapshot['observation'][env],-5,5),
                trajectory_target=np.zeros((24,2,11,3),np.float32),trajectory_valid=np.zeros((24,2,11),bool),
                scene_target=np.zeros((1,24,32,3),np.float32),scene_valid=np.zeros((1,24,32),bool),action_target=np.zeros((8,24),np.float32))
            admitted.append(sample);indices.append(env)
        if self.invalid and not getattr(self,'allow_invalid_zero',False):raise ValueError('BC H4 workspace invalid: '+str(sorted(self.invalid)))
        action=torch.zeros(len(samples),24,device='cuda')
        if getattr(self,'expose_wam_features',False):self.last_wam_features=torch.zeros(len(samples),2304,device='cuda')
        if not admitted:return action
        batch={k:v.cuda() for k,v in self.collate_fn(admitted).items()}
        shared,valid,scene=self.model.forecaster.encode(batch)
        if getattr(self,'expose_wam_features',False):
            hand_count=batch['start_hand_valid'].flatten(1).shape[1]
            self.last_wam_features[indices]=type_summaries(shared,valid,scene['tokens'].shape[1],hand_count)
        if getattr(self,'features_only',False):return action
        prediction=self.model.actor(batch['policy_state'],shared,valid)[:,0]
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite BC')
        action[indices]=prediction.clamp(-1,1)
        return action
