"""Equivalent BC inference without unused future readouts; batched CPU H4 transforms."""
import numpy as np
import torch
from shadow_finetune.execution import Controller
from shadow_finetune.data import collate
from video_data_preparation.unidexgrasp_data import object_poses


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
        features=np.zeros((len(points),18),np.float32);features[:,:3]=normals;features[:,3:12]=dynamics(moved[None])[0];features[:,12]=1
        start=np.zeros((2,11,3),np.float32);start[0]=local_hand[env,-1]
        history=np.zeros((2,11,9),np.float32);history[0]=hand_dynamics[env]
        valid=np.zeros((2,11),bool);valid[0]=True
        query=points[np.linspace(0,len(points)-1,32,dtype=int)].astype(np.float32)
        result.append(dict(xyz=points.astype(np.float32),features=features,start_hand=start,start_hand_history=history,
            start_hand_valid=valid,object_valid=np.ones(1,bool),scene_points=query[None],start_object_points=query,
            goal_object_delta=np.broadcast_to(deltas[env],query.shape).astype(np.float32).copy(),object_goal_valid=np.ones(32,bool)))
    return result


class FastController(Controller):
    def act(self,snapshot,observation,tick):
        if tick==1:self.history=[];self.invalid=set()
        self.history.append(snapshot)
        if len(self.history)>7:self.history.pop(0)
        past=[max(len(self.history)-1+offset,0) for offset in (-6,-4,-2,0)]
        objects=np.stack([self.history[i]['object'] for i in past],1);hands=np.stack([self.history[i]['hand'] for i in past],1)
        samples=batched_inputs(objects,hands,snapshot['goal'],self.geometry)
        for env,sample in enumerate(samples):
            if (sample['start_hand'][0]<-2).any() or not np.isfinite(sample['start_hand']).all():self.invalid.add(env)
            sample.update(policy_state=np.clip(snapshot['observation'][env],-5,5),
                trajectory_target=np.zeros((24,2,11,3),np.float32),trajectory_valid=np.zeros((24,2,11),bool),
                scene_target=np.zeros((1,24,32,3),np.float32),scene_valid=np.zeros((1,24,32),bool),action_target=np.zeros((8,24),np.float32))
        if self.invalid:raise ValueError('BC H4 workspace invalid: '+str(sorted(self.invalid)))
        batch={k:v.cuda() for k,v in collate(samples).items()}
        shared,valid,_=self.model.forecaster.encode(batch)
        prediction=self.model.actor(batch['policy_state'],shared,valid)[:,0]
        if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite BC')
        return prediction.clamp(-1,1)
