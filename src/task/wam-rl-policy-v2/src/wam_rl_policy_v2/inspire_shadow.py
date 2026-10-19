"""Inspire embodiment of the Shadow BC/PPO recipe, without hand references."""
from collections import deque
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .inspire_contract import ACTIVE,NATIVE_TO_WAM,WAM_TO_NATIVE

CURRENT_DIM=312
INPUT_DIM=CURRENT_DIM+264+1
CURRENT_SCHEMA='ref2dex.inspire-shadow-current-bc.v1'
SCHEMA='ref2dex.inspire-shadow-scheduled-bc.v1'
CONTRACT=dict(schema='ref2dex.inspire-shadow-conditioning.v1',robot_id='dexplore.inspire.v1',
    input_dim=INPUT_DIM,current_dim=CURRENT_DIM,action_dim=12,native_action_dim=18,
    hidden_dims=[1024,1024,512,512],activation='elu',native_hz=30,wam_hz=30,
    tau_steps=8,tau_hz=30,tau_frame='world',tau_representation='displacement_from_current_hand',
    tau_hand_order='native_base_then_finger_base_tip',tau_hand_slot='right',tau_units='metres',
    history_indices=[-3,-2,-1,0],startup='zero_tau_with_validity_zero',
    goal_semantics='spatial_path_waypoint',goal_reach_time_guaranteed=False,
    wam_precision='fp32',task='ordered_object_SE3_path',direct_hand_reference=False,phase_conditioning=False,
    interaction_conditioning=False,current_layout=['q18','dq18','measured248','object_pose_velocity13','task_endpoint_xyz3','previous_independent_action12'])


def object_state(poses,velocity):
    poses,velocity=np.asarray(poses),np.asarray(velocity)
    quat=Rotation.from_matrix(poses[...,:3,:3].reshape(-1,3,3)).as_quat()
    quat*=np.where(quat[:,3:4]<0,-1.,1.)
    return np.concatenate((poses[...,:3,3],quat.reshape(poses.shape[:-2]+(4,)),velocity),axis=-1).astype(np.float32)


def current_state(q,dq,measured,objects,goal,previous):
    arrays=[np.asarray(v,dtype=np.float32) for v in (q,dq,measured,objects,goal,previous)]
    sizes=(18,18,248,13,3,12)
    leading=arrays[0].shape[:-1]
    if any(a.shape!=leading+(n,) for a,n in zip(arrays,sizes)):
        raise ValueError('Inspire physical312 fields do not align')
    result=np.concatenate(arrays,axis=-1)
    if not np.isfinite(result).all():raise ValueError('nonfinite physical state')
    return np.clip(result,-5,5)


def measured_from_legacy(current721):
    """Only the audited physical slice; 0:9 and 257:721 contain references."""
    values=np.asarray(current721)
    if values.shape[-1]!=721:raise ValueError('current721 source required')
    return values[...,9:257].copy()


def expanded_model(scheduled=False):
    from .bc_mlp import NormalizedCommandMLP
    return NormalizedCommandMLP(INPUT_DIM if scheduled else CURRENT_DIM,CURRENT_DIM,12)


def initialize_scheduled(model,parent):
    if parent.get('schema')!=CURRENT_SCHEMA or model.input_dim!=INPUT_DIM:
        raise ValueError('full current312 BC parent required')
    state=model.state_dict()
    for key,value in parent['model'].items():
        if key=='net.0.weight':state[key].zero_();state[key][:,:CURRENT_DIM].copy_(value)
        elif key in ('x_mean','x_scale'):state[key][:CURRENT_DIM].copy_(value)
        else:state[key].copy_(value)
    model.load_state_dict(state,strict=True)


def tau_world(trajectory,rotations):
    trajectory=np.asarray(trajectory)
    if trajectory.shape[1:]!=(24,2,11,3):raise ValueError('batched WAM24x2x11x3 required')
    return np.einsum('btpi,bji->btpj',trajectory[:,:8,0][:,:,WAM_TO_NATIVE],rotations).astype(np.float32)


class BatchedInspireInputs:
    """Same ordered scheduler/WAM boundary, native H4 is already 30Hz."""
    def __init__(self,runtime,references,batch=128,inference_mode='full_fp32'):
        if inference_mode not in ('full_fp32','hand_fp32','hand_shared_tf32') or not 1<=batch<=256:
            raise ValueError('bounded explicit Inspire inference mode required')
        self.runtime,self.batch=runtime,batch
        self.inference_mode=inference_mode
        self.inference_audits=[];self.prediction_calls=0
        self.predict_trajectory=True
        self.histories=[deque(maxlen=4) for _ in references]
        self.schedulers=[None]*len(references)
        for i,reference in enumerate(references):self.reset(i,reference)

    def reset(self,i,reference):
        from .goal_scheduler import OrderedGoalScheduler
        self.histories[i].clear()
        self.schedulers[i]=OrderedGoalScheduler(reference,np.arange(len(reference))/30,
            self.runtime.geometry[i][2],interaction_points=self.runtime.geometry[i][0],goal_type='se3_pose')

    def observation(self,current,poses,hands,ticks,indices=None):
        import torch
        from .batched_geometry import online_batch,workspace_valid
        from .goal_scheduler import update_schedulers
        indices=list(range(len(self.histories))) if indices is None else list(indices)
        current,poses,hands,ticks=map(np.asarray,(current,poses,hands,ticks))
        hands=hands[:,NATIVE_TO_WAM]
        out=np.zeros((len(current),INPUT_DIM),np.float32);out[:,:CURRENT_DIM]=current
        invalid=np.zeros(len(current),bool);decisions=[None]*len(current)
        chosen=[self.schedulers[i] for i in indices]
        started=time.monotonic()
        decisions_chosen=update_schedulers(chosen,poses[indices],ticks[indices]/30,hands[indices],fast_geometry=True)
        self.timings=dict(scheduler_s=time.monotonic()-started,input_s=0.,model_s=0.)
        ready=[];objects=[];histories=[];goals=[];geometries=[]
        for i,decision in zip(indices,decisions_chosen):
            decisions[i]=decision
            history=self.histories[i]
            if history and ticks[i]!=history[-1][0]+1:raise ValueError('consecutive episode clock required')
            history.append((int(ticks[i]),poses[i].copy(),hands[i].copy()))
            if len(history)==4:
                rows=list(history);p=np.stack([r[1] for r in rows])
                obj=np.c_[p[:,:3,3],Rotation.from_matrix(p[:,:3,:3]).as_quat()]
                ready.append(i);objects.append(obj);histories.append([r[2] for r in rows]);goals.append(decision['goal_delta']);geometries.append(self.runtime.geometry[i])
        for start in range(0,len(ready),self.batch):
            stop=min(start+self.batch,len(ready));ids=np.asarray(ready[start:stop])
            started=time.monotonic()
            batch=online_batch(np.asarray(objects[start:stop]),np.asarray(histories[start:stop]),geometries[start:stop],np.asarray(goals[start:stop]),self.runtime.device)
            keep=workspace_valid(batch,self.runtime.config);host_keep=keep.cpu().numpy();invalid[ids[~host_keep]]=True
            ids=ids[host_keep]
            if not len(ids):continue
            batch={k:v[keep] for k,v in batch.items()}
            self.timings['input_s']+=time.monotonic()-started;started=time.monotonic()
            if not self.predict_trajectory:
                # Evaluation may elide prediction only after the caller proves
                # the actor's entire condition weight block is exactly zero.
                # History, scheduler and workspace termination stay identical.
                out[ids,-1]=1
                continue
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                if self.inference_mode=='full_fp32':prediction=self.runtime.predict(batch)
                else:prediction=self.runtime.predict_hand(batch,shared_tf32=self.inference_mode=='hand_shared_tf32')
                predicted=prediction['trajectory'].float()
                # Startup and periodic real-state gates cover live histories,
                # in addition to the offline multi-object precision audit.
                if self.inference_mode!='full_fp32' and self.prediction_calls%300==0:
                    reference=self.runtime.predict(batch)['trajectory'].float()
                    error=float((predicted-reference).abs().max())
                    audit=dict(call=self.prediction_calls,batch=len(ids),max_error_m=error,limit_m=.001)
                    self.inference_audits.append(audit)
                    if not torch.isfinite(predicted).all() or not error<=.001:
                        raise ValueError('optimized Inspire WAM exceeds real-state 1mm precision gate: '+str(audit))
                self.prediction_calls+=1
                trajectory=predicted.cpu().numpy()
            self.timings['model_s']+=time.monotonic()-started
            out[ids,CURRENT_DIM:-1]=tau_world(trajectory,poses[ids,:3,:3]).reshape(len(ids),264)
            out[ids,-1]=1
        return dict(observation=out,workspace_invalid=invalid,scheduler=decisions)

    def performance_contract(self):
        return dict(mode=self.inference_mode,batch=self.batch,
            geometry_math='fp32',shared_transformer_math='tf32' if self.inference_mode=='hand_shared_tf32' else 'fp32',
            output_dtype='fp32',ppo_math='unchanged',live_gate_m=.001,live_gate_every_calls=300)
