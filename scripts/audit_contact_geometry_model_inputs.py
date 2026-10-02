#!/usr/bin/env python3
"""Independent world-frame recomputation of model inputs, labels and priors."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def independent(record,data):
    import torch
    b=record;n=len(b['state']);dt=b['control_dt_seconds']
    states=torch.cat((b['state'][:,None],b['future_state']),1)
    pre,post=states[:,:-1],states[:,1:]
    positions=torch.cat((b['initial_key_positions'][:,None],b['future_key_positions']),1)
    velocities=torch.cat((b['initial_key_velocities'][:,None],b['future_key_velocities']),1)
    # Build rotations from the independently saved actual object states.
    x,y,z,w=states[...,39:43].unbind(-1)
    R=torch.stack((1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w,
                   2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w,
                   2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y),-1).reshape(n,11,3,3)
    relpos=torch.einsum('ntji,ntkj->ntki',R,positions-states[...,36:39].unsqueeze(-2))
    relvel=torch.einsum('ntji,ntkj->ntki',R,velocities-states[...,43:46].unsqueeze(-2))
    current_contact=torch.cat((b['history'][:,-1:,49:51],b['future_contact'][:,:-1].float()),1)
    history=[]
    for step in range(10):
        added=[torch.cat((pre[:,i],current_contact[:,i],b['actual_action'][:,i-1]),-1) for i in range(1,step+1)]
        h=torch.cat((b['history'][:,step:],torch.stack(added,1)),1) if added else b['history']
        history.append(h)
    expected_history=torch.stack(history,1).flatten(0,1)
    channels=[[0,1,2,3,4,5],[6,7],[8,9],[10,11],[12,13],[14,15,16,17]]
    q=torch.zeros(n,10,6,6);qd=torch.zeros_like(q);goal=torch.zeros_like(q)
    for i,indices in enumerate(channels):
        for j,index in enumerate(indices):
            q[:,:,i,j]=pre[:,:,index];qd[:,:,i,j]=pre[:,:,18+index]
            goal[:,:,i,j]=b['actual_pd_targets'][:,:,index]-pre[:,:,index]
    identity=torch.eye(6).reshape(1,1,6,6).expand(n,10,-1,-1)
    node_state=torch.cat((relpos[:,:-1],relvel[:,:-1],q,qd,identity),-1).flatten(0,1)
    chosen=b['candidate_weights'][torch.arange(n),b['assignment']]
    actions=torch.cat((goal,chosen[:,None].expand(-1,10,-1,-1)),-1).flatten(0,1)
    hand=torch.cat((b['initial_hand_force'][:,None],b['future_hand_force'][:,:-1]),1)
    obj=torch.cat((b['initial_object_force'][:,None],b['future_object_force'][:,:-1]),1)
    force=torch.cat((hand.flatten(-2),obj),-1)/(b['mass_kg'][:,None,None]*b['gravity_magnitude'])
    clr=torch.cat((b['initial_clearance'][:,None],b['future_clearance']),1)
    height=(pre[:,:,38]-b['rest_z'][:,None]).clamp_min(0)
    physical=torch.cat((force.sign()*force.abs().log1p(),clr[:,:-1,None],height[...,None]),-1).flatten(0,1)
    law=(b['assignment']!=0).float()[:,None,None].expand(-1,10,-1).flatten(0,1)
    node_delta=torch.cat(((relpos[:,1:]-relpos[:,:-1])/.005,
                          (relvel[:,1:]-relvel[:,:-1])/.1),-1).flatten(-2)
    step_target=torch.cat(((post[...,43:46]-pre[...,43:46])/.1,(post[...,36:39]-pre[...,36:39])/.005,
                           ((clr[:,1:]-clr[:,:-1])/.002)[...,None],b['future_contact'].float()),-1)
    final_z=post[:,-3:,38].amin(-1)
    final_height=(final_z-b['rest_z']).clamp_min(0)
    hand_present=b['future_contact'][:,-3:,0].all(-1);obj_present=b['future_contact'][:,-3:,1].all(-1)
    end_support=hand_present&obj_present&(clr[:,-3:]>=.002).all(-1)
    loss_event=torch.zeros(n,dtype=torch.bool);seen=clr[:,0]>=.002
    for step in range(10):
        loss_event |= seen&(clr[:,step+1]<.002)
        seen |= clr[:,step+1]>=.002
    macro=torch.stack(((final_z-b['state'][:,38])/.01,(clr[:,-3:].amin(-1)-clr[:,0])/.002,
                       hand_present.float(),obj_present.float(),loss_event.float(),
                       (end_support&(final_height>=.03)).float(),final_height*end_support/.01),-1)
    target=torch.cat((node_delta,step_target,macro[:,None].expand(-1,10,-1)),-1).flatten(0,1)
    omega=torch.einsum('ntji,ntj->nti',R[:,:-1],pre[...,46:49])
    node_prior=torch.cat(((relvel[:,:-1]-torch.cross(omega[:,:,None].expand_as(relpos[:,:-1]),relpos[:,:-1],dim=-1))*dt/.005,
                          torch.zeros_like(relvel[:,:-1])),-1).flatten(-2)
    logits=(current_contact*2-1)*math.log(9)
    step_prior=torch.cat((torch.zeros_like(pre[...,43:46]),pre[...,43:46]*dt/.005,
                          pre[...,45,None]*dt/.002,logits),-1)
    dz=torch.minimum(pre[...,45]*dt*8,pre[...,45]*dt*10)
    support=current_contact.bool().all(-1)&(clr[:,:-1]+dz>=.002)
    cv_height=(height+dz).clamp_min(0)*support
    loss_now=(clr[:,:-1]>=.002)&(clr[:,:-1]+torch.minimum(pre[...,45]*dt,pre[...,45]*dt*10)<.002)
    macro_prior=torch.cat((dz[...,None]/.01,dz[...,None]/.002,logits,
                           ((loss_now.float()*2-1)*math.log(9))[...,None],
                           (((support&(cv_height>=.03)).float()*2-1)*math.log(9))[...,None],cv_height[...,None]/.01),-1)
    prior=torch.cat((node_prior,step_prior,macro_prior),-1).flatten(0,1)
    expected=dict(history=expected_history,physical=physical,node_state=node_state,node_action=actions,law=law,target=target,prior=prior)
    errors={k:float((value-data[k]).abs().max()) for k,value in expected.items()}
    # Normalized deltas/kinematic skips can be large; retain float32 relative
    # precision instead of applying an unscaled absolute limit to every value.
    for k,value in expected.items():
        atol=2e-4 if k in ('target','prior') else 2e-5
        if not torch.allclose(value,data[k],atol=atol,rtol=2e-6):
            raise ValueError('independent model assembly '+json.dumps(errors))
    if not torch.equal(data['first'],(torch.arange(n*10)%10==0)):
        raise ValueError('initial-only H10 supervision mask')
    if not torch.equal(data['early'],(~b['outcome']['initially_clear']).repeat_interleave(10)):
        raise ValueError('current phase stratum')
    signed=final_height*end_support-(b['state'][:,38]-b['rest_z']).clamp_min(0)
    if not torch.allclose(signed*1000,b['outcome']['supported_change_mm'],atol=1e-5):
        raise ValueError('control score matches actual source label')
    return expected,errors


def run(args):
    begin=time.monotonic()
    if args.output.exists():raise ValueError('unique engineering output required')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.contact_geometry_consequence import transitions,candidate_features,GeometryConsequenceModel,FEATURES,loss
    torch.set_num_threads(2)
    b=torch.load(args.record,map_location='cpu',weights_only=False);data=transitions(b)
    _,errors=independent(b,data)
    c=candidate_features(b,data);selected=torch.arange(len(b['state']))*8+b['assignment']
    candidate_errors={k:float((c[k][selected]-data[k][::10]).abs().max()) for k in list(FEATURES)+['prior']}
    if max(candidate_errors.values())>2e-6:raise ValueError('actual candidate feature drift')
    # Genuine last-post corruption must fail the independent world-label reconstruction.
    corrupt=dict(b);corrupt['future_native_observation']=b['future_native_observation'].clone()
    corrupt['future_native_observation'][0,-1,406]+=.01
    try:
        independent(corrupt,transitions(corrupt))
    except ValueError:
        rejected=True
    else:
        raise ValueError('last-post corruption escaped model audit')
    rows=torch.arange(min(64,len(data['target'])))
    norm={}
    for k in FEATURES:
        x=data[k];dims=(0,1) if k in ('history','node_state','node_action') else 0
        norm[k]=(x.mean(dims),x.std(dims,unbiased=False).clamp_min(.001))
    feats={k:((data[k][rows]-mean)/std).clamp(-8,8).cuda() for k,(mean,std) in norm.items()}
    prior=data['prior'][rows].cuda();target=data['target'][rows].cuda();first=data['first'][rows].cuda()
    gradients={}
    for mode in ('cm','state_only','shuffled','direct_score'):
        torch.manual_seed(12601)
        model=GeometryConsequenceModel(data['physical'].shape[-1],mode).cuda()
        pred=model(*(feats[k] for k in FEATURES),prior)
        value=loss(pred,target,first,mode);value.backward()
        grad=model.nodes[0].weight.grad
        if pred.shape!=(len(rows),52) or not torch.isfinite(pred).all() or not torch.isfinite(value) or grad is None or not torch.isfinite(grad).all() or not grad.abs().sum()>0:
            raise ValueError('GPU finite/shared-node gradient smoke')
        action_gradient=float(grad[:,24:].norm())
        if (mode=='state_only' and action_gradient!=0) or (mode!='state_only' and action_gradient<=0):
            raise ValueError('action ablation/shared-node gradient wiring')
        gradients[mode]=dict(loss=float(value),node_gradient_norm=float(grad.norm()),
                             action_gradient_norm=action_gradient,prediction_shape=list(pred.shape))
    inputs=[Path(__file__),ROOT/'src/task/CmResidual/contact_geometry_consequence.py',args.record,
            ROOT/'docs/experiments/probes/P-20261002-contact-geometry-model-contract.md']
    result=dict(run_status='COMPLETED',engineering_passed=True,gpu=admission,rows=len(b['state']),steps=len(data['target']),
                independent_errors=errors,actual_candidate_errors=candidate_errors,last_post_corruption_rejected=rejected,
                gpu_gradient_smoke=gradients,input_sha256={str(p.resolve()):sha(p) for p in inputs},
                elapsed_seconds=time.monotonic()-begin,
                scope='engineering inputs/all52labels/priors and GPU gradients only; not fitted information or policy utility')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','gpu')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    run(p.parse_args())
