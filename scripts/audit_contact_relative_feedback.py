#!/usr/bin/env python3
"""Independent current-time actuation, raw-contact and mesh geometry audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import gpu_admission,sha


def run(args):
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from audit_contact_risk_interventions import rotation
    torch.set_num_threads(2);begin=time.monotonic()
    b=torch.load(args.record,map_location='cpu',weights_only=False);n=len(b['state'])
    if b['schema']!='ref2dex.contact_relative_feedback_source.v1' or not n:raise ValueError('nonempty fixed source')
    if b['cm_used'] or b['optimizer_used'] or b['model_training'] or not b['frozen_experts']:
        raise ValueError('new feedback source must not use incompatible old model')
    if not b['assignment_after_observation'] or b['future_done'].any() or b['gain']!=.5:
        raise ValueError('randomized complete fixed programs')
    if b['expert_fingerprint_before']!=b['expert_fingerprint_after'] or not b['root_identity_verified']:
        raise ValueError('native root or experts drift')
    if b['native_dof_names'][:6]!=['joint1','joint2','joint3','joint4','joint5','joint6']:
        raise ValueError('native translation DOF order')
    for key,value in b.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('nonfinite '+key)
    if b['future_state'].shape!=(n,10,49) or abs(b['control_dt_seconds']-1/30)>1e-8:
        raise ValueError('H10 dimensions')
    errors={}
    def check(key,x,y,tolerance=2e-5):
        difference=float((x-y).abs().max());errors[key]=difference
        if difference>tolerance:raise ValueError(key+' drift '+str(difference))
    mapping=torch.tensor([0,0,1,1,2,3]);draw=b['allocation']
    if b['allocation_to_option']!=mapping.tolist() or (draw<0).any() or (draw>=6).any():raise ValueError('slot map')
    if not torch.equal(mapping[draw],b['assignment']):raise ValueError('actual selection')
    check('slot_probability',b['allocation_probabilities'],torch.full((n,6),1/6),1e-7)
    check('merged_propensity',b['propensity'],torch.tensor([1/3,1/3,1/6,1/6])[b['assignment']],1e-7)
    # Reproduce allocation in actual tick/environment order, independently of storage order.
    rng=torch.Generator(device='cuda').manual_seed(b['assignment_seed'])
    for tick in sorted(set(b['trigger'].tolist())):
        index=(b['trigger']==tick).nonzero().flatten();index=index[b['env_id'][index].argsort()]
        expected=torch.randint(6,(len(index),),device='cuda',generator=rng).cpu()
        if not torch.equal(expected,draw[index]):raise ValueError('pre-only allocation RNG')
    buckets=torch.tensor([int(hashlib.sha256(f'13671/{int(i)}/{int(j)}'.encode()).hexdigest()[:8],16)%100 for i,j in zip(b['motion_id'],b['start_frame'])])
    if b['split_group_seed']!=13671 or not torch.equal(buckets,b['split_group_bucket']):raise ValueError('fixed group split')
    pre=torch.cat((b['state'][:,None],b['future_state'][:,:-1]),1)
    if not torch.equal(b['history'][:,-1,:49],b['state']):raise ValueError('current state history timing')
    anchor=b['hold_target'];offset,scale=b['pd_offset'],b['pd_scale'];bank=b['expert_bank']
    if bank.shape!=(n,10,6,18) or (bank.abs()>1+1e-6).any() or (scale[:6]<=0).any():raise ValueError('native domain')
    check('initial_relative',b['initial_relative'],b['state'][:,36:39]-b['state'][:,:3],0.)
    correction=torch.zeros(n,10,3,3)
    correction[:,:,1]=.5/30*(pre[:,:,43:46]-pre[:,:,18:21])
    correction[:,:,2]=.5*(pre[:,:,36:39]-pre[:,:,:3]-b['initial_relative'][:,None])
    check('current_correction',b['requested_corrections'],correction,2e-6)
    expected=bank[:,:,1,None].expand(-1,-1,4,-1).clone();expected[:,:,0]=bank[:,:,4]
    expected[:,:,1:,3:6]=((anchor[:,None,None,3:6]-pre[:,:,None,3:6]-offset[3:6])/scale[3:6]).clamp(-1,1)
    expected[:,:,1:,:3]=(expected[:,:,1:,:3]+correction/scale[:3]).clamp(-1,1)
    index=b['assignment'][:,None,None,None].expand(-1,10,1,18)
    actual=expected.gather(2,index).squeeze(2)
    check('actual_command',b['actual_action'],actual,2e-6)
    check('initial_candidates',b['candidate_actions'],expected[:,0],2e-6)
    if not torch.equal(expected[:,0,1],expected[:,0,3]):raise ValueError('position program initial Cup alias')
    raw=actual.clone();raw[...,6:]=(1+raw[...,6:])*.5
    pd=offset+scale*raw;pd[...,:6]+=pre[...,:6]
    for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:pd[...,dst]=pd[...,src]*ratio
    check('native_pd',pd,b['actual_pd_targets'])
    holding=b['assignment']!=0
    if holding.any():check('rotation_hold',pd[holding,:,3:6],anchor[holding,None,3:6])
    selected_bank=bank[:,:,1].clone();selected_bank[b['assignment']==0]=bank[b['assignment']==0,:,4]
    if not torch.equal(actual[:,:,6:],selected_bank[:,:,6:]):raise ValueError('nominal fingers changed')
    if not torch.equal(actual[b['assignment']==1,:,:3],bank[b['assignment']==1,:,1,:3]):raise ValueError('Cup translation changed')
    mass=b['mass_kg'];weight=mass*b['gravity_magnitude']
    observed=torch.stack((b['future_hand_force'].norm(dim=-1).amax(-1),b['future_object_force'].norm(dim=-1)),-1)/weight[:,None,None]
    if not torch.equal(observed>.1,b['future_contact']):raise ValueError('actual force contact bits')
    check('force_ratio_relative',(observed-b['future_force_ratio'])/observed.abs().clamp_min(1),torch.zeros_like(observed),2e-6)
    initial=torch.stack((b['initial_hand_force'].norm(dim=-1).amax(-1),b['initial_object_force'].norm(dim=-1)),-1)/weight[:,None]
    if not (initial>.1).all() or not torch.equal((initial>.1),b['history'][:,-1,49:51].bool()):raise ValueError('contact trigger')
    if (b['state'][:,38]-b['rest_z']<.005).any() or (b['trigger']<10).any():raise ValueError('current trigger phase')
    # Independent collision support and table upper plane on GPU, no producer geometry helper.
    assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    def vertices(path):return torch.tensor([list(map(float,line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')],device='cuda')
    v=vertices(assets/'objects/airplane/airplane.obj')*b['object_scale'];table=vertices(assets/'objects/table/table.obj')
    extent=table.amax(0)-table.amin(0);axis=int(extent.argmin());lo,hi=table.amin(0)[axis],table.amax(0)[axis]
    states=torch.cat((b['state'][:,None],b['future_state']),1)
    poses=states[:,:,36:49].flatten(0,1).cuda();tables=b['table_pose'][:,None].expand(-1,11,-1).flatten(0,1).cuda()
    clears=[]
    for start in range(0,len(poses),96):
        obj,t=poses[start:start+96],tables[start:start+96]
        normal=rotation(t[:,3:7],torch)[:,:,axis];sign=torch.where(normal[:,2]>=0,1.,-1.);normal=normal*sign[:,None]
        direction=torch.einsum('nji,nj->ni',rotation(obj[:,3:7],torch),normal)
        support=(v@direction.T).amin(0);plane=torch.where(sign>0,hi,-lo)
        clears.append((support+((obj[:,:3]-t[:,:3])*normal).sum(-1)-plane).cpu())
    clearance=torch.cat(clears).reshape(n,11)
    check('mesh_clearance',clearance,torch.cat((b['initial_clearance'][:,None],b['future_clearance']),1),2e-6)
    support=b['future_contact'][:,-3:].all(-1).all(-1)&(clearance[:,-3:]>=.002).all(-1)
    clear=(clearance[:,0]>=.002)&(b['state'][:,38]-b['rest_z']>=.03)
    score=((b['future_state'][:,-3:,38].amin(-1)-b['rest_z']).clamp_min(0)*support-(b['state'][:,38]-b['rest_z']).clamp_min(0))*1000
    if not torch.equal(support,b['outcome']['retained_clear']) or not torch.equal(clear,b['outcome']['initially_clear']):raise ValueError('task support labels')
    check('supported_height_mm',score,b['outcome']['supported_change_mm'])
    # Current and next native observation carry the same measured relative geometry.
    for prefix,physical,positions,velocities,observation in (
        ('pre',pre,torch.cat((b['initial_key_positions'][:,None],b['future_key_positions'][:,:-1]),1),
         torch.cat((b['initial_key_velocities'][:,None],b['future_key_velocities'][:,:-1]),1),b['native_observation']),
        ('post',b['future_state'],b['future_key_positions'],b['future_key_velocities'],b['future_native_observation'])):
        x,z=observation[...,649:652],observation[...,652:655];local=torch.stack((x,torch.cross(z,x,dim=-1),z),-1)
        r=rotation(physical[...,39:43],torch)
        pos=observation[...,406:454].reshape(n,10,16,3)[:,:,[0,3,6,9,12,15]]-observation[...,646:649,None].transpose(-1,-2)
        vel=observation[...,156:204].reshape(n,10,16,3)[:,:,[0,3,6,9,12,15]]-observation[...,655:658,None].transpose(-1,-2)
        check(prefix+'_relative_position',torch.einsum('ntij,ntkj->ntki',local.transpose(-1,-2),pos),
            torch.einsum('ntij,ntkj->ntki',r.transpose(-1,-2),positions-physical[...,36:39,None].transpose(-1,-2)))
        check(prefix+'_relative_velocity',torch.einsum('ntij,ntkj->ntki',local.transpose(-1,-2),vel),
            torch.einsum('ntij,ntkj->ntki',r.transpose(-1,-2),velocities-physical[...,43:46,None].transpose(-1,-2)))
    # Same-observation Cup PD difference is actuation, not an unexecuted trajectory.
    allraw=expected.clone();allraw[...,6:]=(1+allraw[...,6:])*.5
    allpd=offset+scale*allraw;allpd[...,:6]+=pre[:,:,None,:6]
    for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:allpd[...,dst]=allpd[...,src]*ratio
    delta=(allpd-allpd[:,:,1,None]).abs().amax(-1)>1e-5
    changed=[]
    for option in range(4):
        rows=b['assignment']==option
        changed.append(dict(option=option,actual_windows=int(rows.sum()),
            changed_windows=int(delta[rows,:,option].any(-1).sum()),changed_steps=int(delta[rows,:,option].sum())))
    result=dict(run_status='COMPLETED',engineering_passed=True,smoke_only=b['smoke_only'],rows=n,steps=n*10,
        episodes=len(set(b['episode_id'])),errors=errors,actuation=changed,record_sha256=sha(args.record),
        elapsed_seconds=time.monotonic()-begin,scope='independent execution and observed physics; no opportunity gate or unexecuted outcomes')
    if args.output.exists():raise ValueError('unique audit output')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5);run(p.parse_args())
