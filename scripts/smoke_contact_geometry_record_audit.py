#!/usr/bin/env python3
"""Synthetic in-memory record tests; never produces simulated/physical source data."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def fixture(torch):
    from src.task.CmResidual.contact_geometry_actions import sample_programs,executable_candidates,feedback_candidates
    n=3;generator=torch.Generator().manual_seed(20701)
    weights=sample_programs(n,'cpu',generator)
    bank=torch.rand(n,10,6,18,generator=generator)*.5-.25
    state=torch.zeros(n,49);state[:,38]=.6;state[:,41]=torch.sin(torch.tensor(.3/2));state[:,42]=torch.cos(torch.tensor(.3/2))
    future=state[:,None].repeat(1,10,1)
    history=torch.zeros(n,10,69);history[:,-1,:49]=state;history[:,-1,49:51]=1
    choice=torch.tensor([0,1,2]);row=torch.arange(n)
    anchor=torch.zeros(n,18);offset=torch.zeros(18);scale=torch.ones(18)
    candidates=[];actual=[];feedback=[]
    for step in range(10):
        commands=executable_candidates(bank[:,step],weights,anchor,state[:,:18],offset,scale)
        candidates.append(commands);actual.append(commands[row,choice])
        feedback.append(feedback_candidates(bank[:,step],weights)[row,choice])
    actual=torch.stack(actual,1);feedback=torch.stack(feedback,1)
    targets=actual.clone();targets[...,6:]=(targets[...,6:]+1)*.5
    for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        targets[...,dst]=targets[...,src]*ratio
    body=torch.zeros(n,10,16,3);body[...,0]=torch.linspace(-.08,.08,16);body[...,2]=.65
    keys=body[:,:,[0,3,6,9,12,15]].clone();native=torch.zeros(n,10,1442)
    native[...,406:454]=body.flatten(-2);native[...,646:649]=state[:,None,36:39]
    from audit_contact_geometry_source import rotation
    rot=rotation(state[:,39:43],torch)
    native[...,649:652]=rot[:,None,:,0];native[...,652:655]=rot[:,None,:,2]
    mass=torch.full((n,),.002593613);g=9.81
    hand=torch.zeros(n,10,5,3);hand[...,2]=mass[:,None,None]*g
    obj=torch.zeros(n,10,3);obj[...,2]=mass[:,None]*g
    bucket=int(hashlib.sha256(b'12651/0/0').hexdigest()[:8],16)%100
    return dict(schema='ref2dex.contact_geometry_source.v1',state=state,future_state=future,history=history,
                future_done=torch.zeros(n,10,dtype=torch.bool),frozen_experts=True,cm_used=False,optimizer_used=False,
                assignment_after_observation=True,control_dt_seconds=1/30,mass_kg=mass,gravity_magnitude=g,
                assignment=choice,allocation=torch.tensor([8,9,2]),allocation_to_option=[0,1,2,3,4,5,6,7,0,1],
                allocation_probabilities=torch.full((n,10),.1),propensity=torch.tensor([.2,.2,.1]),
                split_group_seed=12651,motion_id=torch.zeros(n,dtype=torch.long),start_frame=torch.zeros(n,dtype=torch.long),
                split_group_bucket=torch.full((n,),bucket),candidate_weights=weights,expert_bank=bank,
                hold_target=anchor,pd_offset=offset,pd_scale=scale,actual_action=actual,feedback_action=feedback,
                actual_pd_targets=targets,candidate_actions=candidates[0],initial_hand_force=hand[:,0],initial_object_force=obj[:,0],
                future_hand_force=hand,future_object_force=obj,future_contact=torch.ones(n,10,2,dtype=torch.bool),
                future_force_ratio=torch.ones(n,10,2),rest_z=torch.full((n,),.55),trigger=torch.tensor([10,10,10]),
                table_pose=torch.tensor([0.,0.,0.,0.,0.,0.,1.])[None].repeat(n,1),
                initial_clearance=torch.full((n,),.05),future_clearance=torch.full((n,10),.05),
                initial_key_positions=keys[:,0],future_key_positions=keys,initial_key_velocities=torch.zeros(n,6,3),
                future_key_velocities=torch.zeros(n,10,6,3),native_observation=native,
                future_native_observation=native.clone(),
                outcome=dict(initially_clear=torch.ones(n,dtype=torch.bool),retained_clear=torch.ones(n,dtype=torch.bool),
                             supported_change_mm=torch.zeros(n)))


def run(output):
    import torch
    from audit_contact_geometry_source import audit_record
    torch.set_num_threads(2);begin=time.monotonic()
    if output.exists():raise ValueError('unique output required')
    class MockGeometry:
        def clearance(self,poses,tables):return poses[:,2]-.55
    b=fixture(torch);good=audit_record(b,MockGeometry())
    rejected=[]
    mutations={
        'pd_target':lambda x:x['actual_pd_targets'].__setitem__((2,0,6),x['actual_pd_targets'][2,0,6]+.001),
        'merged_probability':lambda x:x['propensity'].__setitem__(0,.1),
        'allocation':lambda x:x['allocation'].__setitem__(0,9),
        'reference_weights':lambda x:x['candidate_weights'].__setitem__((0,0,0,slice(None)),torch.tensor([1.,0,0,0,0,0])),
        'last_post_position':lambda x:x['future_key_positions'].__setitem__((0,9,0,0),x['future_key_positions'][0,9,0,0]+.001),
        'last_post_velocity':lambda x:x['future_key_velocities'].__setitem__((0,9,0,0),.001),
        'contact_proxy':lambda x:x['future_contact'].__setitem__((0,0,0),False),
        'mesh_label':lambda x:x['future_clearance'].__setitem__((0,0),.01),
        'cross_reset':lambda x:x['future_done'].__setitem__((0,9),True),
        'group_split':lambda x:x['split_group_bucket'].__setitem__(0,-1),
    }
    for name,mutation in mutations.items():
        broken=copy.deepcopy(b);mutation(broken)
        try:audit_record(broken,MockGeometry())
        except ValueError:rejected.append(name)
        else:raise ValueError('corrupt '+name+' accepted')
    result=dict(run_status='COMPLETED',engineering_passed=True,good_record=good,rejected_corruptions=rejected,
                elapsed_seconds=time.monotonic()-begin,
                input_sha256={str(Path(__file__).resolve()):sha(Path(__file__)),str(ROOT/'scripts/audit_contact_geometry_source.py'):sha(ROOT/'scripts/audit_contact_geometry_source.py'),str(ROOT/'src/task/CmResidual/contact_geometry_actions.py'):sha(ROOT/'src/task/CmResidual/contact_geometry_actions.py')},
                device_reason='CPU tiny synthetic record checks; no models/simulator',
                scope='in-memory synthetic unit fixtures; not native runtime, real transitions or Cm utility evidence')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
