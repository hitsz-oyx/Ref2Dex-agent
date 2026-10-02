#!/usr/bin/env python3
"""Independent physical and frozen-planner replay for generated-action source."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha, gpu_admission


def rotation(q, torch):
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)


def audit_record(b, geometry):
    import torch
    n = len(b['state'])
    if b['schema'] != 'ref2dex.structured_contact_source.v1' or not n:
        raise ValueError('source schema')
    if not b['frozen_experts'] or not b['cm_used'] or not b['optimizer_used'] or b['model_training'] or not b['assignment_after_observation']:
        raise ValueError('frozen randomized source contract')
    if b['future_done'].any() or b['future_state'].shape != (n,10,49):
        raise ValueError('complete no-reset windows required')
    for key, value in b.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('nonfinite '+key)
    if abs(b['control_dt_seconds']-1/30) > 1e-8 or (b['mass_kg'] <= 0).any():
        raise ValueError('native physical contract')
    if not torch.equal(b['history'][:,-1,:49],b['state']):
        raise ValueError('prehistory timing')
    mapping = torch.tensor([0,1,2,3,4,5,6,7,0,1])
    if b['allocation_to_option'] != mapping.tolist() or (b['allocation'] < 0).any() or (b['allocation'] > 9).any():
        raise ValueError('allocation map')
    if not torch.equal(mapping[b['allocation']],b['assignment']):
        raise ValueError('allocated/actual option drift')
    probability = b['allocation_probabilities']
    if probability.shape != (n,10) or not torch.allclose(probability,torch.full_like(probability,.1),atol=1e-7):
        raise ValueError('uniform allocation contract')
    merged = torch.stack([probability[:,mapping == k].sum(-1) for k in range(8)],-1)
    if not torch.allclose(merged[torch.arange(n),b['assignment']],b['propensity'],atol=1e-7):
        raise ValueError('actual merged propensity')
    if b['split_group_seed'] != 12651:
        raise ValueError('fixed group split')
    buckets = torch.tensor([int(hashlib.sha256(f'12651/{int(i)}/{int(j)}'.encode()).hexdigest()[:8],16)%100
                            for i,j in zip(b['motion_id'],b['start_frame'])])
    if not torch.equal(buckets,b['split_group_bucket']):
        raise ValueError('group split drift')
    weights, bank = b['candidate_weights'],b['expert_bank']
    if weights.shape != (n,8,6,6) or bank.shape != (n,10,6,18):
        raise ValueError('candidate dimensions')
    if (weights < 0).any() or not torch.allclose(weights.sum(-1),torch.ones(n,8,6),atol=2e-6):
        raise ValueError('nonconvex weights')
    reference = torch.zeros(n,2,6,6);reference[:,0,:,4]=1;reference[:,1,:,1]=1
    if not torch.equal(weights[:,:2],reference):
        raise ValueError('reference programme coefficient drift')
    if (bank.abs() > 1+1e-6).any():
        raise ValueError('expert command domain')
    # Independent coefficient assembly, without importing the producer's helper.
    assembled = bank[:,:,1,None].expand(-1,-1,8,-1).clone()
    blocks = [(0,3),(6,8),(8,10),(10,12),(12,14),(14,18)]
    for block,(start,stop) in enumerate(blocks):
        assembled[...,start:stop] = torch.einsum('nke,nted->ntkd',weights[:,:,block],bank[...,start:stop])
    assembled[:,:,0] = bank[:,:,4]
    assembled[:,:,1] = bank[:,:,1]
    indices = b['assignment'][:,None,None,None].expand(-1,10,1,18)
    feedback = assembled.gather(2,indices).squeeze(2)
    preq = torch.cat((b['state'][:,None,:18],b['future_state'][:,:-1,:18]),1)
    scale, offset = b['pd_scale'],b['pd_offset']
    if (scale[:6] <= 0).any():
        raise ValueError('invalid native scale')
    executed = assembled.clone()
    hold_raw = ((b['hold_target'][:,None,3:6]-preq[:,:,3:6]-offset[3:6])/scale[3:6]).clamp(-1,1)
    executed[:,:,1:,3:6] = hold_raw[:,:,None]
    actual = executed.gather(2,indices).squeeze(2)
    errors = dict(feedback=float((feedback-b['feedback_action']).abs().max()),
                  actual_command=float((actual-b['actual_action']).abs().max()),
                  initial_candidate=float((executed[:,0]-b['candidate_actions']).abs().max()))
    raw = actual.clone();raw[...,6:] = (1+raw[...,6:])*.5
    targets = offset+scale*raw;targets[...,:6] += preq[...,:6]
    for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        targets[...,dst] = targets[...,src]*ratio
    errors['native_pd'] = float((targets-b['actual_pd_targets']).abs().max())
    holding = b['assignment'] != 0
    if holding.any():
        errors['fixed_rotation'] = float((targets[holding,:,3:6]-b['hold_target'][holding,None,3:6]).abs().max())
    for channels in [slice(0,3),slice(6,18)]:
        lower,upper = bank.min(2).values[...,channels],bank.max(2).values[...,channels]
        if (actual[...,channels]<lower-2e-6).any() or (actual[...,channels]>upper+2e-6).any():
            raise ValueError('executed mixture domain')
    weight = b['mass_kg'][:,None]*b['gravity_magnitude']
    ratios = torch.stack((b['future_hand_force'].norm(dim=-1).amax(-1),b['future_object_force'].norm(dim=-1)),-1)/weight[...,None]
    if not torch.equal(ratios>.1,b['future_contact']):
        raise ValueError('raw force proxy labels')
    errors['force_ratio_relative'] = float(((ratios-b['future_force_ratio']).abs()/ratios.abs().clamp_min(1)).max())
    pre_ratios = torch.stack((b['initial_hand_force'].norm(dim=-1).amax(-1),b['initial_object_force'].norm(dim=-1)),-1)/weight[:,0,None]
    if not torch.equal(pre_ratios>.1,b['history'][:,-1,49:51].bool()):
        raise ValueError('pre force proxy timing')
    if not (pre_ratios>.1).all() or (b['state'][:,38]-b['rest_z']<.005).any() or (b['trigger']<10).any():
        raise ValueError('current trigger contract')
    states = torch.cat((b['state'][:,None],b['future_state']),1)
    poses = states[:,:,36:49].flatten(0,1)
    tables = b['table_pose'][:,None].expand(-1,11,-1).flatten(0,1)
    clearance = torch.cat([geometry.clearance(poses[start:start+96],tables[start:start+96]).cpu()
                           for start in range(0,len(poses),96)]).reshape(n,11)
    saved_clearance = torch.cat((b['initial_clearance'][:,None],b['future_clearance']),1)
    errors['mesh_clearance'] = float((clearance-saved_clearance).abs().max())
    clear = (saved_clearance[:,0]>=.002)&(b['state'][:,38]-b['rest_z']>=.03)
    support = (b['future_clearance'][:,-3:]>=.002).all(-1)&b['future_contact'][:,-3:].all(-1).all(-1)
    score = ((b['future_state'][:,-3:,38].amin(-1)-b['rest_z']).clamp_min(0)*support-(b['state'][:,38]-b['rest_z']).clamp_min(0))*1000
    if not torch.equal(clear,b['outcome']['initially_clear']) or not torch.equal(support,b['outcome']['retained_clear']):
        raise ValueError('H10 physical labels')
    errors['supported_score_mm'] = float((score-b['outcome']['supported_change_mm']).abs().max())
    # Both sides of every actual transition, including the last post-state.
    pre = states[:,:-1]
    pos = torch.cat((b['initial_key_positions'][:,None],b['future_key_positions'][:,:-1]),1)
    vel = torch.cat((b['initial_key_velocities'][:,None],b['future_key_velocities'][:,:-1]),1)
    for prefix,physical,keypos,keyvel in [('pre',pre,pos,vel),('post',b['future_state'],b['future_key_positions'],b['future_key_velocities'])]:
        obs = b['native_observation'] if prefix=='pre' else b['future_native_observation']
        if obs.shape != (n,10,1442):
            raise ValueError('full pre/post native observation contract')
        x,z = obs[...,649:652],obs[...,652:655]
        local_rotation = torch.stack((x,torch.cross(z,x,dim=-1),z),-1)
        world_rotation = rotation(physical[...,39:43],torch)
        observed_pos = obs[...,406:454].reshape(n,10,16,3)[:,:,[0,3,6,9,12,15]]-obs[...,646:649,None].transpose(-1,-2)
        observed_vel = obs[...,156:204].reshape(n,10,16,3)[:,:,[0,3,6,9,12,15]]-obs[...,655:658,None].transpose(-1,-2)
        native_pos = torch.einsum('ntij,ntkj->ntki',local_rotation.transpose(-1,-2),observed_pos)
        native_vel = torch.einsum('ntij,ntkj->ntki',local_rotation.transpose(-1,-2),observed_vel)
        world_pos = torch.einsum('ntij,ntkj->ntki',world_rotation.transpose(-1,-2),keypos-physical[...,36:39,None].transpose(-1,-2))
        world_vel = torch.einsum('ntij,ntkj->ntki',world_rotation.transpose(-1,-2),keyvel-physical[...,43:46,None].transpose(-1,-2))
        errors[prefix+'_relative_position'] = float((native_pos-world_pos).abs().max())
        errors[prefix+'_relative_velocity'] = float((native_vel-world_vel).abs().max())
    limits = {key:(2e-6 if key=='mesh_clearance' else 2e-5) for key in errors}
    if any(errors[key]>limits[key] for key in errors):
        raise ValueError('record reconstruction: '+json.dumps(errors))
    return dict(rows=n,steps=n*10,actual_propensity_verified=True,complete_labels=True,
                pre_and_post_geometry_verified=True,errors=errors)


def audit_planning(b, planning, checkpoint, device):
    import torch
    from src.task.CmResidual.structured_contact_actions import FrozenStructuredActionGenerator, live_inputs
    from src.task.CmResidual.structured_contact_consequence import current_features, normalize
    from src.task.CmResidual.paired_evaluation import fingerprint
    if planning['schema'] != 'ref2dex.structured_contact_planning.v1' or planning['model_training']:
        raise ValueError('planning contract')
    if planning['checkpoint_sha256'] != sha(checkpoint) or b['generator_checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('frozen generator checkpoint')
    expected_config = dict(steps=32,lr=.15,clip=1.,risk_penalty_mm=100.,
                           disagreement_multiplier=1.645,allowed_loss_difference=.02,
                           allowed_support_drop=.05,hard_feasibility=True,feature_clip=8.)
    if planning['optimization'] != expected_config:
        raise ValueError('planning configuration drift')
    generator = FrozenStructuredActionGenerator(checkpoint,device)
    before = fingerprint([{mode:[m.state_dict() for m in models]} for mode,models in generator.models.items()])
    if before != planning['model_fingerprint_before'] or before != planning['model_fingerprint_after']:
        raise ValueError('frozen parameter fingerprint')
    lookup = {(int(env),int(tick)):i for i,(env,tick) in enumerate(zip(b['env_id'],b['trigger']))}
    if len(lookup) != len(b['state']):
        raise ValueError('duplicate native window identity')
    proposer = torch.Generator(device=device).manual_seed(b['assignment_seed']+30000)
    allocator = torch.Generator(device=device).manual_seed(b['assignment_seed'])
    errors = {}; covered = set(); reports = []
    def compare(key,actual,expected,atol=2e-5,rtol=2e-6):
        difference = float((actual.detach().cpu()-expected.detach().cpu()).abs().max())
        errors[key] = max(errors.get(key,0.),difference)
        if not torch.allclose(actual.detach().cpu(),expected.detach().cpu(),atol=atol,rtol=rtol):
            raise ValueError('planning replay '+key+': '+str(difference))
    for batch in planning['batches']:
        indices = [lookup[(int(env),batch['tick'])] for env in batch['env_id']]
        if covered.intersection(indices):
            raise ValueError('reused planning row')
        covered.update(indices)
        n = len(indices)
        cpu = {k:v[indices] for k,v in b.items() if torch.is_tensor(v) and v.ndim and len(v)==len(b['state'])}
        c = {k:v.to(device) for k,v in cpu.items()}
        offset,scale = b['pd_offset'].to(device),b['pd_scale'].to(device)
        uniform = torch.rand(n,6,6,6,device=device,generator=proposer)
        positive = (-uniform.clamp_min(torch.finfo(uniform.dtype).tiny).log()).pow(3)
        compare('random_proposal',positive/positive.sum(-1,keepdim=True),batch['random_weights'][:,2:],atol=0,rtol=0)
        draw = torch.randint(10,(n,),device=device,generator=allocator).cpu()
        if not torch.equal(draw,batch['allocation']) or not torch.equal(draw,cpu['allocation']):
            raise ValueError('allocation RNG replay')
        compare('saved_candidate_weights',cpu['candidate_weights'],batch['weights'],atol=0,rtol=0)
        compare('unmodified_random_programmes',cpu['candidate_weights'][:,5:],batch['random_weights'][:,5:],atol=0,rtol=0)
        inputs = live_inputs(c['history'],c['native_observation'][:,0],c['initial_hand_force'],
            c['initial_object_force'],c['mass_kg'],b['gravity_magnitude'],c['initial_clearance'],c['rest_z'])
        for key,value in inputs.items():
            if torch.is_tensor(value):compare('pre_only_'+key,value,batch['inputs'][key])
            elif value!=batch['inputs'][key]:raise ValueError('pre-only scalar input drift')
        mode_reports = {}
        for option,mode in enumerate(('cm','direct_score','shuffled'),2):
            bank = c['expert_bank'][:,0];position=c['state'][:,:18];anchor=c['hold_target']
            with torch.no_grad():
                weights,info = generator.optimize(mode,bank,anchor,position,offset,scale,inputs)
            compare(mode+'_generated_weights',weights,c['candidate_weights'][:,option])
            for key in ('trace','predicted_score_mm','predicted_loss','predicted_support','initial_command','initial_pd_targets'):
                compare(mode+'_'+key,info[key],batch['modes'][mode][key])
            compare(mode+'_executed_first_candidate',info['initial_command'],c['candidate_actions'][:,option])
            if not torch.equal(info['changed_reference'].cpu(),batch['modes'][mode]['changed_reference']):
                raise ValueError('planner change flag')
            if info['optimize_steps'] != 32 or batch['modes'][mode]['optimize_steps'] != 32:
                raise ValueError('bounded optimizer steps')
            # Replay final predictions with the full frozen network, without cached heads.
            original = current_features(**inputs,bank=bank,cup_action=c['candidate_actions'][:,1],
                candidate_action=info['initial_command'],weights=weights,law=torch.ones(n,device=device),
                offset=offset,scale=scale)
            normalized = normalize(original,generator.norm)
            with torch.no_grad():
                full = [model(**normalized) for model in generator.models[mode]]
                compare(mode+'_full_model_score',torch.stack([v['supported_height'] for v in full])*10,info['predicted_score_mm'])
                if mode=='direct_score':full=[model(**normalized) for model in generator.models['state_only']]
                compare(mode+'_full_model_loss',torch.stack([v['loss_probability'] for v in full]),info['predicted_loss'])
                compare(mode+'_full_model_support',torch.stack([v['support_probability'] for v in full]),info['predicted_support'])
            changed=info['changed_reference']
            delta=info['predicted_score_mm']-info['reference_score_mm']
            mean=delta.mean(0);std=((delta-mean[None]).square().mean(0)+1e-6).sqrt()
            risk=(info['predicted_loss']-info['reference_loss']).mean(0)
            support=(info['predicted_support']-info['reference_support']).mean(0)
            if changed.any() and ((mean[changed]-1.645*std[changed]<=0).any() or (risk[changed]>.02+1e-6).any()
                    or (support[changed]<-.05-1e-6).any() or info['context_ood'][changed].any() or info['candidate_ood'][changed].any()):
                raise ValueError('selected structured program feasibility')
            expected_risk='state_only_static' if mode=='direct_score' else 'action_conditioned_joint_model'
            if info['direct_score_risk_source']!=expected_risk or batch['modes'][mode]['direct_score_risk_source']!=expected_risk:
                raise ValueError('risk head semantics')
            mode_reports[mode] = int(info['changed_reference'].sum())
        reports.append(dict(tick=batch['tick'],rows=n,changed_reference=mode_reports))
    if len(covered) != len(b['state']):
        raise ValueError('planning does not cover all executed windows')
    after = fingerprint([{mode:[m.state_dict() for m in models]} for mode,models in generator.models.items()])
    if before != after or any(p.requires_grad for models in generator.models.values() for m in models for p in m.parameters()):
        raise ValueError('model parameters changed')
    return dict(rows=len(covered),batches=reports,errors=errors,all_current_inputs_reconstructed=True,
                allocation_and_proposal_rng_replayed=True,full_frozen_network_verified=True,
                optimizer_inputs_only=True,model_fingerprint=after)


def run(args):
    begin = time.monotonic()
    if args.output.exists():
        raise ValueError('unique audit output required')
    manifest = json.loads((args.run/'run_manifest.json').read_text())
    if manifest['run_status'] != 'COMPLETED' or manifest.get('child_exit_code') != 0:
        raise ValueError('authoritative terminal native run required')
    if any(sha(Path(k))!=v for k,v in manifest['input_sha256'].items()):
        raise ValueError('source drift')
    admission = gpu_admission(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = admission['uuid']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from src.task.CmResidual.executable_contact_options import TableClearance,obj_vertices
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    assets = ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    parent_geometry = TableClearance(obj_vertices(assets/'objects/airplane/airplane.obj','cuda'),
                                    obj_vertices(assets/'objects/table/table.obj','cuda'))
    class DeviceGeometry:
        def clearance(self,object_pose,table_pose):
            return parent_geometry.clearance(object_pose.to('cuda'),table_pose.to('cuda'))
    audits=[]
    for phase in manifest['phases']:
        directory=Path(phase['directory']);path=directory/'records.pt';plan_path=directory/'planning.pt'
        if phase['run_status']!='COMPLETED' or sha(path)!=phase['result']['record_sha256']:
            raise ValueError('terminal record/hash')
        if sha(plan_path)!=phase['result']['planning_sha256']:
            raise ValueError('planning hash')
        if phase['result']['saved_observation_expert_replay_max_error']>2e-5:
            raise ValueError('source six-expert replay failed')
        b=torch.load(path,map_location='cpu',weights_only=False)
        planning=torch.load(plan_path,map_location='cpu',weights_only=False)
        if b['planning_sha256']!=sha(plan_path):
            raise ValueError('record planning link')
        audits.append(dict(seed=phase['seed'],physical=audit_record(b,DeviceGeometry()),
            planning=audit_planning(b,planning,Path(planning['checkpoint']),'cuda')))
        if time.monotonic()-begin>360:
            raise TimeoutError('6min structured source audit budget')
    if not audits:
        raise ValueError('no completed phases')
    result=dict(run_status='COMPLETED',phases=audits,gpu=admission,elapsed_seconds=time.monotonic()-begin,
                run_manifest_sha256=sha(args.run/'run_manifest.json'),auditor_sha256=sha(Path(__file__)),
                scope='actual generated command, current planning and every H10 transition; no utility conclusion')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(run_status='COMPLETED',phases=len(audits),elapsed_seconds=result['elapsed_seconds'],
                         rows=sum(x['physical']['rows'] for x in audits))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    run(p.parse_args())
