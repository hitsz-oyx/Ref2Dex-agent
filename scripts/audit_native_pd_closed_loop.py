#!/usr/bin/env python3
"""Independent GPU replay of actual controller choices, native PD and labels."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    begin=time.monotonic();m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or args.output.exists():raise ValueError('terminal run/unique output required')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('source/input drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.executable_contact_options import TableClearance,obj_vertices,INDEPENDENT,COUPLINGS
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts
    from src.task.CmResidual.native_pd_selector import FrozenNativePDSelector
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    geometry=TableClearance(obj_vertices(asset/'objects/airplane/airplane.obj','cuda'),obj_vertices(asset/'objects/table/table.obj','cuda'))
    checkpoint=ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-native-pd-policy-control-r1/native_pd_controls.pt';selector=FrozenNativePDSelector(checkpoint,'cuda');audits=[]
    for p in m['phases']:
        path=Path(p['directory'])/'records.pt'
        if p['run_status']!='COMPLETED' or sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
        b=torch.load(path,map_location='cpu',weights_only=False);n=len(b['assignment']);row=torch.arange(n)
        if b['schema']!='ref2dex.native_pd_closed_loop.v1' or b['future_done'].any() or b['future_state'].shape!=(n,10,49) or b['optimizer_used']:raise ValueError('actual frozen complete labels required')
        if b['checkpoint_sha256']!=sha(checkpoint):raise ValueError('checkpoint drift')
        if not all(int(hashlib.sha256(f'9851/{int(i)}/{int(j)}'.encode()).hexdigest()[:8],16)%100>=70 for i,j in zip(b['motion_id'],b['start_frame'])):raise ValueError('fit/cal initial group leakage')
        for key in ['state','history','future_state','actual_action','actual_pd_targets','native_observation','decision_history','rotation_anchor','future_hand_force','future_object_force']:
            if not torch.isfinite(b[key]).all():raise ValueError('nonfinite '+key)
        if not torch.equal(b['history'][:,-1,:49],b['state']) or not torch.equal(b['decision_history'][:,0],b['history']):raise ValueError('trigger history mismatch')
        q=torch.arange(10);selected=b['recommendations'][row[:,None],q[None,:],b['assignment'][:,None]]
        if not torch.equal(selected,b['program']):raise ValueError('owner recommendations not actually executed')
        if not torch.allclose(b['allocation_probabilities'],torch.full((n,10),.2)) or not torch.allclose(b['propensity'],torch.full((n,),.2)):raise ValueError('distinct policy propensity mismatch')
        prior_state=torch.cat((b['state'][:,None],b['future_state'][:,:-1]),1)
        if not torch.equal(b['decision_history'][:,:,-1,:49],prior_state):raise ValueError('future state in model input')
        if not torch.equal(b['decision_history'][:,1:,:-1],b['decision_history'][:,:-1,1:]):raise ValueError('history shifts not actual two new frames')
        for t in range(1,10):
            offsets=[t-1]
            last2=torch.cat((b['future_state'][:,offsets],b['future_contact'][:,offsets].float(),b['actual_action'][:,offsets]),-1)
            if not torch.equal(b['decision_history'][:,t,-1:],last2):raise ValueError('model history has wrong previous command/contact/state')
        pred_error=0.;replay_choices=[]
        history=b['decision_history'].reshape(-1,10,69);native=b['native_observation'].reshape(-1,b['native_observation'].shape[-1]);rest=b['rest_z'][:,None].expand(-1,10).reshape(-1)
        pre_hand=torch.cat((b['initial_hand_force'][:,None],b['future_hand_force'][:,:-1]),1).reshape(-1,b['future_hand_force'].shape[-2],3)
        pre_object=torch.cat((b['initial_object_force'][:,None],b['future_object_force'][:,:-1]),1).reshape(-1,3)
        pre_clear=torch.cat((b['initial_clearance'][:,None],b['future_clearance'][:,:-1]),1).reshape(-1)
        pre_mass=b['mass_kg'][:,None].expand(-1,10).reshape(-1);motors=b['candidate_pd_targets'].reshape(-1,8,18)
        for i,mode in enumerate(b['policy_names'][:3]):
            choices=[];saved={key:value[:,:,i].reshape(-1,8) for key,value in b['diagnostics'].items()}
            for offset in range(0,len(history),96):
                choice,d=selector.choose(mode,history[offset:offset+96].cuda(),native[offset:offset+96].cuda(),rest[offset:offset+96].cuda(),motors[offset:offset+96].cuda(),pre_hand[offset:offset+96].cuda(),pre_object[offset:offset+96].cuda(),pre_mass[offset:offset+96].cuda(),b['gravity_magnitude'],pre_clear[offset:offset+96].cuda());choices.append(choice.cpu())
                for key in saved:pred_error=max(pred_error,float((d[key].cpu()-saved[key][offset:offset+96]).abs().max()))
                if not torch.equal(d['ood'].cpu(),b['ood'][:,:,i].reshape(-1)[offset:offset+96]):raise ValueError('OOD replay mismatch')
            choices=torch.cat(choices).reshape(n,10)
            if not torch.equal(choices,b['recommendations'][:,:,i]):raise ValueError('frozen selector replay mismatch')
            replay_choices.append(choices)
        if pred_error>2e-3:raise ValueError('frozen forecast replay drift')
        if not torch.equal(b['recommendations'][:,:,3],torch.full((n,10),4)) or not torch.equal(b['recommendations'][:,:,4],torch.full((n,10),7)):raise ValueError('fixed policy drift')
        scale=b['pd_scale'];offset=b['pd_offset'];raw=b['actual_action'].clone();raw[:,:,6:]=(1+raw[:,:,6:])/2
        targets=offset+scale*raw;targets[:,:,:6]+=prior_state[:,:,:6]
        for dst,src,ratio in COUPLINGS:targets[:,:,dst]=targets[:,:,src]*ratio
        pd_error=float((targets-b['actual_pd_targets']).abs().max())
        if pd_error>2e-5:raise ValueError('native independent/coupled PD mismatch')
        if not torch.allclose(b['candidate_pd_targets'][row[:,None],torch.arange(10)[None,:],b['program']],b['actual_pd_targets'],atol=2e-5,rtol=1e-5):raise ValueError('scored motor command differs from native execution')
        hold=b['program']>=6
        if not torch.allclose(b['actual_pd_targets'][hold][:,3:6],b['rotation_anchor'][hold][:,3:6],atol=2e-5,rtol=1e-5):raise ValueError('rotation hold wrong')
        if not torch.equal(b['rotation_anchor'][:,:,:6],prior_state[:,:,:6]):raise ValueError('reobserve/anchor duration mismatch')
        if not torch.equal(b['actual_action'][:,:,:3],b['feedback_action'][:,:,:3]) or not torch.equal(b['actual_action'][:,:,6:],b['feedback_action'][:,:,6:]) or p['result']['saved_observation_expert_replay_max_error']>2e-5 or p['result']['countercommand_replay_max_error']>2e-5:raise ValueError('independent expert replay/finger/XYZ mismatch')
        for name in ['base','fixed']:
            delta=((b['actual_pd_targets']-b[name+'_pd_targets'])[:,:,list(INDEPENDENT)].abs()>2e-5).any(-1)
            if not torch.equal(delta,b['changed_'+name]):raise ValueError('physical changed command count mismatch')
        if b['changed_base'][b['program']==4].any() or b['changed_fixed'][b['program']==7].any():raise ValueError('identity base/fixed command mismatch')
        if not torch.allclose(b['fixed_pd_targets'][:,:,3:6],b['fixed_rotation_anchor'][:,:,3:6],atol=2e-5,rtol=1e-5):raise ValueError('fixed rotation target mismatch')
        hf=b['future_hand_force'].reshape(-1,b['future_hand_force'].shape[-2],3).cuda();of=b['future_object_force'].reshape(-1,3).cuda();masses=b['mass_kg'][:,None].expand(-1,10).reshape(-1).cuda()
        bits,ratio=weight_normalized_contacts(hf,of,masses,b['gravity_magnitude']);force_error=float((ratio.cpu().reshape(n,10,2)-b['future_force_ratio']).abs().max())
        if not torch.equal(bits.cpu().reshape(n,10,2),b['future_contact']) or force_error>2e-5:raise ValueError('raw force normalization mismatch')
        legacy=torch.stack((hf.norm(dim=-1).amax(-1)>.1,of.norm(dim=-1)>.1),-1).cpu().reshape(n,10,2)
        if not torch.equal(legacy,b['legacy_future_contact']):raise ValueError('legacy force drift')
        initial,_=weight_normalized_contacts(b['initial_hand_force'].cuda(),b['initial_object_force'].cuda(),b['mass_kg'].cuda(),b['gravity_magnitude'])
        if not torch.equal(initial.cpu(),b['history'][:,-1,49:51].bool()):raise ValueError('initial proxy drift')
        poses=torch.cat((b['state'][:,None,36:49],b['future_state'][:,:,36:49]),1).reshape(-1,13);tables=b['table_pose'][:,None].expand(-1,11,-1).reshape(-1,7);values=[]
        for offset in range(0,len(poses),256):values.append(geometry.clearance(poses[offset:offset+256].cuda(),tables[offset:offset+256].cuda()).cpu())
        clr=torch.cat(values).reshape(n,11);geometry_error=float((clr-torch.cat((b['initial_clearance'][:,None],b['future_clearance']),1)).abs().max())
        if geometry_error>2e-6:raise ValueError('geometry replay drift')
        if not ((b['initial_clearance']>=.002)&(b['state'][:,38]-b['rest_z']>=.03)).all():raise ValueError('clear trigger not pre-action')
        pair=b['future_contact'].all(-1);retain=pair[:,-3:].all(-1)&(b['future_clearance'][:,-3:]>=.002).all(-1)
        score=((b['future_state'][:,-3:,38].amin(-1)-b['rest_z']).clamp_min(0)*retain-(b['state'][:,38]-b['rest_z']).clamp_min(0))*1000
        if not torch.allclose(score,b['outcome']['score_mm'],atol=2e-5) or not torch.equal(retain,b['outcome']['retained']) or not torch.equal((b['future_clearance']<.002).any(-1),b['outcome']['lost_clearance']) or not torch.equal(pair[:,-3:].all(-1),b['outcome']['joint_last3']):raise ValueError('local outcome replay mismatch')
        audits.append(dict(seed=p['seed'],rows=n,policies=torch.bincount(b['assignment'],minlength=5).tolist(),model_decisions_replayed=n*10*3,prediction_max_error=pred_error,native_pd_max_error=pd_error,geometry_max_error=geometry_error,force_ratio_max_error=force_error,physical_changed_base_windows=int(b['changed_base'].any(-1).sum()),physical_changed_fixed_windows=int(b['changed_fixed'].any(-1).sum()),all_contracts_passed=True))
    result=dict(run_status='COMPLETED',phases=audits,input_hashes_unchanged=True,checkpoint_sha256=sha(checkpoint),gpu=admission,elapsed_seconds=time.monotonic()-begin,scope='replay of frozen selector/real observed histories/native PD/raw forces/source mesh/known controller propensities; no true counterfactual paths or final grasp claim')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
