#!/usr/bin/env python3
"""HF15 terminal raw-force/feedback/native-PD/full-mesh replay on GPU."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    m=json.loads((args.run/'run_manifest.json').read_text())
    if not all(sha(Path(k))==v for k,v in m['input_sha256'].items()):raise ValueError('source/input drift')
    if m['run_status']!='COMPLETED':raise ValueError('terminal data required')
    if args.output.exists():raise ValueError('audit exists')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.executable_contact_options import TableClearance,obj_vertices
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts
    torch.set_num_threads(2);asset=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf'
    geometry=TableClearance(obj_vertices(asset/'objects/airplane/airplane.obj','cuda'),obj_vertices(asset/'objects/table/table.obj','cuda'))
    audits=[]
    for p in m['phases']:
        d=Path(p['directory']);r=p['result']
        if p['run_status']!='COMPLETED' or sha(d/'records.pt')!=r['record_sha256']:raise ValueError('terminal record/hash contract')
        b=torch.load(d/'records.pt',weights_only=False,map_location='cpu');n=len(b['assignment']);row=torch.arange(n)
        if b['future_done'].any() or b['future_state'].shape!=(n,10,49):raise ValueError('complete label contract')
        for name in ['state','history','future_state','actual_action','actual_pd_targets','initial_clearance','future_clearance']:
            if not torch.isfinite(b[name]).all():raise ValueError('nonfinite '+name)
        if not torch.equal(b['history'][:,-1,:49],b['state']):raise ValueError('prehistory drift')
        if not torch.equal(b['actual_action'][:,0],b['candidate_actions'][row,b['assignment']]):raise ValueError('first option command mismatch')
        allocations=b.get('allocation_probabilities',torch.full((n,9),1/9))
        if not torch.allclose(allocations.sum(-1),torch.ones(n)):raise ValueError('allocation vector not normalized')
        actual_p=allocations[row,b['assignment']]+torch.where(b['assignment']==4,allocations[:,8],torch.zeros(n))
        if not torch.allclose(actual_p,b['propensity'],atol=1e-7):raise ValueError('actual merged propensity mismatch')
        selected=torch.where(b['allocation']==8,4,b['allocation'])
        if not torch.equal(selected,b['assignment']):raise ValueError('allocation/option mismatch')
        hold=b['assignment']>=6
        if hold.any() and not torch.allclose(b['actual_pd_targets'][hold,:,3:6],b['hold_target'][hold,None,3:6].expand(-1,10,-1),atol=2e-5,rtol=1e-5):raise ValueError('anchored wrist PD drift')
        if b['schema']!='ref2dex.orientation_feedback_options.v1':raise ValueError('new force schema required')
        if not torch.equal(b['actual_action'][:,:,:3],b['feedback_action'][:,:,:3]):raise ValueError('expert translation channel changed')
        if not torch.equal(b['actual_action'][:,:,6:],b['feedback_action'][:,:,6:]):raise ValueError('feedback fingers changed')
        if r['saved_observation_expert_replay_max_error']>2e-5:raise ValueError('native expert replay failed')
        hf=b['future_hand_force'].reshape(-1,b['future_hand_force'].shape[-2],3).cuda();of=b['future_object_force'].reshape(-1,3).cuda()
        masses=b['mass_kg'][:,None].expand(-1,10).reshape(-1).cuda()
        bits,ratio=weight_normalized_contacts(hf,of,masses,b['gravity_magnitude'])
        if not torch.equal(bits.cpu().reshape(n,10,2),b['future_contact']):raise ValueError('normalized force bits mismatch')
        force_error=float((ratio.cpu().reshape(n,10,2)-b['future_force_ratio']).abs().max())
        if force_error>2e-5:raise ValueError('raw force ratio mismatch')
        legacy=torch.stack((hf.norm(dim=-1).amax(-1)>.1,of.norm(dim=-1)>.1),-1).cpu().reshape(n,10,2)
        if not torch.equal(legacy,b['legacy_future_contact']):raise ValueError('legacy force audit mismatch')
        prebits,_=weight_normalized_contacts(b['initial_hand_force'].cuda(),b['initial_object_force'].cuda(),b['mass_kg'].cuda(),b['gravity_magnitude'])
        if not torch.equal(prebits.cpu(),b['history'][:,-1,49:51].bool()):raise ValueError('initial normalized proxy mismatch')
        previous=torch.cat((b['state'][:,None,:18],b['future_state'][:,:-1,:18]),1)
        scale=torch.tensor([1.,1.,1.,3.141592653589793,3.141592653589793,3.141592653589793])
        expected=previous[:,:,:6]+b['actual_action'][:,:,:6]*scale
        wrist_error=float((expected-b['actual_pd_targets'][:,:,:6]).abs().max())
        if wrist_error>2e-5:raise ValueError('native relative-wrist mapping mismatch')
        poses=torch.cat((b['state'][:,36:49,None].transpose(1,2),b['future_state'][:,:,36:49]),1).reshape(-1,13)
        tables=b['table_pose'][:,None].expand(-1,11,-1).reshape(-1,7)
        values=[]
        for offset in range(0,len(poses),256):values.append(geometry.clearance(poses[offset:offset+256].cuda(),tables[offset:offset+256].cuda()).cpu())
        clearance=torch.cat(values).reshape(n,11);saved=torch.cat((b['initial_clearance'][:,None],b['future_clearance']),1)
        error=float((clearance-saved).abs().max())
        if error>2e-6:raise ValueError('source-mesh/table-plane replay mismatch')
        rare=(b['initial_clearance']>=.002)&(b['state'][:,38]-b['rest_z']>=.03)
        if not torch.equal(rare,b['outcome']['initially_clear']):raise ValueError('prestate stratum mismatch')
        if 'sampling_cohort' in b and not rare[b['sampling_cohort']==0].all():raise ValueError('clear-first cohort contract')
        varying=(b['actual_action'][:,1:]!=b['actual_action'][:,:1]).any(-1).any(-1)
        audits.append(dict(seed=p['seed'],rows=n,initially_clear=int(rare.sum()),hold_windows=int(hold.sum()),
                           varying_feedback_expert_windows=int((varying&~hold).sum()),varying_anchored_finger_windows=int((hold&(b['actual_action'][:,1:,6:]!=b['actual_action'][:,:1,6:]).any(-1).any(-1)).sum()),raw_force_ratio_error=force_error,saved_observation_expert_replay_max_error=r['saved_observation_expert_replay_max_error'],wrist_mapping_max_error=wrist_error,geometry_max_error=error,
                           complete_labels=True,wrist_anchor_target=True,actual_propensity_verified=True,geometry_replayed=True))
    args.output.write_text(json.dumps(dict(run_status='COMPLETED',gpu=admission,phases=audits,scope='actual native-PD and full source-mesh replay; no model training/inference, no utility conclusion'),indent=2)+'\n')
    print(json.dumps(audits,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
