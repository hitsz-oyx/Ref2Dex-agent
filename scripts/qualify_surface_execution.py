"""Corrected-trace causal hand-motion and frozen prior qualification on one GPU."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.surface_execution import FLOW_MODES,episode_split,selected_rows,fit_execution,predict_execution,area_hand_samples,execution_gates
from src.task.CmResidual.surface_motion_prior import SurfaceMotionPrior,encode_raw,metrics
from src.task.CmResidual.v118_planner import TorchInspireKinematics,QUERY_LINKS
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
from scripts.run_contact_response_probe import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-source',type=Path,required=True);p.add_argument('--prior-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists() and torch.cuda.is_available()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
    d=a.native_source/'s655';initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
    metadata=json.loads((d/'physical_metadata.json').read_text());r=json.loads((d/'results.json').read_text())
    assert sha(d/'trace.pt')==r['trace_sha256'] and sha(d/'initial.pt')==r['initial_sha256'] and sha(d/'physical_metadata.json')==r['physical_metadata_sha256']
    assert metadata['correct_shape_ownership_verified'] and json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED'
    assert torch.equal(trace['hand_root'],trace['hand_root'][0:1].expand_as(trace['hand_root'])), 'fixed actor base required'
    assert torch.equal(trace['progress'],torch.arange(1,203)[:,None].expand(202,768)), 'no reset or missing tick'
    train_ids,held_ids=episode_split(initial['motion'].numpy(),initial['arm_assignment'].numpy())
    train=selected_rows(initial,trace,train_ids);held=selected_rows(initial,trace,held_ids)
    assert not set(train_ids)&set(held_ids)
    # Input context is already independently audited; revalidate causal q/dq here.
    for rows in (train,held):
        actual=trace['context'][rows['tick'],rows['env']].numpy()
        assert np.array_equal(actual[:,:18],rows['q']) and np.array_equal(actual[:,18:36],rows['dq'])
    fitted=fit_execution(train,device);qhat=predict_execution(held,fitted,initial['native_lower'].numpy(),initial['native_upper'].numpy())
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    surface=area_hand_samples(urdf);fk=TorchInspireKinematics(urdf,device)
    local=torch.from_numpy(surface['points']).to(device);normal=torch.from_numpy(surface['normals']).to(device);links=torch.from_numpy(surface['links']).to(device)
    source_prov=json.loads((a.prior_source/'data/provenance.json').read_text());records=source_prov['selection']['inspire_train']
    object_row=next(i for i,r in enumerate(records) if r['object']=='airplane')
    with np.load(a.prior_source/'data/inspire_train.npz') as old:
        pose=old['pose'][object_row]; object_local=((old['obj'][object_row]-pose[:3,3])@pose[:3,:3]).astype(np.float32)
        object_normal=(old['normal'][object_row]@pose[:3,:3]).astype(np.float32)
    out.mkdir();np.savez(out/'train_rows.npz',**train);np.savez(out/'held_rows.npz',**held)
    np.savez(out/'geometry.npz',**surface,object_local=object_local,object_normal=object_normal)
    (out/'execution_fit.json').write_text(json.dumps(dict(scale=fitted['scale'].tolist(),coefficients={k:v.tolist() for k,v in fitted['coefficients'].items()},ridge=fitted['ridge']),indent=2)+'\n')
    np.savez(out/'joint_predictions.npz',**qhat)
    sources={name:a.prior_source/'fit'/(name+'_7168.pt') for name in ('mano','inspire')};models={}
    for name,path in sources.items():
        ckpt=torch.load(path,map_location='cpu',weights_only=False);model=SurfaceMotionPrior().to(device).eval();model.load_state_dict(ckpt['state']);models[name]=model
    feature_chunks={mode:[] for mode in FLOW_MODES};target_chunks=[];hand_errors={mode:[] for mode in FLOW_MODES};pred_chunks={name+'__'+mode:[] for name in models for mode in FLOW_MODES}
    knn_chunks=[];sdk_max=0.;sdk_rotation_max=0.;contact_names=[metadata['native_body_names'][i] for i in metadata['contact_body_ids']];contact_ids=[QUERY_LINKS.index(n) for n in contact_names]
    def world_links(q,base):return base[:,None]@fk.forward(q[:,None])[:,0]
    def points(q,base):
        poses=world_links(q,base);rot=poses[:,links,:3,:3];trans=poses[:,links,:3,3]
        return (rot@local[None,:,:,None]).squeeze(-1)+trans, (rot@normal[None,:,:,None]).squeeze(-1),poses
    object_local_tensor=torch.from_numpy(object_local).to(device);object_normal_tensor=torch.from_numpy(object_normal).to(device)
    with torch.no_grad():
        for start in range(0,len(held['env']),16):
            stop=min(start+16,len(held['env']));rows={k:v[start:stop] for k,v in held.items()}
            base=dexplore_root_pose(torch.from_numpy(rows['hand_root']).to(device))
            current_hand,current_normals,_=points(torch.from_numpy(rows['q']).to(device),base)
            actual_hand,_,actual_links=points(torch.from_numpy(rows['next_q']).to(device),base)
            sdk_error=float((actual_links[:,contact_ids,:3,3]-torch.from_numpy(rows['sdk_next_positions']).to(device)).abs().max());sdk_max=max(sdk_max,sdk_error)
            sdk_rotation=dexplore_root_pose(torch.cat((torch.zeros(len(rows['env'])*5,3),torch.from_numpy(rows['sdk_next_quaternions']).reshape(-1,4),torch.zeros(len(rows['env'])*5,6)),-1).to(device))[:,:3,:3].reshape(-1,5,3,3)
            sdk_rotation_max=max(sdk_rotation_max,float((actual_links[:,contact_ids,:3,:3]-sdk_rotation).abs().max()))
            assert sdk_error<2e-4, 'FK/native body origin mismatch'
            poses={k:dexplore_root_pose(torch.from_numpy(rows[k]).to(device)) for k in ('current_obj','previous_obj','next_obj')}
            obj={k:object_local_tensor[None]@v[:,:3,:3].transpose(-1,-2)+v[:,None,:3,3] for k,v in poses.items()}
            normals=object_normal_tensor[None]@poses['current_obj'][:,:3,:3].transpose(-1,-2)
            distances=torch.cdist(obj['current_obj'],current_hand,compute_mode='donot_use_mm_for_euclid_dist');knn=distances.topk(4,largest=False,sorted=True).indices
            knn_chunks.append(knn.cpu().numpy())
            batch=torch.arange(len(rows['env']),device=device)[:,None,None]
            gathered=current_hand[batch,knn];gathered_normals=current_normals[batch,knn]
            for mode in FLOW_MODES:
                next_hand=actual_hand if mode=='oracle' else points(torch.from_numpy(qhat[mode][start:stop]).to(device),base)[0]
                err=(next_hand-actual_hand).norm(dim=-1).mean(-1)*1000;hand_errors[mode].append(err.cpu().numpy())
                raw=dict(pose=poses['current_obj'].cpu().numpy(),obj=obj['current_obj'].cpu().numpy(),normal=normals.cpu().numpy(),
                         previous_obj=obj['previous_obj'].cpu().numpy(),next_obj=obj['next_obj'].cpu().numpy(),hand=gathered.cpu().numpy(),
                         hand_normal=gathered_normals.cpu().numpy(),next_hand=next_hand[batch,knn].cpu().numpy(),
                         global_hand_flow=(next_hand-current_hand).mean(1).cpu().numpy())
                x,y=encode_raw(raw);feature_chunks[mode].append(x)
                if mode=='oracle':target_chunks.append(y)
                features=torch.from_numpy(x).to(device)
                for name,model in models.items():pred_chunks[name+'__'+mode].append(model(features).cpu().numpy())
            if (start//16+1)%64==0:print(json.dumps(dict(held_windows=stop,total=len(held['env']))),flush=True)
    features={mode:np.concatenate(parts) for mode,parts in feature_chunks.items()};target=np.concatenate(target_chunks)
    errors={mode:np.concatenate(parts) for mode,parts in hand_errors.items()};predictions={name:np.concatenate(parts) for name,parts in pred_chunks.items()}
    np.savez(out/'features.npz',**features,target=target,knn_indices=np.concatenate(knn_chunks));np.savez(out/'hand_errors.npz',**errors);np.savez(out/'predictions.npz',**predictions)
    parents=[str(env) for env in held['env']];reports={name:{mode:metrics(predictions[name+'__'+mode],target,parents) for mode in FLOW_MODES} for name in models}
    hand={mode:dict(episode_epe_mm=float(value.reshape(384,16).mean(1).mean()),per_episode_epe_mm=value.reshape(384,16).mean(1).tolist()) for mode,value in errors.items()}
    baselines=dict(zero=metrics(np.zeros_like(target),target,parents),persistence=metrics(features['stationary'][...,19:22],target,parents))
    diagnostics={}
    for field,count in (('motion',3),('arm',4)):
        diagnostics[field]={}
        for group in range(count):
            mask=held[field]==group;subset=[parents[i] for i in np.flatnonzero(mask)]
            diagnostics[field][str(group)]=dict(episodes=len(set(subset)),windows=int(mask.sum()),
                hand_mm={mode:float(value[mask].mean()) for mode,value in errors.items()},
                prior_mm={name:{mode:metrics(predictions[name+'__'+mode][mask],target[mask],subset)['parent_epe_mm'] for mode in FLOW_MODES} for name in models},
                persistence_mm=metrics(features['stationary'][mask,:,19:22],target[mask],subset)['parent_epe_mm'])
    # Prospective support diagnostic only; the all-window primary gates stand.
    near=(features['stationary'][...,15].min(-1)*.05)<.02
    diagnostics['proximity']={}
    for name,mask in (('near',near),('far',~near)):
        if not mask.any():diagnostics['proximity'][name]=dict(windows=0,episodes=0);continue
        subset=[parents[i] for i in np.flatnonzero(mask)]
        diagnostics['proximity'][name]=dict(windows=int(mask.sum()),episodes=len(set(subset)),
            prior_mm={model_name:{mode:metrics(predictions[model_name+'__'+mode][mask],target[mask],subset)['parent_epe_mm'] for mode in FLOW_MODES} for model_name in models},
            persistence_mm=metrics(features['stationary'][mask,:,19:22],target[mask],subset)['parent_epe_mm'])
    gates,label=execution_gates(hand,reports,baselines['persistence']['parent_epe_mm'])
    result=dict(run_status='COMPLETED',label=label,gates=gates,hand=hand,prior=reports,baselines=baselines,diagnostics=diagnostics,
                train_episodes=384,held_episodes=384,train_windows=len(train['env']),held_windows=len(held['env']),
                sdk_fk_max_m=sdk_max,sdk_rotation_max=sdk_rotation_max,frozen_prior_sha256={name:sha(path) for name,path in sources.items()},
                object_geometry_record=records[object_row],new_native_ticks=0,actual_optimizer_updates=0,execution_coefficients_each=54,
                observed_same_seed_episode_holdout=True,kinematic_hand_labels_from_measured_q=True,policy_utility_unmeasured=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=label,gates=gates,hand_mm={k:v['episode_epe_mm'] for k,v in hand.items()},prior_mm={n:{k:v['parent_epe_mm'] for k,v in r.items()} for n,r in reports.items()})),flush=True)


if __name__=='__main__':main()
