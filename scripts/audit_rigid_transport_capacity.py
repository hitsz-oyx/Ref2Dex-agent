"""Independent SciPy fields and scalar solves; full convex certificates/metrics."""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_surface_execution import independent_links,pose
NAMES=('hand_base_link','thumb_proximal_base','thumb_proximal','thumb_intermediate','thumb_distal','thumb_tip',
       'index_proximal','index_intermediate','index_tip','middle_proximal','middle_intermediate','middle_tip',
       'ring_proximal','ring_intermediate','ring_tip','pinky_proximal','pinky_intermediate','pinky_tip')


def means(prediction,target,env):
    e=np.linalg.norm(prediction-target,axis=-1).mean(-1)*1000
    groups={str(parent):float(e[env==parent].mean()) for parent in np.sort(np.unique(env))}
    return float(np.mean(list(groups.values()))),groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execution-source',type=Path,required=True);a=p.parse_args()
    old=a.execution_source/'qualified';out=a.root/'capacity';r=json.loads((out/'results.json').read_text())
    ancestor=json.loads((a.execution_source/'run_manifest.json').read_text());source=Path(ancestor['source_native'])/'s655'
    assert ancestor['run_status']=='COMPLETED' and json.loads((a.execution_source/'audit.json').read_text())['run_status']=='COMPLETED'
    with np.load(old/'held_rows.npz') as f:rows={k:f[k] for k in f.files}
    with np.load(old/'geometry.npz') as f:local=f['object_local'];sampled_links=np.unique(f['links'])
    initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(source/'trace.pt',map_location='cpu',weights_only=False)
    rng=np.random.default_rng(4001);held=[]
    for motion in range(3):
        for arm in range(4):
            ids=np.flatnonzero((initial['motion'].numpy()==motion)&(initial['arm_assignment'].numpy()==arm));assert len(ids)==64
            held.extend(rng.permutation(ids)[32:])
    env=np.repeat(np.sort(held),16);tick=np.tile(np.linspace(1,200,16,dtype=int),384)
    old_obj=np.concatenate((initial['object_root'][None].numpy(),trace['object_root'].numpy()),0)
    expected=dict(env=env,tick=tick,q=trace['native_q'][tick-1,env].numpy(),dq=trace['native_dq'][tick-1,env].numpy(),
                  target=trace['target'][tick,env].numpy(),next_q=trace['native_q'][tick,env].numpy(),current_obj=trace['object_root'][tick-1,env].numpy(),
                  previous_obj=old_obj[tick-1,env],next_obj=trace['object_root'][tick,env].numpy(),hand_root=trace['hand_root'][tick-1,env].numpy(),
                  sdk_next_positions=trace['hand_body_position'][tick,env].numpy(),sdk_next_quaternions=trace['hand_body_quaternion'][tick,env].numpy(),
                  motion=initial['motion'][env].numpy(),arm=initial['arm_assignment'][env].numpy())
    for key,value in expected.items():assert np.array_equal(value,rows[key]),key
    fit=json.loads((old/'execution_fit.json').read_text())
    signals=np.stack((rows['target']-rows['q'],rows['dq']/30),-1).astype(np.float64)/np.asarray(fit['scale'])
    x=np.concatenate((signals,np.ones_like(signals[...,:1])),-1)
    predicted=rows['q'].astype(np.float64)+(x*np.asarray(fit['coefficients']['action_velocity'])[None]).sum(-1)
    predicted[:,6:]=np.clip(predicted[:,6:],initial['native_lower'].numpy()[6:],initial['native_upper'].numpy()[6:]);predicted=predicted.astype(np.float32)
    with np.load(old/'joint_predictions.npz') as f:assert np.array_equal(predicted,f['action_velocity'])
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    visual_names=[link.get('name') for link in ET.parse(urdf).getroot().findall('link') if link.find('visual/geometry/mesh') is not None]
    visual_ids=sorted(NAMES.index(name) for name in visual_names);assert np.array_equal(visual_ids,sampled_links) and len(visual_ids)==13
    assert r['endpoint_names']==['zero','rigid_inertia']+[NAMES[i] for i in visual_ids]
    fields=np.load(out/'fields.npz');anchor=fields['anchor'];target=fields['target']
    current_pose=pose(rows['current_obj']);previous_pose=pose(rows['previous_obj']);next_pose=pose(rows['next_obj'])
    def points(p):return local[None]@p[:,:3,:3].transpose(0,2,1)+p[:,None,:3,3]
    current=points(current_pose);previous=points(previous_pose);following=points(next_pose)
    state_rotation=current_pose[:,:3,:3]@previous_pose[:,:3,:3].transpose(0,2,1)@current_pose[:,:3,:3]
    state_translation=2*current_pose[:,:3,3]-previous_pose[:,:3,3]
    inertia=local[None]@state_rotation.transpose(0,2,1)+state_translation[:,None]-current
    state=np.stack((np.zeros_like(anchor),inertia),1)
    state_max=max(float(np.abs(anchor-(current-previous)).max()),float(np.abs(target-(following-current)).max()),float(np.abs(state-fields['state_only']).max()))
    assert state_max<1e-12
    assert np.array_equal(fields['causal'][:,:2],fields['state_only']) and np.array_equal(fields['measured_hand'][:,:2],fields['state_only'])
    # Independent link geometry first fixed window of every held episode.
    ids=np.arange(0,6144,16);base=pose(rows['hand_root'][ids]);cur=independent_links(rows['q'][ids],urdf)
    next_links={mode:independent_links(q,urdf) for mode,q in (('causal',predicted[ids]),('measured_hand',rows['next_q'][ids]))}
    metadata=json.loads((source/'physical_metadata.json').read_text());contact=[metadata['native_body_names'][i] for i in metadata['contact_body_ids']]
    sdk_max=0.;transport_max=0.;orthogonality_max=0.
    for name in contact:
        world=base@next_links['measured_hand'][name];j=contact.index(name)
        error=float(np.abs(world[:,:3,3]-rows['sdk_next_positions'][ids,j]).max());sdk_max=max(sdk_max,error);assert error<2e-4
    for mode,poses in next_links.items():
        for j,link in enumerate(visual_ids):
            name=NAMES[link];delta=(base@poses[name])@np.linalg.inv(base@cur[name])
            transported=current[ids]@delta[:,:3,:3].transpose(0,2,1)+delta[:,None,:3,3]-current[ids]
            error=float(np.abs(transported-fields[mode][ids,j+2]).max());transport_max=max(transport_max,error);assert error<2e-6
            rot=delta[:,:3,:3];ortho=float(np.abs(rot@rot.transpose(0,2,1)-np.eye(3)).max());orthogonality_max=max(orthogonality_max,ortho);assert ortho<1e-10
    fits={mode:np.load(out/(mode+'_oracle.npz')) for mode in ('state_only','causal','measured_hand')}
    metric_max=certificate_max=scipy_gap_max=stored_error_max=0.;primary={}
    with np.load(old/'features.npz') as f:near=f['stationary'][...,15].min(-1)*.05<.02
    masks={**{'motion_'+str(i):rows['motion']==i for i in range(3)},**{'arm_'+str(i):rows['arm']==i for i in range(4)},'near':near,'far':~near}
    for mode,record in fits.items():
        endpoint=fields[mode];coefficient=record['coefficients'];assert np.isfinite(coefficient).all() and (coefficient>=0).all() and (coefficient<=1).all()
        for start in range(0,6144,64):
            stop=min(start+64,6144);c=coefficient[start:stop];direction=endpoint[start:stop]-anchor[start:stop,None]
            prediction=anchor[start:stop,None]+c[...,None,None]*direction
            residual=prediction-target[start:stop,None]
            errors=np.linalg.norm(residual,axis=-1).mean(-1)
            discrepancy=float(np.abs(errors-record['segment_errors_m'][start:stop]).max());stored_error_max=max(stored_error_max,discrepancy);assert discrepancy<1e-12
            derivative=((residual*direction).sum(-1)/np.sqrt((residual**2).sum(-1)+1e-18)).mean(-1)
            # Convex smooth-EPE supporting-line gap plus true/smooth bound.
            certificate=np.maximum(c*derivative,(c-1)*derivative)+1e-9
            maximum=float(certificate.max());certificate_max=max(certificate_max,maximum);assert maximum<1e-7,maximum
            winner=errors.argmin(1);stored=record['winner'][start:stop]
            assert np.all(errors[np.arange(len(winner)),stored]<=errors.min(1)+1e-12)
            rebuilt=prediction[np.arange(len(winner)),stored];assert np.allclose(rebuilt,record['prediction'][start:stop],atol=1e-12,rtol=0)
        # Independently re-solve all segments in the384 selected geometry rows.
        for index in ids:
            for j in range(endpoint.shape[1]):
                direction=endpoint[index,j]-anchor[index];offset=anchor[index]-target[index]
                objective=lambda weight:float(np.linalg.norm(offset+weight*direction,axis=-1).mean())
                optimum=minimize_scalar(objective,bounds=(0.,1.),method='bounded',options={'xatol':1e-12,'maxiter':100})
                assert optimum.success
                value=min(objective(0),objective(1),optimum.fun);stored=record['segment_errors_m'][index,j]
                gap=abs(float(value-stored));scipy_gap_max=max(scipy_gap_max,gap);assert gap<1e-7,(mode,index,j,gap)
        value,groups=means(record['prediction'],target,env);primary[mode]=value
        discrepancy=abs(value-r['reports'][mode]['episode_epe_mm']);metric_max=max(metric_max,discrepancy);assert discrepancy<1e-9
        for parent,v in groups.items():assert abs(v-r['reports'][mode]['per_episode_epe_mm'][parent])<1e-9
        for name,mask in masks.items():
            value,_=means(record['prediction'][mask],target[mask],env[mask]);assert abs(value-r['diagnostics'][name][mode]['episode_epe_mm'])<1e-9
            assert r['diagnostics'][name][mode]['windows']==int(mask.sum()) and r['diagnostics'][name][mode]['episodes']==len(np.unique(env[mask]))
        counts={r['endpoint_names'][i]:int((record['winner']==i).sum()) for i in range(endpoint.shape[1])}
        assert counts==r['selection'][mode]['winner_counts']
        quantiles=np.quantile(coefficient[np.arange(6144),record['winner']],[0,.25,.5,.75,1]);assert np.allclose(quantiles,r['selection'][mode]['coefficient_quantiles'],atol=1e-12,rtol=0)
    state_error=np.linalg.norm(fits['state_only']['prediction']-target,axis=-1).mean(-1)
    for mode in ('causal','measured_hand'):
        error=np.linalg.norm(fits[mode]['prediction']-target,axis=-1).mean(-1);assert np.all(error<=state_error+1e-12)
    for name,prediction in (('persistence',anchor),('zero',np.zeros_like(target)),('rigid_inertia',state[:,1])):
        value,_=means(prediction,target,env);assert abs(value-r['baselines'][name]['episode_epe_mm'])<1e-9
    for name,mask in masks.items():
        value,_=means(anchor[mask],target[mask],env[mask]);assert abs(value-r['diagnostics'][name]['persistence']['episode_epe_mm'])<1e-9
    prior=json.loads((old/'results.json').read_text())
    assert abs(r['baselines']['persistence']['episode_epe_mm']-prior['baselines']['persistence']['parent_epe_mm'])<1e-3
    gates=dict(action_capacity=primary['causal']<=.9*primary['state_only'],
               near_action_capacity=r['diagnostics']['near']['causal']['episode_epe_mm']<=.9*r['diagnostics']['near']['state_only']['episode_epe_mm'],
               execution_gap=primary['causal']<=1.1*primary['measured_hand'])
    label='PROMISING' if all(gates.values()) else ('UNCLEAR' if gates['action_capacity'] or gates['near_action_capacity'] else 'UNPROMISING')
    assert gates==r['gates'] and label==r['label'] and r['oracle_segments_solved']==196608
    audit=dict(run_status='COMPLETED',label=label,raw_rows=6144,independent_geometry_rows=384,
               full_geometry_not_reconstructed=True,full_state_field_max_m=state_max,
               hand_transport_max_m=transport_max,sdk_origin_max_m=sdk_max,
               independent_endpoint_rotation_orthogonality_max=orthogonality_max,
               full_segment_certificates=196608,convex_optimality_gap_max_m=certificate_max,
               independent_scipy_segment_solves=384*32,scipy_objective_gap_max_m=scipy_gap_max,
               stored_error_max_m=stored_error_max,metric_max_mm=metric_max,
               no_new_neural_fit_or_physics=True,oracle_coefficient_labels_not_deployable=True)
    (a.root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
