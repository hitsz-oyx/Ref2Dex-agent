"""Four-tick causal physical labels for actually held Gaussian commands."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import rotation
from src.task.CmResidual.measured_geometry_barriers import relative_geometry
SCHEMA='coherent-request4-sdk80-state70-command12-barriers2-v1'

def read_panel(path,sources):
    r=json.loads((path/'results.json').read_text());audit=json.loads((path/'panel_audit.json').read_text());assert r['run_status']==audit['run_status']=='COMPLETED' and not r['deterministic'] and r['continuous_updates']==20 and r['coherent_period']==4 and audit['held_request_and_noise_verified']
    for f in ['initial.pt','trace.pt','results.json','panel_audit.json','physical_metadata.json']:sources[str((path/f).resolve())]=sha(path/f)
    assert sha(path/'initial.pt')==r['initial_sha256'] and sha(path/'trace.pt')==r['trace_sha256']
    i=torch.load(path/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(path/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((path/'physical_metadata.json').read_text())
    names=[meta['native_body_names'][k] for k in meta['contact_body_ids']];assert names==['index_intermediate','middle_intermediate','pinky_intermediate','ring_intermediate','thumb_distal']
    group=i['policy_group'].numpy();motion=i['motion'].numpy();stop=i['phase_stop'].numpy()[motion];lift=i['lift_start'].numpy()[motion];progress=t['progress'].numpy();clock=np.arange(202)
    mask=(progress>=stop[None]-74)&(progress+3<=stop[None]+30)&(group[None]>0)&((clock%4==0)&(clock>=2)&(clock+3<202))[:,None];tick,env=np.where(mask)
    assert t['request_is_decision'][tick,env].all() and np.array_equal(t['request_decision_tick'][tick,env].numpy(),tick)
    for step in range(1,4):assert torch.equal(t['request'][tick,env],t['request'][tick+step,env])
    root=t['object_root'].numpy();pre=root[tick-1,env].astype(np.float64);older=root[tick-2,env].astype(np.float64);assert np.max(np.abs(pre-t['context'].numpy()[tick,env,36:49]))<1e-7
    pos,rot=relative_geometry(pre,t['hand_body_position'].numpy()[tick-1,env],t['hand_body_quaternion'].numpy()[tick-1,env]);previous,_=relative_geometry(older,t['hand_body_position'].numpy()[tick-2,env],t['hand_body_quaternion'].numpy()[tick-2,env])
    force=np.concatenate([t['object_force'].numpy()[tick-1,env,None],t['hand_force'].numpy()[tick-1,env]],axis=1).astype(np.float64);mass=np.array([p['mass'] for p in meta['object_body_properties']]);weight=mass*np.linalg.norm(meta['gravity']);assert (weight>0).all()
    force=np.einsum('nij,nbj->nbi',rotation(pre[:,3:7]).swapaxes(-1,-2),force)/weight[env,None,None];clr=t['clearance'].numpy().astype(np.float64)
    prior=4*np.stack([pre[:,2]-older[:,2],clr[tick-1,env]-clr[tick-2,env]],-1)/.005;y=np.stack([root[tick+3,env,2]-pre[:,2],clr[tick+3,env]-clr[tick-1,env]],-1)/.005
    extra=np.concatenate([pos.reshape(-1,15),rot,force.reshape(-1,18),prior,(pos-previous).reshape(-1,15)],-1).astype(np.float32);assert extra.shape[-1]==80
    phase=np.where(progress[tick,env]<lift[env],0,np.where(progress[tick,env]<stop[env]-74,1,np.where(progress[tick,env]<=stop[env],2,3)));cells=motion[env]*4+phase;slots=np.full(768,-1,int)
    for arm in (1,2,3):slots[np.flatnonzero(group==arm)]=np.arange(192)
    full=[t['request_noise'][:,np.flatnonzero(group==arm)] for arm in (1,2,3)];assert torch.equal(full[0],full[1]) and torch.equal(full[0],full[2])
    result=dict(state=t['normalized_context'].numpy()[tick,env],extra=extra,prior=prior.astype(np.float32),action=t['request'][tick,env].tanh().numpy(),y=y.astype(np.float32),cells=cells,cluster=slots[env],tick=tick,environment=env,policy_group=group[env],motion=motion[env],eligible_episodes=len(np.unique(env)))
    assert all(np.isfinite(result[k]).all() for k in ['state','extra','prior','action','y']);return result
