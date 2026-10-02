"""Short prefix replay and unchanged matching gates; no task outcomes."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import rotation
from scripts.audit_truth_successor_native_panel import couple,COUPLING


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);a=parser.parse_args();root=a.directory.resolve();d=root/'s607'
    m=json.loads((root/'run_manifest.json').read_text());r=json.loads((d/'results.json').read_text());assert r['run_status']=='COMPLETED' and r['engineering_only'] and not r['options_executed'] and r['physics_steps_each']==56
    assert sha(d/'initial.pt')==r['initial_sha256'] and sha(d/'trace.pt')==r['trace_sha256'] and sha(d/'physical_metadata.json')==r['physical_metadata_sha256']
    i=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    meta=json.loads((d/'physical_metadata.json').read_text());assert len(meta['environment_origins'])==768 and not np.any(meta['environment_origins'])
    n=lambda k:i[k].numpy();v=lambda k:t[k].numpy();q=np.concatenate((n('base_q')[None],v('native_q')[:-1]),0);dq=np.concatenate((n('initial_dof_vel')[None],v('native_dq')[:-1]),0);obj=np.concatenate((n('object_root')[None],v('object_root')[:-1]),0);contact=np.concatenate((n('initial_contact')[None],v('contact')[:-1]),0)
    assert np.array_equal(v('progress'),np.arange(1,57)[:,None]+np.zeros((56,768),dtype=np.int64))
    planned=np.minimum(v('progress'),n('phase_stop')[n('motion')][None]);ref=n('native_reference_q')[n('motion')[None],planned]
    context=np.concatenate((q,dq,obj,contact,ref,(planned/n('phase_stop')[n('motion')][None])[...,None]),-1).astype(np.float32)
    assert np.max(np.abs(context-v('context')))<=1e-6
    z=np.clip((context-base['observation_mean'].numpy())/base['observation_std'].numpy(),-10,10);assert np.max(np.abs(z-v('normalized_context')))<=2e-6
    state=base['model'];layers=sorted(int(k.split('.')[1]) for k in state if k.endswith('.weight'));x=v('normalized_context').reshape(-1,70).astype(np.float64)
    for j,layer in enumerate(layers):
        x=x@state[f'network.{layer}.weight'].numpy().astype(np.float64).T+state[f'network.{layer}.bias'].numpy();x=np.tanh(x) if j==len(layers)-1 else np.maximum(x,0)
    x=x.reshape(56,768,18);x[...,[7,9,11,13,16,17]]=0;p0_error=float(np.abs(x-v('model_residual')).max());assert p0_error<=2e-5
    goal=ref+v('model_residual')*np.array([.02]*3+[.10]*3+[.40]*12,dtype=np.float32);goal[...,14]=np.clip(goal[...,14],n('native_lower')[14],n('native_upper')[14]);goal=couple(goal,n('native_lower'),n('native_upper'))
    assert np.max(np.abs(goal-v('target')))<=1e-5 and not v('request').any()
    act=v('action').copy();assert np.max(np.abs(act))<=1+1e-6;act[...,6:]=(1+act[...,6:])/2;pd=n('pd_offset')+n('pd_scale')*act;pd[...,:6]+=q[...,:6]
    for parent,children in COUPLING.items():
        for child,ratio in children:pd[...,child]=pd[...,parent]*ratio
    pd_error=float(np.abs(pd-v('target')).max());assert pd_error<=1e-5
    lookup=np.full((192,4),-1,dtype=np.int64);lookup[n('option_cluster'),n('policy_group')]=np.arange(768);assert (lookup>=0).all()
    maximum=dict(initial=0.,xyz=0.,joint_angles=0.,linear_velocity=0.,angular_velocity=0.,rotation=0.)
    for ids in lookup:
        end=int(n('decision_steps')[ids[0]])+1
        assert end<=56
        for values in (n('base_q'),n('initial_dof_vel'),n('object_root')):maximum['initial']=max(maximum['initial'],float(np.abs(values[ids]-values[ids[0]]).max()))
        for key,values in [('xyz',q[:end,ids,:3]),('xyz',obj[:end,ids,:3]),('joint_angles',q[:end,ids,3:]),('linear_velocity',dq[:end,ids,:3]),('linear_velocity',obj[:end,ids,7:10]),('angular_velocity',dq[:end,ids,3:]),('angular_velocity',obj[:end,ids,10:13]),('rotation',rotation(obj[:end,ids,3:7]))]:
            maximum[key]=max(maximum[key],float(np.abs(values-values[:,:1]).max()))
    gates=dict(initial=maximum['initial']<=1e-7,positions=maximum['xyz']<=.0005,joint_angles=maximum['joint_angles']<=.005,translational_velocities=maximum['linear_velocity']<=.01,angular_joint_velocities=maximum['angular_velocity']<=.1,object_rotation=maximum['rotation']<=.005)
    result=dict(run_status='COMPLETED',engineering_only=True,prefix_matching_passes=all(gates.values()),matching_gates=gates,matching_maximum=maximum,p0_numpy_forward_error=p0_error,pd_error=pd_error,all_origins_zero=True,options_executed=False,no_task_outcomes=True)
    assert not (root/'results.json').exists();(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
