"""Independent causal features, coefficient certificates, NumPy model/AdamW."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_surface_execution import independent_links,pose
NAMES=('hand_base_link','thumb_proximal_base','thumb_proximal','thumb_intermediate','thumb_distal','thumb_tip',
       'index_proximal','index_intermediate','index_tip','middle_proximal','middle_intermediate','middle_tip',
       'ring_proximal','ring_intermediate','ring_tip','pinky_proximal','pinky_intermediate','pinky_tip')
LINKS=(0,1,2,3,4,6,7,9,10,12,13,15,16)
STATE=np.asarray([0,1]*7+[0],np.int64)


def independent_inputs(rows,anchor,endpoints,local,scale,urdf):
    current=pose(rows['current_obj']);previous=pose(rows['previous_obj']);rotation=current[:,:3,:3];transpose=rotation.transpose(0,2,1)
    lever=local.astype(np.float64)-local.astype(np.float64).mean(0)
    gram=(lever**2).sum()*np.eye(3)-lever.T@lever;inverse=np.linalg.inv(gram+1e-12*np.eye(3))
    def twist(flow):
        value=flow@rotation[:,None];linear=value.mean(-2)
        angular=np.cross(lever[None,None],value-linear[:,:,None]).sum(-2)@inverse.T
        return np.concatenate((linear/.01,angular/.05),-1)
    end=twist(endpoints);past_twist=twist(anchor[:,None])[:,0]
    base=pose(rows['hand_root']);links=independent_links(rows['q'],urdf);hand=np.stack([base@links[NAMES[i]] for i in LINKS],1)
    relative_rotation=transpose[:,None]@hand[:,:,:3,:3]
    relative_position=np.einsum('bij,bnj->bni',transpose,hand[:,:,:3,3]-current[:,None,:3,3])/.05
    relative=np.concatenate((relative_position,relative_rotation.reshape(-1,13,9)),-1)
    joint=np.concatenate((rows['q'].astype(np.float64)/scale,rows['dq'].astype(np.float64)/30/scale,
                          (rows['target'].astype(np.float64)-rows['q'].astype(np.float64))/scale),-1)
    obj=np.concatenate((current[:,:3,3]/.1,rotation[:,:,:2].reshape(-1,6)),-1)
    past=np.concatenate(((current[:,:3,3]-previous[:,:3,3])/.01,
                         ((rotation@previous[:,:3,:3].transpose(0,2,1)-np.eye(3))/.05).reshape(-1,9)),-1)
    common=np.concatenate((joint,obj,past,past_twist,relative.mean(1),end[:,2:].mean(1)),-1)
    per_relative=np.concatenate((np.zeros((len(common),2,12)),relative),1)
    kind=np.zeros((len(common),15,3));kind[:,0,0]=1;kind[:,1,1]=1;kind[:,2:,2]=1
    return np.concatenate((np.broadcast_to(common[:,None],(len(common),15,99)),per_relative,end,kind),-1)


def numpy_forward(state,x):
    values=[x]
    for index in (0,2):values.append(np.maximum(values[-1]@state['layers.'+str(index)+'.weight'].T+state['layers.'+str(index)+'.bias'],0))
    raw=values[-1]@state['layers.4.weight'].T+state['layers.4.bias']
    coefficient=1/(1+np.exp(-raw[...,0]));score=np.logaddexp(0,raw[...,1])
    return coefficient,score,values,raw


def replay(checkpoint,inputs,fields):
    state={k:v.numpy().astype(np.float64) for k,v in checkpoint['common_initial'].items()}
    first={k:np.zeros_like(v) for k,v in state.items()};second={k:np.zeros_like(v) for k,v in state.items()}
    arm=checkpoint['arm'];weights=checkpoint['component_weights'].numpy();maximum=loss_max=0.
    for step in range(3):
        ids=checkpoint['batch_schedule'][step].numpy();x=((inputs[ids]-checkpoint['input_mean'].numpy())/checkpoint['input_std'].numpy())
        endpoints=fields['endpoints'][ids].astype(np.float32);anchor=fields['anchor'][ids].astype(np.float32)
        labels=fields['shuffled_coefficients'][ids].astype(np.float32) if arm=='shuffled' else fields['coefficients'][ids].astype(np.float32)
        target=fields['target'][fields['permutation'][ids]].astype(np.float32) if arm=='shuffled' else fields['target'][ids].astype(np.float32)
        if arm=='state_only':x=x[:,STATE];endpoints=endpoints[:,STATE];labels=labels[:,STATE]
        c,s,activations,raw=numpy_forward(state,x.astype(np.float64));direction=endpoints.astype(np.float64)-anchor[:,None].astype(np.float64)
        prediction=anchor[:,None]+c[...,None,None]*direction
        confidence=np.log1p(np.linalg.norm(prediction-target[:,None],axis=-1).mean(-1)*1000)
        coefficient_weight=np.minimum(np.linalg.norm(direction,axis=-1).mean(-1)/.01,1)
        loss=float((((c-labels)**2*coefficient_weight+(s-confidence)**2)*weights[None]).mean());loss_max=max(loss_max,abs(loss-checkpoint['losses'][step]))
        # The confidence target is detached; no gradient through its coefficient.
        delta=np.stack((2*(c-labels)*coefficient_weight*c*(1-c),2*(s-confidence)/(1+np.exp(-raw[...,1]))),-1)*weights[None,:,None]/c.size
        gradients={}
        for index,activation in ((4,activations[2]),(2,activations[1]),(0,activations[0])):
            key='layers.'+str(index);flat=delta.reshape(-1,delta.shape[-1]);act=activation.reshape(-1,activation.shape[-1])
            gradients[key+'.weight']=flat.T@act;gradients[key+'.bias']=flat.sum(0)
            if index>0:delta=(delta@state[key+'.weight'])*(activation>0)
        norm=np.sqrt(sum((v**2).sum() for v in gradients.values()));clip=min(1.,10/(norm+1e-6))
        for key,v in state.items():
            g=gradients[key]*clip;first[key]=.9*first[key]+.1*g;second[key]=.999*second[key]+.001*g*g
            state[key]=v*(1-3e-4*1e-4)-3e-4*(first[key]/(1-.9**(step+1)))/(np.sqrt(second[key]/(1-.999**(step+1)))+1e-8)
            error=float(np.abs(state[key]-checkpoint['early_states'][step][key].numpy()).max());maximum=max(maximum,error);assert error<5e-5,(step,key,error)
    assert loss_max<2e-5,loss_max
    return maximum,loss_max


def metric(prediction,target,env):
    error=np.linalg.norm(prediction-target,axis=-1).mean(-1)*1000
    groups={str(i):float(error[env==i].mean()) for i in np.sort(np.unique(env))}
    return float(np.mean(list(groups.values()))),groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--capacity-source',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    data=a.root/'data';fit=a.root/'fit';old=a.execution_source/'qualified';r=json.loads((fit/'results.json').read_text())
    ancestor=json.loads((a.execution_source/'run_manifest.json').read_text());source=Path(ancestor['source_native'])/'s655'
    initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(source/'trace.pt',map_location='cpu',weights_only=False)
    banks=np.load(data/'inputs.npz');fields=np.load(data/'train_fields.npz');held=np.load(a.capacity_source/'capacity/fields.npz')
    permutation=np.random.default_rng(4203).permutation(6144);assert np.array_equal(permutation,fields['permutation'])
    scale=(initial['native_upper']-initial['native_lower']).numpy().astype(np.float64);scale[:3]=.1;scale[3:6]=np.pi;assert np.array_equal(scale,banks['joint_scale'])
    mean=banks['train'].astype(np.float64).mean((0,1));std=np.maximum(banks['train'].astype(np.float64).std((0,1)),.01)
    assert np.array_equal(mean,banks['mean']) and np.array_equal(std,banks['std'])
    rng=np.random.default_rng(4001);selected={'train':[],'held':[]};rows={}
    for motion in range(3):
        for arm in range(4):
            ids=np.flatnonzero((initial['motion'].numpy()==motion)&(initial['arm_assignment'].numpy()==arm));assert len(ids)==64;ids=rng.permutation(ids)
            selected['train'].extend(ids[:32]);selected['held'].extend(ids[32:])
    assert not set(selected['train'])&set(selected['held'])
    history=np.concatenate((initial['object_root'][None].numpy(),trace['object_root'].numpy()),0);ticks=np.linspace(1,200,16,dtype=int)
    with np.load(old/'geometry.npz') as f:local=f['object_local']
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf';input_max=field_max=sdk_max=0.
    actuator=json.loads((old/'execution_fit.json').read_text())
    metadata=json.loads((source/'physical_metadata.json').read_text());contact=[metadata['native_body_names'][i] for i in metadata['contact_body_ids']]
    for group in ('train','held'):
        with np.load(old/(group+'_rows.npz')) as f:rows[group]={k:f[k] for k in f.files}
        env=np.repeat(np.sort(selected[group]),16);tick=np.tile(ticks,384)
        expected=dict(env=env,tick=tick,q=trace['native_q'][tick-1,env].numpy(),dq=trace['native_dq'][tick-1,env].numpy(),target=trace['target'][tick,env].numpy(),next_q=trace['native_q'][tick,env].numpy(),
                      current_obj=trace['object_root'][tick-1,env].numpy(),previous_obj=history[tick-1,env],next_obj=trace['object_root'][tick,env].numpy(),hand_root=trace['hand_root'][tick-1,env].numpy(),
                      sdk_next_positions=trace['hand_body_position'][tick,env].numpy(),sdk_next_quaternions=trace['hand_body_quaternion'][tick,env].numpy(),motion=initial['motion'][env].numpy(),arm=initial['arm_assignment'][env].numpy())
        for key,value in expected.items():assert np.array_equal(value,rows[group][key]),(group,key)
        anchor=fields['anchor'] if group=='train' else held['anchor'];endpoints=fields['endpoints'] if group=='train' else held['causal'];target=fields['target'] if group=='train' else held['target']
        current=pose(rows[group]['current_obj']);previous=pose(rows[group]['previous_obj']);following=pose(rows[group]['next_obj'])
        obj=lambda p:local[None]@p[:,:3,:3].transpose(0,2,1)+p[:,None,:3,3]
        points=obj(current);assert np.allclose(anchor,points-obj(previous),atol=1e-12,rtol=0) and np.allclose(target,obj(following)-points,atol=1e-12,rtol=0)
        inertia=local[None]@(current[:,:3,:3]@previous[:,:3,:3].transpose(0,2,1)@current[:,:3,:3]).transpose(0,2,1)+(2*current[:,:3,3]-previous[:,:3,3])[:,None]-points
        assert np.allclose(inertia,endpoints[:,1],atol=1e-12,rtol=0) and np.count_nonzero(endpoints[:,0])==0
        ids=np.arange(0,6144,16);subset={k:v[ids] for k,v in rows[group].items()}
        rebuilt=independent_inputs(subset,anchor[ids],endpoints[ids],local,scale,urdf)
        error=float(np.abs(rebuilt-banks[group][ids]).max());input_max=max(input_max,error);assert np.allclose(rebuilt,banks[group][ids],atol=2e-5,rtol=2e-6),error
        signals=np.stack((rows[group]['target']-rows[group]['q'],rows[group]['dq']/30),-1).astype(np.float64)/np.asarray(actuator['scale'])
        design=np.concatenate((signals,np.ones_like(signals[...,:1])),-1)
        predicted=rows[group]['q'].astype(np.float64)+(design*np.asarray(actuator['coefficients']['action_velocity'])[None]).sum(-1)
        predicted[:,6:]=np.clip(predicted[:,6:],initial['native_lower'].numpy()[6:],initial['native_upper'].numpy()[6:]);predicted=predicted.astype(np.float32)
        base=pose(subset['hand_root']);cur=independent_links(subset['q'],urdf);nxt=independent_links(predicted[ids],urdf);actual=independent_links(subset['next_q'],urdf)
        for j,i in enumerate(LINKS):
            name=NAMES[i];delta=(base@nxt[name])@np.linalg.inv(base@cur[name]);candidate=points[ids]@delta[:,:3,:3].transpose(0,2,1)+delta[:,None,:3,3]-points[ids]
            error=float(np.abs(candidate-endpoints[ids,j+2]).max());field_max=max(field_max,error);assert error<2e-6
        for j,name in enumerate(contact):
            error=float(np.abs((base@actual[name])[:,:3,3]-subset['sdk_next_positions'][:,j]).max());sdk_max=max(sdk_max,error);assert error<2e-4
    certificate_max=0.
    for label_key,error_key,target in (('coefficients','oracle_errors_m',fields['target']),('shuffled_coefficients','shuffled_errors_m',fields['target'][permutation])):
        c=fields[label_key];assert np.isfinite(c).all() and (c>=0).all() and (c<=1).all()
        for start in range(0,6144,64):
            coefficient=c[start:start+64];direction=fields['endpoints'][start:start+64]-fields['anchor'][start:start+64,None]
            residual=fields['anchor'][start:start+64,None]+coefficient[...,None,None]*direction-target[start:start+64,None]
            errors=np.linalg.norm(residual,axis=-1).mean(-1);assert np.allclose(errors,fields[error_key][start:start+64],atol=1e-12,rtol=0)
            derivative=((residual*direction).sum(-1)/np.sqrt((residual**2).sum(-1)+1e-18)).mean(-1)
            maximum=float((np.maximum(coefficient*derivative,(coefficient-1)*derivative)+1e-9).max());certificate_max=max(certificate_max,maximum);assert maximum<1e-7
    schedule=torch.randint(6144,(1500,32),generator=torch.Generator().manual_seed(4202));reports={};checkpoints={};forward_max=metric_max=early_max=loss_max=0.
    x=((banks['held']-mean.astype(np.float32))/std.astype(np.float32));target=held['target'];env=rows['held']['env']
    with np.load(old/'features.npz') as f:near=f['stationary'][...,15].min(-1)*.05<.02
    masks={**{'motion_'+str(i):rows['held']['motion']==i for i in range(3)},**{'arm_'+str(i):rows['held']['arm']==i for i in range(4)},'near':near,'far':~near}
    for arm in ('full','state_only','shuffled'):
        ck=torch.load(fit/(arm+'.pt'),map_location='cpu',weights_only=False);checkpoints[arm]=ck;output=np.load(fit/(arm+'_outputs.npz'))
        assert ck['steps']==1500 and torch.equal(ck['batch_schedule'],schedule) and len(ck['losses'])==1500 and np.isfinite(ck['losses']).all()
        assert all(int(v['step'])==1500 for v in ck['optimizer']['state'].values())
        assert np.array_equal(ck['input_mean'].numpy(),mean.astype(np.float32)) and np.array_equal(ck['input_std'].numpy(),std.astype(np.float32))
        if arm=='shuffled':assert np.array_equal(ck['label_permutation'].numpy(),permutation)
        if arm=='state_only':assert np.array_equal(ck['state_replica_ids'].numpy(),STATE)
        expected_weights=np.ones(15,dtype=np.float32) if arm!='state_only' else (15/(2*np.bincount(STATE)[STATE])).astype(np.float32)
        assert np.array_equal(expected_weights,ck['component_weights'].numpy())
        state={k:v.numpy() for k,v in ck['state'].items()};features=x[:,STATE] if arm=='state_only' else x;endpoint=held['causal'][:,STATE] if arm=='state_only' else held['causal']
        for start in range(0,6144,64):
            c,s,_,_=numpy_forward(state,features[start:start+64]);error=max(float(np.abs(c-output['coefficients'][start:start+64]).max()),float(np.abs(s-output['scores'][start:start+64]).max()));forward_max=max(forward_max,error);assert error<2e-4
        c=output['coefficients'];s=output['scores'];winner=output['winner'];assert (c>=0).all() and (c<=1).all() and np.array_equal(winner,s.argmin(1))
        prediction=held['anchor']+c[np.arange(6144),winner].astype(np.float64)[:,None,None]*(endpoint[np.arange(6144),winner]-held['anchor'])
        assert np.allclose(prediction,output['prediction'],atol=1e-12,rtol=0)
        oracle=np.load(a.capacity_source/'capacity'/('state_only_oracle.npz' if arm=='state_only' else 'causal_oracle.npz'))
        model_errors=np.linalg.norm(prediction-target,axis=-1).mean(-1);oracle_errors=np.linalg.norm(oracle['prediction']-target,axis=-1).mean(-1)
        assert np.all(model_errors>=oracle_errors-1e-7), 'learned output must remain in audited segment family'
        value,groups=metric(prediction,target,env);reports[arm]=value;error=abs(value-r['reports'][arm]['episode_epe_mm']);metric_max=max(metric_max,error);assert error<1e-9
        for parent,v in groups.items():assert abs(v-r['reports'][arm]['per_episode_epe_mm'][parent])<1e-9
        for name,mask in masks.items():
            value,_=metric(prediction[mask],target[mask],env[mask]);assert abs(value-r['diagnostics'][name][arm]['episode_epe_mm'])<1e-9
        early,loss=replay(ck,banks['train'],fields);early_max=max(early_max,early);loss_max=max(loss_max,loss)
        assert sum(v.numel() for v in ck['state'].values())==23874 and max(float((v-ck['common_initial'][k]).abs().max()) for k,v in ck['state'].items())>0
    torch.manual_seed(4201);first=torch.nn.Linear(120,128);middle=torch.nn.Linear(128,64);last=torch.nn.Linear(64,2)
    expected_initial={'layers.0.weight':first.weight.detach(),'layers.0.bias':first.bias.detach(),'layers.2.weight':middle.weight.detach(),'layers.2.bias':middle.bias.detach(),
                      'layers.4.weight':torch.zeros_like(last.weight),'layers.4.bias':torch.tensor([-2.1972245773362196,.541324854612918])}
    for ck in checkpoints.values():assert all(torch.equal(v,expected_initial[k]) for k,v in ck['common_initial'].items())
    baseline,_=metric(held['anchor'],target,env);zero,_=metric(np.zeros_like(target),target,env)
    assert abs(baseline-r['baselines']['persistence']['episode_epe_mm'])<1e-9 and abs(zero-r['baselines']['zero']['episode_epe_mm'])<1e-9
    for name,mask in masks.items():
        value,_=metric(held['anchor'][mask],target[mask],env[mask]);assert abs(value-r['diagnostics'][name]['persistence']['episode_epe_mm'])<1e-9
    gates=dict(persistence=reports['full']<=.9*baseline,state_only=reports['full']<=.9*reports['state_only'],shuffled=reports['full']<=.95*reports['shuffled'],near_state_only=r['diagnostics']['near']['full']['episode_epe_mm']<=.95*r['diagnostics']['near']['state_only']['episode_epe_mm'])
    useful=gates['persistence'] and gates['state_only'];label='PROMISING' if all(gates.values()) else ('UNCLEAR' if useful else 'UNPROMISING')
    assert gates==r['gates'] and label==r['label'] and r['actual_optimizer_updates']==4500
    audit=dict(run_status='COMPLETED',label=label,raw_rows=12288,independent_feature_geometry_rows=768,full_geometry_not_reconstructed=True,
               input_max=input_max,hand_transport_max_m=field_max,sdk_origin_max_m=sdk_max,train_oracle_certificates=184320,convex_gap_max_m=certificate_max,
               full_held_numpy_output_max=forward_max,metric_max_mm=metric_max,independent_numpy_adamw_steps=9,early_parameter_max=early_max,early_loss_max=loss_max,
               remaining_4491_optimizer_steps_not_replayed=True,no_held_outcome_used_for_deployment=True)
    (a.root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
