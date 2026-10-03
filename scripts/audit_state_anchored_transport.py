"""Independent frozen-base/mixture neural, current proximity, metric and AdamW audit."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_rigid_coupling import numpy_forward as state_forward,metric
from scripts.audit_surface_execution import independent_links,pose
from src.task.CmResidual.v118_planner import QUERY_LINKS

STATE=np.asarray([0,1]*6+[0],np.int64)
OLD_STATE=np.asarray([0,1]*7+[0],np.int64)


def independent_tokens(raw,arm):
    indices=STATE if arm=='state_only' else np.arange(2,15)
    value=np.concatenate((raw[:,0:1],raw[:,indices]),1).copy()
    value[:,0,-3:]=0
    return value


def prior(arm):
    counts=np.bincount(STATE,minlength=2)
    return np.r_[np.log(99),-np.log(2*counts[STATE]) if arm=='state_only' else np.full(13,-np.log(13))].astype(np.float32)


def forward(state,x,bias):
    values=[x]
    for i in (0,2):values.append(np.maximum(values[-1]@state['layers.'+str(i)+'.weight'].T+state['layers.'+str(i)+'.bias'],0))
    logits=(values[-1]@state['layers.4.weight'].T)[...,0]
    z=logits+bias;z=z-z.max(1,keepdims=True);probability=np.exp(z);probability/=probability.sum(1,keepdims=True)
    return logits,probability,values


def replay(ck,raw,base,endpoints,target,gate):
    state={k:v.numpy().astype(np.float64) for k,v in ck['common_initial'].items()}
    first={k:np.zeros_like(v) for k,v in state.items()};second={k:np.zeros_like(v) for k,v in state.items()}
    arm=ck['arm'];bias=prior(arm).astype(np.float64);maximum=loss_max=0.
    for step in range(3):
        ids=ck['batch_schedule'][step].numpy()
        x=((independent_tokens(raw[ids],arm)-ck['input_mean'].numpy())/ck['input_std'].numpy()).astype(np.float64)
        b=base[ids].astype(np.float32).astype(np.float64)
        e=endpoints[ids][:,STATE if arm=='state_only' else np.arange(2,15)].astype(np.float32).astype(np.float64)
        y=target[ck['label_permutation'].numpy()[ids] if arm=='shuffled' else ids].astype(np.float32).astype(np.float64)
        logits,probability,activations=forward(state,x,bias);direction=e-b[:,None]
        prediction=b+(probability[:,1:,None,None]*gate[ids,None,None,None]*direction).sum(1)
        error=(prediction-y)*1000;norm=np.sqrt((error**2).sum(-1)+1e-12)
        loss=float(norm.mean());loss_max=max(loss_max,abs(loss-ck['losses'][step]))
        g=error/norm[...,None]/norm.size
        gradient_p=np.zeros_like(probability)
        gradient_p[:,1:]=np.einsum('bpx,bjpx->bj',g,direction)*1000*gate[ids,None]
        delta=(probability*(gradient_p-(gradient_p*probability).sum(1,keepdims=True)))[...,None]
        gradients={}
        for i,activation in ((4,activations[2]),(2,activations[1]),(0,activations[0])):
            key='layers.'+str(i);flat=delta.reshape(-1,delta.shape[-1]);act=activation.reshape(-1,activation.shape[-1])
            gradients[key+'.weight']=flat.T@act
            if key+'.bias' in state:gradients[key+'.bias']=flat.sum(0)
            if i>0:delta=(delta@state[key+'.weight'])*(activation>0)
        norm_g=np.sqrt(sum((v**2).sum() for v in gradients.values()));clip=min(1.,10/(norm_g+1e-6))
        for key,value in state.items():
            g=gradients[key]*clip;first[key]=.9*first[key]+.1*g;second[key]=.999*second[key]+.001*g*g
            state[key]=value*(1-3e-4*1e-4)-3e-4*(first[key]/(1-.9**(step+1)))/(np.sqrt(second[key]/(1-.999**(step+1)))+1e-8)
            error_max=float(np.abs(state[key]-ck['early_states'][step][key].numpy()).max());maximum=max(maximum,error_max)
            assert error_max<5e-5,(step,key,error_max)
    assert loss_max<2e-5,loss_max
    return maximum,loss_max


def main():
    p=argparse.ArgumentParser()
    for n in ('root','coupling-source','execution-source','capacity-source','calibration-source'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);fit=a.root/'fit';r=json.loads((fit/'results.json').read_text())
    banks=np.load(a.coupling_source/'data/inputs.npz');tf=np.load(a.coupling_source/'data/train_fields.npz');hf=np.load(a.capacity_source/'capacity/fields.npz')
    original=torch.load(a.coupling_source/'fit/state_only.pt',map_location='cpu',weights_only=False)
    base_state={k:v.numpy() for k,v in original['state'].items()}
    mean=banks['train'].astype(np.float64).mean((0,1)).astype(np.float32);std=np.maximum(banks['train'].astype(np.float64).std((0,1)),.01).astype(np.float32)
    assert np.array_equal(mean,banks['mean'].astype(np.float32)) and np.array_equal(std,banks['std'].astype(np.float32))
    calibration=json.loads((a.calibration_source/'run_manifest.json').read_text());oldtrain=Path(calibration['source_run'])/'data/train_features.npz'
    rawgates={}
    with np.load(oldtrain) as f:rawgates['train']=f['features'][...,15].min(-1)*.05
    with np.load(a.execution_source/'qualified/features.npz') as f:rawgates['held']=f['stationary'][...,15].min(-1)*.05
    base={};rows={};gate={};base_max=distance_max=0.
    with np.load(a.execution_source/'qualified/geometry.npz') as f:geometry={k:f[k] for k in f.files}
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    for group,fields in [('train',tf),('held',hf)]:
        saved=np.load(fit/(group+'_base.npz'));base[group]=saved['prediction'];gate[group]=saved['gate']
        assert np.array_equal(gate[group],rawgates[group]<.02)
        with np.load(a.execution_source/'qualified'/(group+'_rows.npz')) as f:rows[group]={k:f[k] for k in f.files}
        assert np.array_equal(rows[group]['tick'],np.tile(np.linspace(1,200,16,dtype=int),384))
        assert len(np.unique(rows[group]['env']))==384
        ep=fields['endpoints'] if group=='train' else fields['causal'];x=((banks[group]-mean)/std)[:,OLD_STATE]
        for start in range(0,6144,64):
            c,s,_,_=state_forward(base_state,x[start:start+64]);stored_c=saved['coefficients'][start:start+64];stored_s=saved['scores'][start:start+64]
            error=max(float(np.abs(c-stored_c).max()),float(np.abs(s-stored_s).max()));base_max=max(base_max,error);assert error<2e-4
            # Replica ties need not name the same copy, but must name the same state endpoint.
            assert np.array_equal(OLD_STATE[s.argmin(1)],OLD_STATE[saved['winner'][start:start+64]])
        assert np.array_equal(saved['winner'],saved['scores'].argmin(1))
        winner=saved['winner'];index=np.arange(6144)
        rebuilt=fields['anchor']+saved['coefficients'][index,winner].astype(np.float64)[:,None,None]*(ep[:,OLD_STATE][index,winner]-fields['anchor'])
        assert np.max(np.abs(rebuilt-base[group]))<1e-12
        ids=np.arange(0,6144,16);subset={k:v[ids] for k,v in rows[group].items()}
        links=independent_links(subset['q'],urdf);handroot=pose(subset['hand_root']);obj=pose(subset['current_obj'])
        matrices=np.stack([handroot@links[name] for name in QUERY_LINKS],1)
        selected=matrices[:,geometry['links']]
        assert selected.shape==(384,10135,4,4)
        hand=np.einsum('bpij,pj->bpi',selected[:,:,:3,:3],geometry['points'])+selected[:,:,:3,3]
        query=geometry['object_local'][None]@obj[:,:3,:3].transpose(0,2,1)+obj[:,None,:3,3]
        distance=np.asarray([cKDTree(h).query(o,k=1)[0].min() for h,o in zip(hand,query)])
        error=float(np.abs(distance-rawgates[group][ids]).max());distance_max=max(distance_max,error);assert error<2e-6
        assert np.array_equal(distance<.02,gate[group][ids])
    assert not set(rows['train']['env'])&set(rows['held']['env'])
    old=np.load(a.coupling_source/'fit/state_only_outputs.npz');assert np.array_equal(base['held'],old['prediction'])
    env=rows['held']['env'];y=hf['target'];masks={**{'motion_'+str(i):rows['held']['motion']==i for i in range(3)},**{'arm_'+str(i):rows['held']['arm']==i for i in range(4)},'near':gate['held'],'far':~gate['held']}
    schedule=torch.randint(6144,(1500,32),generator=torch.Generator().manual_seed(4302));permutation=np.random.default_rng(4303).permutation(6144)
    torch.manual_seed(4301);l0=torch.nn.Linear(120,128);l2=torch.nn.Linear(128,64);l4=torch.nn.Linear(64,1,bias=False)
    initial={'layers.0.weight':l0.weight.detach(),'layers.0.bias':l0.bias.detach(),'layers.2.weight':l2.weight.detach(),'layers.2.bias':l2.bias.detach(),
             'layers.4.weight':torch.zeros_like(l4.weight)}
    metric_max=forward_max=early_max=loss_max=0.;reports={}
    for arm in ('full','state_only','shuffled'):
        ck=torch.load(fit/(arm+'.pt'),map_location='cpu',weights_only=False);out=np.load(fit/(arm+'_outputs.npz'))
        assert ck['steps']==1500 and len(ck['losses'])==1500 and np.isfinite(ck['losses']).all() and torch.equal(ck['batch_schedule'],schedule)
        assert all(int(v['step'])==1500 for v in ck['optimizer']['state'].values())
        assert all(torch.equal(v,initial[k]) for k,v in ck['common_initial'].items())
        assert all(torch.equal(v,original['state'][k]) for k,v in ck['frozen_state'].items())
        assert len(ck['optimizer']['state'])==5 and sum(v.numel() for v in ck['state'].values())==23808
        assert max(float((v-initial[k]).abs().max()) for k,v in ck['state'].items())>0
        assert np.array_equal(ck['input_mean'].numpy(),mean) and np.array_equal(ck['input_std'].numpy(),std) and np.array_equal(ck['log_prior'].numpy(),prior(arm))
        if arm=='shuffled':assert np.array_equal(ck['label_permutation'].numpy(),permutation)
        if arm=='state_only':assert np.array_equal(ck['state_replica_ids'].numpy(),STATE)
        raw=independent_tokens(banks['held'],arm);assert np.array_equal(raw[:,:,:99],np.broadcast_to(banks['held'][:,0:1,:99],(6144,14,99)))
        x=(raw-mean)/std;state={k:v.numpy() for k,v in ck['state'].items()}
        for start in range(0,6144,64):
            logits,prob,_=forward(state,x[start:start+64],prior(arm));other=prob[:,1:]*gate['held'][start:start+64,None]
            weights=np.concatenate((1-other.sum(1,keepdims=True),other),1)
            error=max(float(np.abs(logits-out['logits'][start:start+64]).max()),float(np.abs(weights-out['weights'][start:start+64]).max()));forward_max=max(forward_max,error);assert error<2e-4
        ep=hf['causal'][:,STATE if arm=='state_only' else np.arange(2,15)]
        weights=out['weights'];assert np.isfinite(weights).all() and weights.min()>=-1e-7 and np.allclose(weights.sum(1),1,atol=2e-6,rtol=0)
        prediction=base['held']+(weights[:,1:].astype(np.float64)[...,None,None]*(ep-base['held'][:,None])).sum(1)
        assert np.max(np.abs(prediction-out['prediction']))<1e-12 and np.array_equal(prediction[~gate['held']],base['held'][~gate['held']])
        assert np.count_nonzero(weights[~gate['held'],1:])==0 and np.all(weights[~gate['held'],0]==1)
        value,per=metric(prediction,y,env);reports[arm]=value;metric_max=max(metric_max,abs(value-r['reports'][arm]['episode_epe_mm']))
        assert all(abs(v-r['reports'][arm]['per_episode_epe_mm'][k])<1e-9 for k,v in per.items())
        for name,mask in masks.items():
            value,_=metric(prediction[mask],y[mask],env[mask]);assert abs(value-r['diagnostics'][name][arm]['episode_epe_mm'])<1e-9
        early,loss=replay(ck,banks['train'],base['train'],tf['endpoints'],tf['target'],gate['train']);early_max=max(early_max,early);loss_max=max(loss_max,loss)
    for name,value in [('frozen_state',base['held']),('persistence',hf['anchor']),('zero',np.zeros_like(y))]:
        error=abs(metric(value,y,env)[0]-r['baselines'][name]['episode_epe_mm']);metric_max=max(metric_max,error)
        if name!='zero':
            for m,mask in masks.items():assert abs(metric(value[mask],y[mask],env[mask])[0]-r['diagnostics'][m][name]['episode_epe_mm'])<1e-9
    assert metric_max<1e-9
    full=reports['full'];gates=dict(frozen_state=full<=.9*r['baselines']['frozen_state']['episode_epe_mm'],state_only=full<=.9*reports['state_only'],shuffled=full<=.95*reports['shuffled'],persistence=full<=.9*r['baselines']['persistence']['episode_epe_mm'],near_state_only=r['diagnostics']['near']['full']['episode_epe_mm']<=.95*r['diagnostics']['near']['state_only']['episode_epe_mm'])
    useful=gates['frozen_state'] and gates['state_only'];label='PROMISING' if all(gates.values()) else ('UNCLEAR' if useful else 'UNPROMISING')
    assert gates==r['gates'] and label==r['label'] and r['actual_optimizer_updates']==4500
    result=dict(run_status='COMPLETED',label=label,base_numpy_rows=12288,frozen_base_forward_max=base_max,
                independently_rebuilt_current_proximity_rows=768,distance_max_m=distance_max,
                inherited_raw_and_transport_audits_by_sha=True,new_full_transport_geometry_not_rebuilt=True,
                all_held_mixture_forward_max=forward_max,metric_max_mm=metric_max,
                independent_numpy_adamw_steps=9,early_parameter_max=early_max,early_loss_max=loss_max,
                remaining_4491_steps_not_replayed=True,frozen_state_preserved=True,far_predictions_exact=True,
                held_outcomes_not_deployment_inputs=True)
    (a.root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
