"""Source reconstruction, independent train geometry, NumPy predictions/AdamW."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_surface_execution import independent_links,pose

NAMES=('hand_base_link','thumb_proximal_base','thumb_proximal','thumb_intermediate','thumb_distal','thumb_tip',
       'index_proximal','index_intermediate','index_tip','middle_proximal','middle_intermediate','middle_tip',
       'ring_proximal','ring_intermediate','ring_tip','pinky_proximal','pinky_intermediate','pinky_tip')


def embedded(state,x,removed):
    if removed:
        x=x.copy();x[..., [12,13,14,16,17,18]]=0
    local=np.maximum(x@state['0.weight'].numpy().T+state['0.bias'].numpy(),0)
    local=np.maximum(local@state['2.weight'].numpy().T+state['2.bias'].numpy(),0)
    return np.concatenate((local,np.broadcast_to(local.mean(1,keepdims=True),local.shape)),-1)


def head_prediction(state,z,anchor):
    h=np.maximum(z@state['layers.0.weight'].numpy().T+state['layers.0.bias'].numpy(),0)
    return anchor+h@state['layers.2.weight'].numpy().T+state['layers.2.bias'].numpy()


def report(prediction,target,env):
    error=np.linalg.norm(prediction.astype(np.float64)-target.astype(np.float64),axis=-1).mean(-1)*10
    groups={str(parent):float(error[env==parent].mean()) for parent in np.sort(np.unique(env))}
    return float(np.mean(list(groups.values()))),groups


def replay_early(checkpoint,features,target):
    # NumPy gradients and AdamW, independent of the training autograd path.
    state={k:v.numpy().astype(np.float64) for k,v in checkpoint['common_initial'].items()}
    first={k:np.zeros_like(v) for k,v in state.items()};second={k:np.zeros_like(v) for k,v in state.items()}
    maximum=0.;loss_max=0.
    for step in range(3):
        ids=checkpoint['batch_schedule'][step].numpy()
        z=embedded(checkpoint['encoder_state'],features[ids],checkpoint['hand_flow_removed']).astype(np.float64)
        anchor=features[ids,:,19:22].astype(np.float64);y=target[ids].astype(np.float64)
        linear=z@state['layers.0.weight'].T+state['layers.0.bias'];hidden=np.maximum(linear,0)
        prediction=anchor+hidden@state['layers.2.weight'].T+state['layers.2.bias'];delta=prediction-y
        loss_max=max(loss_max,abs(float((delta**2).mean())-checkpoint['losses'][step]))
        g=2*delta/delta.size;flat=g.reshape(-1,3);hidden_flat=hidden.reshape(-1,64)
        upstream=(g@state['layers.2.weight'])*(linear>0)
        gradients={'layers.2.weight':flat.T@hidden_flat,'layers.2.bias':flat.sum(0),
                   'layers.0.weight':upstream.reshape(-1,64).T@z.reshape(-1,128),
                   'layers.0.bias':upstream.sum((0,1))}
        norm=np.sqrt(sum((value**2).sum() for value in gradients.values()));clip=min(1.,10/(norm+1e-6))
        for key,value in state.items():
            g=gradients[key]*clip;first[key]=.9*first[key]+.1*g;second[key]=.999*second[key]+.001*g*g
            state[key]=value*(1-3e-4*1e-4)-3e-4*(first[key]/(1-.9**(step+1)))/(np.sqrt(second[key]/(1-.999**(step+1)))+1e-8)
            error=float(np.abs(state[key]-checkpoint['early_head_states'][step][key].numpy()).max())
            maximum=max(maximum,error);assert error<5e-5,(step,key,error)
    assert loss_max<2e-5,loss_max
    return maximum,loss_max


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--prior-source',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);old=a.execution_source/'qualified';fit=a.root/'fit';r=json.loads((fit/'results.json').read_text())
    assert json.loads((a.execution_source/'audit.json').read_text())['run_status']=='COMPLETED'
    initial=torch.load(a.native_source/'s655/initial.pt',map_location='cpu',weights_only=False);trace=torch.load(a.native_source/'s655/trace.pt',map_location='cpu',weights_only=False)
    rows={};train_ids=[];held_ids=[];rng=np.random.default_rng(4001)
    for motion in range(3):
        for arm in range(4):
            env=np.flatnonzero((initial['motion'].numpy()==motion)&(initial['arm_assignment'].numpy()==arm));assert len(env)==64
            env=rng.permutation(env);train_ids.extend(env[:32]);held_ids.extend(env[32:])
    assert not set(train_ids)&set(held_ids)
    ticks=np.linspace(1,200,16,dtype=int);previous=np.concatenate((initial['object_root'][None].numpy(),trace['object_root'].numpy()),0)
    for group,ids in (('train',train_ids),('held',held_ids)):
        with np.load(old/(group+'_rows.npz')) as f:rows[group]={k:f[k] for k in f.files}
        env=np.repeat(np.sort(ids),16);tick=np.tile(ticks,384)
        expected=dict(env=env,tick=tick,q=trace['native_q'][tick-1,env].numpy(),dq=trace['native_dq'][tick-1,env].numpy(),
                      target=trace['target'][tick,env].numpy(),next_q=trace['native_q'][tick,env].numpy(),
                      current_obj=trace['object_root'][tick-1,env].numpy(),previous_obj=previous[tick-1,env],next_obj=trace['object_root'][tick,env].numpy(),
                      hand_root=trace['hand_root'][tick-1,env].numpy(),sdk_next_positions=trace['hand_body_position'][tick,env].numpy(),
                      sdk_next_quaternions=trace['hand_body_quaternion'][tick,env].numpy(),motion=initial['motion'][env].numpy(),arm=initial['arm_assignment'][env].numpy())
        for key,value in expected.items():assert np.array_equal(value,rows[group][key]),(group,key)
    with np.load(a.root/'data/train_features.npz') as f:bank={k:f[k] for k in f.files}
    with np.load(old/'features.npz') as f:held_x=f['action_velocity'];held_y=f['target']
    with np.load(old/'geometry.npz') as f:geometry={k:f[k] for k in f.files}
    actuator=json.loads((old/'execution_fit.json').read_text());train=rows['train']
    signals=np.stack((train['target']-train['q'],train['dq']/30),-1).astype(np.float64)/np.asarray(actuator['scale'])
    x=np.concatenate((signals,np.ones_like(signals[...,:1])),-1)
    predicted=train['q'].astype(np.float64)+(x*np.asarray(actuator['coefficients']['action_velocity'])[None]).sum(-1)
    predicted[:,6:]=np.clip(predicted[:,6:],initial['native_lower'].numpy()[6:],initial['native_upper'].numpy()[6:])
    assert np.array_equal(predicted.astype(np.float32),bank['predicted_q'])
    # First selected window per train episode, fixed before run. All raw/prediction rows checked separately.
    ids=np.arange(0,6144,16);urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    local=geometry['points'].astype(np.float64);normal=geometry['normals'].astype(np.float64);link=geometry['links']
    feature_max=target_max=knn_max=sdk_max=0.
    metadata=json.loads((a.native_source/'s655/physical_metadata.json').read_text());contact=[metadata['native_body_names'][i] for i in metadata['contact_body_ids']]
    def hand(q,base):
        relative=independent_links(q,urdf);poses=np.stack([base@relative[name] for name in NAMES],1)
        transform=poses[:,link];points=np.einsum('bnij,nj->bni',transform[:,:,:3,:3],local)+transform[:,:,:3,3]
        normals=np.einsum('bnij,nj->bni',transform[:,:,:3,:3],normal)
        return points,normals,poses
    for start in range(0,len(ids),16):
        select=ids[start:start+16];base=pose(train['hand_root'][select]);current,hn,_=hand(train['q'][select],base)
        nxt,_,_=hand(bank['predicted_q'][select],base);_,_,measured=hand(train['next_q'][select],base)
        sdk_error=float(np.abs(measured[:,[NAMES.index(n) for n in contact],:3,3]-train['sdk_next_positions'][select]).max());sdk_max=max(sdk_max,sdk_error);assert sdk_error<2e-4
        po=pose(train['current_obj'][select]);rotation=po[:,:3,:3]
        def obj(key):
            p=pose(train[key][select]);return geometry['object_local'][None]@p[:,:3,:3].transpose(0,2,1)+p[:,None,:3,3]
        objects=obj('current_obj');indices=bank['knn_indices'][select];batch=np.arange(len(select))[:,None,None]
        nearest=current[batch,indices];nearest_normal=hn[batch,indices]
        for i in range(len(select)):
            distance,_=cKDTree(current[i]).query(objects[i],k=4)
            err=float(np.abs(np.linalg.norm(nearest[i]-objects[i,:,None],axis=-1)-distance).max());knn_max=max(knn_max,err);assert err<1e-5
        relative=np.einsum('b...i,bij->b...j',nearest-objects[:,:,None],rotation)
        hn_local=np.einsum('b...i,bij->b...j',nearest_normal,rotation);hn_local/=np.linalg.norm(hn_local,axis=-1,keepdims=True)
        obj_normal=geometry['object_normal'][None]@rotation.transpose(0,2,1)@rotation;obj_normal/=np.linalg.norm(obj_normal,axis=-1,keepdims=True)
        obj_local=(objects-po[:,None,:3,3])@rotation
        flow=np.einsum('b...i,bij->b...j',nxt[batch,indices]-nearest,rotation).mean(2)/.01
        global_flow=((nxt-current).mean(1)[:,None]@rotation)/.01
        previous=(objects-obj('previous_obj'))@rotation/.01
        rebuilt=np.concatenate((obj_local/.05,obj_normal,relative.mean(2)/.05,hn_local.mean(2),flow,
                                np.linalg.norm(relative,axis=-1).min(-1)[...,None]/.05,np.broadcast_to(global_flow,objects.shape),previous),-1)
        y=(obj('next_obj')-objects)@rotation/.01
        value=float(np.abs(rebuilt-bank['features'][select]).max());feature_max=max(feature_max,value)
        assert np.allclose(rebuilt,bank['features'][select],atol=2e-5,rtol=2e-6),value
        value=float(np.abs(y-bank['target'][select]).max());target_max=max(target_max,value)
        assert np.allclose(y,bank['target'][select],atol=2e-5,rtol=2e-6),value
    good=torch.load(a.prior_source/'fit/mano_7168.pt',map_location='cpu',weights_only=False)
    shuffle=torch.load(a.prior_source/'fit/mano_shuffled.pt',map_location='cpu',weights_only=False)
    source={'pretrained':good['state'],'shuffled':shuffle['state'],'scratch':good['common_initial'],'hand_flow_removed':good['state']}
    assert good['steps']==shuffle['steps']==1500 and good['size']==shuffle['size']==7168
    assert torch.equal(good['batch_schedule'],shuffle['batch_schedule']) and shuffle['shuffled'] and not good['shuffled']
    assert all(torch.equal(v,shuffle['common_initial'][k]) for k,v in good['common_initial'].items())
    stored=np.load(fit/'predictions.npz');checkpoints={};metrics={};network_max=metric_max=early_max=early_loss_max=0.
    expected_schedule=torch.randint(6144,(1200,32),generator=torch.Generator().manual_seed(4102))
    masks={**{'motion_'+str(i):rows['held']['motion']==i for i in range(3)},**{'arm_'+str(i):rows['held']['arm']==i for i in range(4)},'near':held_x[...,15].min(-1)*.05<.02};masks['far']=~masks['near']
    for arm in ('pretrained','shuffled','scratch','hand_flow_removed'):
        ck=torch.load(fit/(arm+'.pt'),map_location='cpu',weights_only=False);checkpoints[arm]=ck
        assert ck['steps']==1200 and torch.equal(ck['batch_schedule'],expected_schedule) and ck['encoder_frozen']
        assert len(ck['losses'])==1200 and np.isfinite(ck['losses']).all()
        for key,value in ck['encoder_state'].items():assert torch.equal(value,source[arm]['encoder.'+key]),(arm,key)
        assert all(int(value['step'])==1200 for value in ck['optimizer']['state'].values())
        for start in range(0,6144,32):
            z=embedded(ck['encoder_state'],held_x[start:start+32],arm=='hand_flow_removed')
            prediction=head_prediction(ck['state'],z,held_x[start:start+32,:,19:22])
            error=float(np.abs(prediction-stored[arm][start:start+32]).max());network_max=max(network_max,error);assert error<2e-4
        value,groups=report(stored[arm],held_y,rows['held']['env']);metrics[arm]=value
        error=abs(value-r['reports'][arm]['parent_epe_mm']);metric_max=max(metric_max,error);assert error<1e-4
        for parent,v in groups.items():assert abs(v-r['reports'][arm]['per_parent_epe_mm'][parent])<1e-4
        for name,mask in masks.items():
            value,_=report(stored[arm][mask],held_y[mask],rows['held']['env'][mask]);assert abs(value-r['diagnostics'][name]['arms'][arm])<1e-4
            assert int(mask.sum())==r['diagnostics'][name]['windows'] and len(np.unique(rows['held']['env'][mask]))==r['diagnostics'][name]['episodes']
        early,loss=replay_early(ck,bank['features'],bank['target']);early_max=max(early_max,early);early_loss_max=max(early_loss_max,loss)
    initial=checkpoints['pretrained']['common_initial']
    torch.manual_seed(4101)
    first=torch.nn.Linear(128,64);last=torch.nn.Linear(64,3)
    expected_initial={'layers.0.weight':first.weight.detach(),'layers.0.bias':first.bias.detach(),
                      'layers.2.weight':torch.zeros_like(last.weight),'layers.2.bias':torch.zeros_like(last.bias)}
    assert all(torch.equal(v,expected_initial[k]) for k,v in initial.items())
    for ck in checkpoints.values():
        assert all(torch.equal(v,ck['common_initial'][k]) for k,v in initial.items())
        assert torch.count_nonzero(ck['common_initial']['layers.2.weight'])==0 and torch.count_nonzero(ck['common_initial']['layers.2.bias'])==0
        assert max(float((v-ck['common_initial'][k]).abs().max()) for k,v in ck['state'].items())>0
        assert sum(v.numel() for v in ck['state'].values())==8451 and sum(v.numel() for v in ck['encoder_state'].values())==11200
    persistence,_=report(held_x[...,19:22],held_y,rows['held']['env']);zero,_=report(np.zeros_like(held_y),held_y,rows['held']['env'])
    assert abs(persistence-r['baselines']['persistence']['parent_epe_mm'])<1e-4 and abs(zero-r['baselines']['zero']['parent_epe_mm'])<1e-4
    for name,mask in masks.items():
        value,_=report(held_x[mask,:,19:22],held_y[mask],rows['held']['env'][mask]);assert abs(value-r['diagnostics'][name]['persistence_mm'])<1e-4
    e=metrics['pretrained'];gates=dict(persistence=e<=.9*persistence,scratch=e<=.9*metrics['scratch'],shuffled=e<=.95*metrics['shuffled'],hand_flow=e<=.95*metrics['hand_flow_removed'])
    evidence=gates['persistence'] and (gates['scratch'] or gates['shuffled'] or gates['hand_flow'])
    label='PROMISING' if all(gates.values()) else ('UNCLEAR' if evidence else 'UNPROMISING')
    assert gates==r['gates'] and label==r['label'] and r['actual_optimizer_updates']==4800
    audit=dict(run_status='COMPLETED',label=label,raw_rows=12288,new_train_geometry_rows=384,
               all_geometry_rows_not_reconstructed=True,inherited_held_geometry_audit=True,
               feature_max=feature_max,target_max=target_max,knn_distance_max_m=knn_max,sdk_origin_max_m=sdk_max,
               full_held_numpy_prediction_max=network_max,metric_max_mm=metric_max,
               independent_numpy_adamw_steps=12,early_parameter_max=early_max,early_loss_max=early_loss_max,
               all_4800_optimizer_steps_not_replayed=True,all_encoders_unchanged=True)
    (a.root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
