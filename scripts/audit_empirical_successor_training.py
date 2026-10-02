"""Full raw FIT/physical kernel/conditional-law/value/actor independent replay."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.run_contact_response_probe import sha
from scripts.audit_option_feature_contract import independent_points
from src.task.CmResidual.option_model_policy import DYNAMIC,numpy_forward,read_options,state_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());r=json.loads((root/'fit/results.json').read_text());assert r['run_status']=='COMPLETED'
    assert sha(root/'fit/models.pt')==r['models_sha256'] and sha(root/'fit/actors.pt')==r['actors_sha256']
    model=torch.load(root/'fit/models.pt',map_location='cpu',weights_only=False);actor=torch.load(root/'fit/actors.pt',map_location='cpu',weights_only=False);packet=torch.load(root/'fit/fit_rows.pt',map_location='cpu',weights_only=False);source=Path(m['fit_source']);original=torch.load(source/'fit/models.pt',map_location='cpu',weights_only=False);control=torch.load(source/'fit/actors.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    parts=[];errors={}
    for seed in (603,604):
        d=source/f's{seed}';initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((d/'physical_metadata.json').read_text());primary=read_options(d,base);parts.append(primary)
        for name,ticks in (('current',initial['decision_steps'].numpy()),('future',initial['decision_steps'].numpy()+8)):
            compact,extra=independent_points(initial,trace,meta,base,ticks);errors[f'{seed}_{name}_compact']=float(np.abs(compact-primary[name]).max());errors[f'{seed}_{name}_sdk']=float(np.abs(extra-primary['extra' if name=='current' else 'future_extra']).max());assert errors[f'{seed}_{name}_compact']<=2e-6 and errors[f'{seed}_{name}_sdk']<=2e-5
    data={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    for key in data:assert np.array_equal(data[key],packet['data'][key])
    current=state_features(data['current'],data['extra'],original['extra_mean'],original['extra_std']);future=state_features(data['future'],data['future_extra'],original['extra_mean'],original['extra_std'])
    assert np.array_equal(current,packet['current']) and np.array_equal(future,packet['future']) and np.array_equal(current,model['support_current']) and np.array_equal(future,model['support_future']) and np.array_equal(data['raw'],model['support_raw'])
    for key in ('extra_mean','extra_std'):assert np.array_equal(model[key],original[key]) and np.array_equal(actor[key],original[key])
    actual_physical=torch.from_numpy(future[:,DYNAMIC]).double();kernel=-torch.cdist(actual_physical,actual_physical,compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/(2*.5**2)
    errors['physical_kernel']=float(np.abs(kernel-packet['kernel_logscore']).max());assert errors['physical_kernel']<=1e-8
    action=np.tanh(data['raw'].astype(np.float64));physical=future[:,DYNAMIC];known=data['known'];value_parameters=original['models']['value']
    for name,state in model['models'].items():
        inp=np.concatenate((current,action if name=='cm' else np.zeros_like(action)),-1);emb=numpy_forward(state,inp,None);errors[name+'_encoder']=float(np.abs(emb-packet['encoder_gpu_outputs'][name]).max());assert errors[name+'_encoder']<=2e-5
        p_error=0.;log_error=0.;expect_error=0.;mean_error=0.;kernel_error=0.
        for start in range(0,1536,8):
            ids=np.arange(start,min(start+8,1536));logits=-np.sum((emb[ids,None]-emb[None])**2,-1)/16;logits[np.arange(len(ids)),ids]=float('-inf');logw=logits-logsumexp(logits,-1,keepdims=True);prob=np.exp(logw);saved=packet['probabilities'][name][ids];saved_log=packet['log_probabilities'][name][ids]
            p_error=max(p_error,float(np.abs(prob-saved).max()));finite=np.isfinite(logw);assert np.array_equal(finite,np.isfinite(saved_log));log_error=max(log_error,float(np.abs(logw[finite]-saved_log[finite]).max()));assert np.allclose(logw[finite],saved_log[finite],atol=.002,rtol=2e-6)
            assert not np.any(saved[np.arange(len(ids)),ids]) and np.max(np.abs(saved.sum(-1)-1))<=2e-6
            scores=-logsumexp(saved_log.astype(np.float64)+kernel[ids],-1);kernel_error=max(kernel_error,float(np.abs(scores-packet['kernel_scores'][name][ids]).max()))
            # Use validated float32 probabilities for the separate literal
            # full 164-input value forwards. No first-layer decomposition.
            nodes=np.broadcast_to(future[None],(len(ids),1536,152)).copy();nodes[:,:,51:72]=known[ids,None]
            options=np.broadcast_to(action[ids,None],(len(ids),1536,12));vinput=np.concatenate((nodes,options),-1).reshape(-1,164);pv=numpy_forward(value_parameters,vinput,'sigmoid').reshape(len(ids),1536)
            expected=(pv*saved).sum(-1);expect_error=max(expect_error,float(np.abs(expected-packet['expected_value'][name][ids]).max()))
            mean=current[ids].astype(np.float64).copy();mean[:,51:72]=known[ids];mean[:,DYNAMIC]=saved.astype(np.float64)@physical
            mv=numpy_forward(value_parameters,np.concatenate((mean,action[ids]),-1),'sigmoid').flatten();mean_error=max(mean_error,float(np.abs(mv-packet['mean_value'][name][ids]).max()))
        errors.update({name+'_probability':p_error,name+'_logprobability':log_error,name+'_expected_value':expect_error,name+'_mean_value':mean_error,name+'_kernel_score':kernel_error})
        assert p_error<=2e-5 and expect_error<=2e-5 and mean_error<=2e-5 and kernel_error<=2e-5
    assert model['model_updates_each']==1500 and model['own_episode_excluded_in_model_and_actor_fit'] and actor['actor_updates_each']==1000
    assert all(torch.equal(actor['actors']['direct_q'][key],control['actors']['direct_q'][key]) for key in control['actors']['direct_q'])
    assert all(torch.equal(actor['actor_initial_parameters'][key],control['actor_initial_parameters'][key]) for key in control['actor_initial_parameters'])
    for name,state in actor['actors'].items():
        pred=numpy_forward(state,current,'tanh');error=float(np.abs(pred-packet['actor_gpu_outputs'][name]).max());errors[name+'_actor']=error;assert error<=2e-5
    assert r['new_physical_optimizer_steps']==3000 and r['new_actor_optimizer_steps']==2000 and r['reused_direct_q_actor_steps']==1000
    result=dict(run_status='COMPLETED',all_fit_current_and_successor_raw_sdk_rebuilt=True,fit_only_support_and_normalizers_exact=True,physical_kernel_independently_rebuilt=True,all_encoder_probabilities_and_self_exclusion_replayed=True,all_value_sum_and_mean_literal_forward_replayed=True,validated_float32_probabilities_reused_for_composition=True,all_final_actor_forwards_replayed=True,direct_q_weights_exact_source=True,same_initial_actor_parameters=True,independent_optimizer_replay=False,maximum_errors=errors,wall_seconds=time.monotonic()-begin)
    (root/'training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
