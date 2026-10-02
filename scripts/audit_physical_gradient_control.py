"""Independent live feature/critic derivative/complete actor gradient/gate audit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.audit_option_feature_contract import independent_points
from src.task.CmResidual.option_model_policy import DYNAMIC,numpy_forward,state_features
from src.task.CmResidual.return_corrected_gradient import numpy_reverse,raw_gradients


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());r=json.loads((root/'results.json').read_text());assert sha(root/'gradients.pt')==r['gradients_sha256']
    packet=torch.load(root/'gradients.pt',map_location='cpu',weights_only=False);cp=torch.load(m['source_models'],map_location='cpu',weights_only=False);actor=torch.load(m['source_actors'],map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    d=root/'s618';initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);metadata=json.loads((d/'physical_metadata.json').read_text());rows=json.loads((d/'rows.json').read_text())
    ticks=initial['decision_steps'].numpy();motion=initial['motion'].numpy();stop=initial['phase_stop'].numpy()[motion];compact,extra=independent_points(initial,trace,metadata,base,ticks)
    errors=dict(compact=float(np.abs(compact-packet['compact']).max()),sdk=float(np.abs(extra-packet['extra']).max()));assert errors['compact']<=2e-6 and errors['sdk']<=2e-5
    current=state_features(packet['compact'],packet['extra'],cp['extra_mean'],cp['extra_std']);assert np.array_equal(current,packet['current'])
    progress=np.minimum(ticks+9,stop);ref=initial['native_reference_q'].numpy()[motion,progress]
    planned=np.concatenate((ref,(progress/stop).astype(np.float32)[:,None]),-1)
    known=np.concatenate((np.clip((planned-base['observation_mean'].numpy()[51:70])/base['observation_std'].numpy()[51:70],-10,10),np.ones((768,1),np.float32),((stop+30-ticks-8)/np.float32(202))[:,None]),-1).astype(np.float32)
    assert np.array_equal(known,packet['known'])
    mask=initial['policy_group'].numpy()>0;assert np.array_equal(mask,packet['mask']) and mask.sum()==576
    epsilon=initial['option_raw12'].numpy()[mask];reward=np.array([row['physical105'] for row in rows],np.float32)[mask]
    assert np.array_equal(epsilon,packet['epsilon']) and np.array_equal(reward,packet['reward'])
    x=current[mask].astype(np.float64);zeros=np.zeros((576,12));input_q=np.concatenate((x,zeros),-1);states=cp['models'];derivatives={}
    direct=numpy_forward(states['direct_q'],input_q,'sigmoid').flatten();derivatives['direct_q']=numpy_reverse(states['direct_q'],input_q,np.ones((576,1)),'sigmoid')[:,152:]
    errors['direct_q_forward']=float(np.abs(direct-packet['baseline']).max());assert errors['direct_q_forward']<=2e-5
    for name in ('cm','dynamics_off'):
        dyn=numpy_forward(states[name],input_q,None);unclipped=x[:,DYNAMIC]+cp['delta_mean']+cp['delta_std']*dyn
        future=x.copy();future[:,51:72]=known[mask];future[:,DYNAMIC]=np.clip(unclipped,-10,10);value_input=np.concatenate((future,zeros),-1)
        value=numpy_forward(states['value'],value_input,'sigmoid').flatten();value_gradient=numpy_reverse(states['value'],value_input,np.ones((576,1)),'sigmoid')
        errors[name+'_forward']=float(np.abs(value-packet['critic_outputs'][name]).max());assert errors[name+'_forward']<=2e-5
        derivative=value_gradient[:,152:]
        if name=='cm':
            physical=value_gradient[:,DYNAMIC]*cp['delta_std']*((unclipped>=-10)&(unclipped<=10))
            derivative=derivative+numpy_reverse(states[name],input_q,physical)[:,152:]
        derivatives[name]=derivative
    for name,b in derivatives.items():
        errors[name+'_raw_derivative']=float(np.abs(b-packet['derivatives'][name]).max());assert errors[name+'_raw_derivative']<=2e-5
    # Independently validated float32 critic outputs/derivatives retain the
    # exact primary roundoff for subsequent vector and gradient statistics.
    baseline=packet['baseline'];vectors={'baseline':(reward-baseline)[:,None]*epsilon}
    for name in derivatives:vectors[name]=raw_gradients(reward,baseline,epsilon,packet['derivatives'][name])
    initial_params=actor['actor_initial_parameters'];assert not initial_params['4.weight'].any() and not initial_params['4.bias'].any()
    hidden=np.maximum(x@initial_params['0.weight'].numpy().T+initial_params['0.bias'].numpy(),0)
    hidden=np.maximum(hidden@initial_params['2.weight'].numpy().T+initial_params['2.bias'].numpy(),0)
    parameter_count=sum(v.numel() for v in initial_params.values());assert parameter_count==r['actor_parameter_count']
    active={}
    for name,vector in vectors.items():
        errors[name+'_raw_vector']=float(np.abs(vector-packet['raw_gradient_vectors'][name]).max());assert errors[name+'_raw_vector']<=2e-5
        # Check the COMPLETE zero-head actor Jacobian: four hidden blocks zero,
        # final weights v_i*h_j and final biases v_i, all episodes/dimensions.
        saved=packet['parameter_gradients'][name]
        assert set(saved)==set(initial_params)
        for key in ('0.weight','0.bias','2.weight','2.bias'):
            assert saved[key].shape==(576,*initial_params[key].shape) and not saved[key].any()
        v=packet['raw_gradient_vectors'][name].astype(np.float64)
        error=max(float(np.abs(v[:,:,None]*hidden[:,None,:]-saved['4.weight'].numpy()).max()),float(np.abs(v-saved['4.bias'].numpy()).max()))
        errors[name+'_complete_actor_gradient']=error;assert error<=2e-5
        active[name]=np.concatenate((saved['4.weight'].numpy().reshape(576,-1),saved['4.bias'].numpy()),-1).astype(np.float64)
    generator=torch.Generator(device='cpu').manual_seed(3557);selected_motion=motion[mask]
    idx=np.concatenate([np.flatnonzero(selected_motion==j)[torch.randint(192,(1000,192),generator=generator).numpy()] for j in range(3)],1);assert np.array_equal(idx,packet['bootstrap_indices'])
    weights=np.zeros((1000,576));np.add.at(weights,(np.arange(1000)[:,None],idx),1/576)
    variance={};bootstrap={}
    for name,g in active.items():
        norm=(g*g).sum(-1);variance[name]=float((norm.mean()-(g.mean(0)**2).sum())*576/575)
        assert np.isclose(variance[name],r['complete_parameter_gradient_variance'][name],rtol=1e-10,atol=1e-10)
        bootstrap[name]=(weights@norm-((weights@g)**2).sum(-1))*576/575
    intervals={name:np.quantile(bootstrap['cm']-bootstrap[name],(.025,.975)).tolist() for name in ('baseline','dynamics_off','direct_q')}
    for name,interval in intervals.items():assert np.allclose(interval,r['paired_bootstrap_interval_cm_minus_control'][name],rtol=1e-10,atol=1e-10)
    gates=dict(gain10percent_all=all(variance['cm']<=.9*variance[name] for name in intervals),paired_interval_upper_negative_all=all(v[1]<0 for v in intervals.values()))
    assert gates==r['gates'] and r['label']==('PROMISING' if all(gates.values()) else 'UNPROMISING')
    result=dict(run_status='COMPLETED',current_sdk_and_public_plan_rebuilt=True,all_critic_forwards_and_raw_derivatives_replayed=True,all_complete_actor_parameter_gradients_replayed=True,float32_critic_vectors_shared_after_independent_reverse_check=True,all_variance_bootstrap_and_gates_rebuilt=True,maximum_errors=errors,new_optimizer_steps=0,no_actual_future_physics_input=True,wall_seconds=time.monotonic()-begin)
    (root/'gradient_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
