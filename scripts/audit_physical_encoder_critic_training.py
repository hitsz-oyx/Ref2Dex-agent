"""Independent full FIT inputs, physical encoder transfer and task-Q/actor forwards."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.audit_option_feature_contract import independent_points
from src.task.CmResidual.option_model_policy import numpy_forward,read_options,state_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-physical-encoder-critic';r=json.loads((root/'fit/results.json').read_text());assert r['run_status']=='COMPLETED'
    assert sha(root/'fit/models.pt')==r['models_sha256'] and sha(root/'fit/actors.pt')==r['actors_sha256']
    model=torch.load(root/'fit/models.pt',map_location='cpu',weights_only=False);actor=torch.load(root/'fit/actors.pt',map_location='cpu',weights_only=False);packet=torch.load(root/'fit/fit_rows.pt',map_location='cpu',weights_only=False);source=Path(m['fit_source']);previous=torch.load(source/'fit/fit_rows.pt',map_location='cpu',weights_only=False);original=torch.load(source/'fit/models.pt',map_location='cpu',weights_only=False);control=torch.load(source/'fit/actors.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    parts=[];errors={}
    for seed in (603,604):
        d=source/f's{seed}';initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((d/'physical_metadata.json').read_text());primary=read_options(d,base);parts.append(primary)
        for name,ticks in (('current',initial['decision_steps'].numpy()),('future',initial['decision_steps'].numpy()+8)):
            compact,extra=independent_points(initial,trace,meta,base,ticks);errors[f'{seed}_{name}_compact']=float(np.abs(compact-primary[name]).max());errors[f'{seed}_{name}_sdk']=float(np.abs(extra-primary['extra' if name=='current' else 'future_extra']).max());assert errors[f'{seed}_{name}_compact']<=2e-6 and errors[f'{seed}_{name}_sdk']<=2e-5
    data={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    for key in data:assert np.array_equal(data[key],packet['data'][key])
    current=state_features(data['current'],data['extra'],original['extra_mean'],original['extra_std']);future=state_features(data['future'],data['future_extra'],original['extra_mean'],original['extra_std'])
    assert np.array_equal(current,packet['current']) and np.array_equal(current,previous['current']) and np.array_equal(future,previous['future']) and np.array_equal(data['raw'],packet['data']['raw'])
    for key in ('extra_mean','extra_std'):assert np.array_equal(model[key],original[key]) and np.array_equal(actor[key],original[key])
    from src.task.CmResidual.option_model_policy import initialized_network
    head=initialized_network('value').state_dict()
    for name in ('cm','dynamics_off'):
        state=model['initial_parameters'][name]
        for key in ('0.weight','0.bias','2.weight','2.bias'):assert torch.equal(state[key],original['models'][name][key])
        for key in ('4.weight','4.bias'):assert torch.equal(state[key],head[key])
        assert model['critics'][name]['4.weight'].shape==(1,64)
    action=np.tanh(data['raw'].astype(np.float64));qinput=np.concatenate((current,action),-1)
    errors['task_input']=float(np.abs(qinput-packet['task_inputs']).max());assert errors['task_input']<=2e-7
    for name,state in model['critics'].items():
        pred=numpy_forward(state,qinput,'sigmoid');error=float(np.abs(pred-packet['critic_gpu_outputs'][name]).max());errors[name+'_task_critic']=error;assert error<=2e-5
        assert max(float((state[key]-model['initial_parameters'][name][key]).abs().max()) for key in state)>0
    assert model['task_critic_updates_each']==1500 and model['physical_decoder_discarded'] and model['all_critic_parameters_trainable'] and model['measured_returns_only']
    assert actor['actor_updates_each']==1000 and not actor['deploy_model_or_future_input']
    assert all(torch.equal(actor['actors']['direct_q'][key],control['actors']['direct_q'][key]) for key in control['actors']['direct_q'])
    assert all(torch.equal(actor['actor_initial_parameters'][key],control['actor_initial_parameters'][key]) for key in control['actor_initial_parameters'])
    for name,state in actor['actors'].items():
        pred=numpy_forward(state,current,'tanh');error=float(np.abs(pred-packet['actor_gpu_outputs'][name]).max());errors[name+'_actor']=error;assert error<=2e-5
    assert r['new_task_critic_optimizer_steps']==3000 and r['new_actor_optimizer_steps']==2000 and r['reused_direct_q_actor_steps']==1000
    result=dict(run_status='COMPLETED',all_fit_current_and_successor_raw_sdk_rebuilt=True,fit_only_normalizers_exact=True,initial_physical_encoder_copies_exact=True,common_fresh_task_head_exact=True,physical_decoder_removed=True,all_task_critic_and_actor_forwards_replayed=True,both_task_critic_inputs_include_executed_action=True,direct_q_weights_exact_source=True,same_initial_actor_parameters=True,independent_optimizer_replay=False,maximum_errors=errors,wall_seconds=time.monotonic()-begin)
    (root/'training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
