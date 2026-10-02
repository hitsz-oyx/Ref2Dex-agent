"""Independent decision/successor features and full final model/actor replay."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.audit_option_feature_contract import independent_points
from src.task.CmResidual.option_model_policy import DYNAMIC,HORIZON,VARIANTS,read_options,state_features,known_plan,numpy_forward


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());r=json.loads((root/'fit/results.json').read_text());assert r['run_status']=='COMPLETED'
    assert sha(root/'fit/models.pt')==r['models_sha256'] and sha(root/'fit/actors.pt')==r['actors_sha256']
    cp=torch.load(root/'fit/models.pt',map_location='cpu',weights_only=False);actor=torch.load(root/'fit/actors.pt',map_location='cpu',weights_only=False);packet=torch.load(root/'fit/fit_rows.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    errors={};parts=[]
    for seed in (603,604):
        d=root/f's{seed}';i=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((d/'physical_metadata.json').read_text());primary=read_options(d,base);parts.append(primary)
        for name,ticks in [('current',i['decision_steps'].numpy()),('future',i['decision_steps'].numpy()+HORIZON)]:
            compact,extra=independent_points(i,t,meta,base,ticks)
            errors[f'{seed}_{name}_compact']=float(np.abs(compact-primary[name]).max());errors[f'{seed}_{name}_sdk']=float(np.abs(extra-primary['extra' if name=='current' else 'future_extra']).max())
            assert errors[f'{seed}_{name}_compact']<=2e-6 and errors[f'{seed}_{name}_sdk']<=2e-5
        known=known_plan(i,i['decision_steps'].numpy()+HORIZON,base)
        assert np.array_equal(known,primary['known'])
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for key in data:assert np.array_equal(data[key],packet['raw_data'][key]),key
    extras=np.concatenate((data['extra'],data['future_extra']));mean=extras.mean(0);std=extras.std(0).clip(.001)
    assert np.array_equal(mean,cp['extra_mean']) and np.array_equal(std,cp['extra_std']) and np.array_equal(mean,actor['extra_mean']) and np.array_equal(std,actor['extra_std'])
    current=state_features(data['current'],data['extra'],mean,std);future=state_features(data['future'],data['future_extra'],mean,std);residual=future[:,DYNAMIC]-current[:,DYNAMIC]
    assert np.array_equal(current,packet['current']) and np.array_equal(future,packet['future'])
    assert np.array_equal(residual.mean(0),cp['delta_mean']) and np.array_equal(residual.std(0).clip(.001),cp['delta_std']) and np.array_equal(DYNAMIC,cp['physical_indices'])
    assert cp['model_updates_each']==1500 and actor['actor_updates_each']==1000 and r['actual_model_optimizer_steps']==6000 and r['actual_actor_optimizer_steps']==3000
    # Independently compare all final predictions against the recorded GPU
    # outputs; this does not replay optimizer steps.
    action=np.tanh(data['raw'])
    model_diagnostics={}
    for name,state in cp['models'].items():
        x=np.concatenate((future if name=='value' else current,np.zeros_like(action) if name=='dynamics_off' else action),-1)
        pred=numpy_forward(state,x,'sigmoid' if name in ('value','direct_q') else None);assert np.isfinite(pred).all()
        error=float(np.abs(pred-packet['model_gpu_outputs'][name]).max());assert error<=2e-5
        model_diagnostics[name]=dict(minimum=float(pred.min()),maximum=float(pred.max()),gpu_numpy_maximum=error)
    actor_diagnostics={}
    for name,state in actor['actors'].items():
        pred=numpy_forward(state,current,'tanh');assert np.isfinite(pred).all() and np.abs(pred).max()<=1+1e-6
        initial=actor['actor_initial_parameters'];delta=max(float((state[k]-initial[k]).abs().max()) for k in initial)
        assert abs(delta-r['actor_parameter_max_change'][name])<=1e-12
        error=float(np.abs(pred-packet['actor_gpu_outputs'][name]).max());assert error<=2e-5
        actor_diagnostics[name]=dict(maximum_absolute=float(np.abs(pred).max()),parameter_max_change=delta,gpu_numpy_maximum=error)
    result=dict(run_status='COMPLETED',all_current_and_successor_sdk_inputs_rebuilt=True,reconstruction_maximum=errors,all_fit_statistics_exact=True,fit_statistics_share_primary_float32_decoder_after_independent_check=True,known_future_plan_rebuilt_without_physics=True,full_final_models_and_actors_numpy_replayed=True,independent_optimizer_replay=False,model_diagnostics=model_diagnostics,actor_diagnostics=actor_diagnostics,source_training_result_sha256=sha(root/'fit/results.json'),wall_seconds=time.monotonic()-begin)
    (root/'training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
