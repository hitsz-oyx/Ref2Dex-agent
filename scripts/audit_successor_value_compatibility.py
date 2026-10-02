"""Independent observed successors and all fixed-model forwards/statistics."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.audit_option_feature_contract import independent_points
from src.task.CmResidual.option_model_policy import DYNAMIC,numpy_forward,state_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());r=json.loads((root/'results.json').read_text());assert sha(root/'predictions.pt')==r['predictions_sha256']
    packet=torch.load(root/'predictions.pt',map_location='cpu',weights_only=False);cp=torch.load(m['source_models'],map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    d=Path(m['source_run'])/'s618';initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);metadata=json.loads((d/'physical_metadata.json').read_text());rows=json.loads((d/'rows.json').read_text());data=packet['data'];errors={}
    tick=initial['decision_steps'].numpy();env=np.arange(768);motion=initial['motion'].numpy();stop=initial['phase_stop'].numpy()[motion]
    for name,ticks in (('current',tick),('future',tick+8)):
        compact,extra=independent_points(initial,trace,metadata,base,ticks)
        errors[name+'_compact']=float(np.abs(compact-data[name]).max());errors[name+'_sdk']=float(np.abs(extra-data['extra' if name=='current' else 'future_extra']).max())
        assert errors[name+'_compact']<=2e-6 and errors[name+'_sdk']<=2e-5
        inputs=state_features(data[name],data['extra' if name=='current' else 'future_extra'],cp['extra_mean'],cp['extra_std']);assert np.array_equal(inputs,packet[name])
    progress=np.minimum(tick+9,stop);ref=initial['native_reference_q'].numpy()[motion,progress];planned=np.concatenate((ref,(progress/stop).astype(np.float32)[:,None]),-1)
    known=np.concatenate((np.clip((planned-base['observation_mean'].numpy()[51:70])/base['observation_std'].numpy()[51:70],-10,10),np.ones((768,1),np.float32),((stop+30-tick-8)/np.float32(202))[:,None]),-1).astype(np.float32)
    assert np.array_equal(known,data['known']) and np.max(np.abs(known-data['future'][:,51:72]))<=2e-6
    raw=initial['option_raw12'].numpy();reward=np.array([row['physical105'] for row in rows],np.float32);assert np.array_equal(raw,data['raw']) and np.array_equal(reward,data['reward'])
    roots=trace['object_root'].numpy();valid=(roots[...,2]-initial['initial_height'].numpy()[None]>=np.float32(.03))&(trace['clearance'].numpy()>=np.float32(.02));window=(trace['progress'].numpy()>=stop[None]-74)&(trace['progress'].numpy()<=stop[None]+30)
    assert (window.sum(0)==105).all() and np.array_equal(reward.astype(bool),~(window&~valid).any(0))
    x=packet['current'].astype(np.float64);action=np.tanh(raw.astype(np.float64));states=cp['models'];pred={}
    pred['direct_q']=numpy_forward(states['direct_q'],np.concatenate((x,action),-1),'sigmoid').flatten()
    pred['oracle']=numpy_forward(states['value'],np.concatenate((packet['future'],action),-1),'sigmoid').flatten()
    for name in ('cm','dynamics_off'):
        z=numpy_forward(states[name],np.concatenate((x,action if name=='cm' else np.zeros_like(action)),-1),None);future=x.copy();future[:,51:72]=known;future[:,DYNAMIC]=np.clip(x[:,DYNAMIC]+cp['delta_mean']+cp['delta_std']*z,-10,10)
        pred[name]=numpy_forward(states['value'],np.concatenate((future,action),-1),'sigmoid').flatten()
    for name,v in pred.items():errors[name+'_forward']=float(np.abs(v-packet['predictions'][name]).max());assert errors[name+'_forward']<=2e-5
    losses={name:(v.astype(np.float64)-reward)**2 for name,v in packet['predictions'].items()};brier={name:float(v.mean()) for name,v in losses.items()};assert brier==r['brier']
    generator=torch.Generator(device='cpu').manual_seed(3558);indices=np.concatenate([np.flatnonzero(motion==j)[torch.randint(256,(1000,256),generator=generator).numpy()] for j in range(3)],1);assert np.array_equal(indices,packet['bootstrap_indices'])
    intervals={name:np.quantile(delta[indices].mean(-1),(.025,.975)).tolist() for name,delta in {'oracle_minus_direct_q':losses['oracle']-losses['direct_q'],'cm_minus_oracle':losses['cm']-losses['oracle']}.items()};assert intervals==r['paired_bootstrap_intervals']
    gates=dict(oracle_gain10percent=brier['oracle']<=.9*brier['direct_q'],oracle_interval_upper_negative=intervals['oracle_minus_direct_q'][1]<0,cm_excess10percent_over_oracle=brier['cm']>=1.1*brier['oracle'],cm_minus_oracle_interval_lower_positive=intervals['cm_minus_oracle'][0]>0)
    useful=gates['oracle_gain10percent'] and gates['oracle_interval_upper_negative'];gap=gates['cm_excess10percent_over_oracle'] and gates['cm_minus_oracle_interval_lower_positive'];assert gates==r['gates'] and r['label']==('UNPROMISING' if not useful else ('PROMISING' if gap else 'UNCLEAR'))
    result=dict(run_status='COMPLETED',current_and_actual_successor_sdk_rebuilt=True,known_future_plan_and_physical105_rebuilt=True,all_network_predictions_replayed=True,all_statistics_bootstrap_and_gates_exact=True,float32_predictions_shared_after_independent_forward_check=True,maximum_errors=errors,reused_dataset=True,new_optimizer_steps=0,wall_seconds=time.monotonic()-begin)
    (root/'compatibility_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
