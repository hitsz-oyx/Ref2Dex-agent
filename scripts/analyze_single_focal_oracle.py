"""Actual independent subject deployments and same-world truth fidelity."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args()
    import numpy as np,torch
    from src.task.CmResidual.oracle_features import load,descriptor,ARMS
    out=a.source;cp=load(a.models);packet=load(out/'selection/feature_packet.pt');scores=load(out/'selection/scores.pt');choices=json.loads((out/'selection/choices.json').read_text());motion=packet['motion'];counts={};audits={};rows=[];valid=True
    selection_meta=json.loads((out/'selection/results.json').read_text());stable=selection_meta.get('score_order')=='raw_logit';gpu_logits=load(out/'selection/logits.pt') if stable else None
    def labels(path):
        t=load(path/'trace.pt');i=load(path/'initial.pt');s=i['phase_stop'][i['motion']];mask=(torch.arange(202)[:,None]>=s[None,:]-74)&(torch.arange(202)[:,None]<=s[None,:]+30);good=(t['object_root'][:,:,2]-i['initial_height'][None,:]>=.03)&(t['clearance']>=.02)
        if not torch.equal(mask.sum(0),torch.full((12,),105)):raise ValueError('full105')
        return (good|~mask).all(0).numpy()
    def count(y):return dict(successes=int(sum(y)),episodes=12,by_motion=[int(np.asarray(y)[motion==m].sum()) for m in range(3)])
    baseline_labels=labels(out/'baseline');counts['P0']=count(baseline_labels)
    original=packet['features'];normalized=np.clip((original.reshape(-1,cp['input_dim'])-cp['mean'].numpy())/cp['std'].numpy(),-10,10)
    cache={};query_errors={}
    for arm in ARMS:
        x=normalized.copy();c,e=cp['common_dim'],cp['effect_dim']
        if arm in ('state','interaction'):x[:,c:c+e]=0
        if arm in ('state','effect'):x[:,c+e:]=0
        weights=cp['models'][arm]['model']
        for layer in (0,2,4):
            x=x@weights['network.%d.weight'%layer].numpy().T+weights['network.%d.bias'%layer].numpy()
            if layer!=4:x=np.maximum(x,0)
        logits=x.reshape(12,8);probabilities=(1/(1+np.exp(-np.clip(x,-80,80)))).reshape(12,8);error=float(np.max(np.abs(probabilities-scores[arm])));match=np.array_equal((logits if stable else probabilities).argmax(1),choices[arm]);logit_error=float(np.max(np.abs(logits-gpu_logits[arm]))) if stable else None;valid&=error<=5e-5 and match and (not stable or logit_error<=1e-4);y=[]
        for target,index in enumerate(choices[arm]):
            name='e%02d_c%02d'%(target,index)
            if index==0:path=out/'baseline';success=bool(baseline_labels[target])
            else:
                path=out/'deploy'/name
                if name not in cache:
                    actual=load(path/'trace.pt');query=load(out/'queries'/name/'trace.pt')
                    errors={key:float((actual[key][:68]-query[key]).abs().max()) for key in ('object_root','native_q','native_dq','rigid_state','net_force','target','action')}
                    exact=all(value==0 for value in errors.values());valid&=exact
                    requests=np.load(out/'requests'/(name+'.npy'));d=descriptor(path,out/'baseline',requests,include_labels=False);feature_error=float(np.max(np.abs(d['features'][target]-original[target,index])));valid&=feature_error<=1e-4
                    query_errors[name]=dict(entire_world_first68_errors=errors,entire_world_bit_identical=exact,subject_descriptor_max_error=feature_error)
                    cache[name]=labels(path)
                success=bool(cache[name][target])
            y.append(success);rows.append(dict(arm=arm,subject=target,motion=int(motion[target]),option=index,success=success,actual_rollout=str(path)))
        counts[arm]=count(y);audits[arm]=dict(numpy_probability_max_error=error,numpy_logit_max_error=logit_error,score_order='raw_logit' if stable else 'sigmoid_probability',all_choices_match=bool(match))
    gates=dict(joint_vs_state_5pp=(counts['joint']['successes']-counts['state']['successes'])/12>=.05,joint_vs_effect_5pp=(counts['joint']['successes']-counts['effect']['successes'])/12>=.05,joint_at_least_P0=counts['joint']['successes']>=counts['P0']['successes'],oracle_contract_verified=bool(valid))
    result=dict(run_status='COMPLETED',conclusion='PROMISING' if all(gates.values()) else ('UNPROMISING' if valid else 'UNCLEAR'),score_order='raw_logit' if stable else 'sigmoid_probability',counts=counts,gates=gates,query_fidelity=query_errors,neural_audit=audits,actual_new_optimizer_steps=0,inherited_optimizer_steps=6000,unique_nonzero_deployed_worlds=len(cache),all12subjects_included=True,no_terminal_success_input_to_selection=True,frozen_model_transfer_to_fixed_background=True,no_mathematical_upper_bound=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(conclusion=result['conclusion'],counts=counts,gates=gates)),flush=True)

if __name__=='__main__':main()
