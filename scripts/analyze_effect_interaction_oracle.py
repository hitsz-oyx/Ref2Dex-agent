"""All-episode actual deployment, query fidelity and independent Q inference audit."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    import torch,numpy as np
    from src.task.CmResidual.oracle_features import load,descriptor,ARMS,DECISION,HORIZON
    from scripts.run_contact_response_probe import sha
    out=a.source;packet=load(out/'fit/feature_packet.pt');cp=load(out/'fit/selector.pt');options=np.load(out/'options.npy')
    eval_packet=packet['eval'];motion=eval_packet['motion'];counts={};rows=[];audit={};valid=True
    limits=dict(native_q=1e-6,native_dq=1e-5,object_position=1e-6,object_quaternion=1e-6,object_velocity=1e-5,rigid_position=1e-6,rigid_quaternion=1e-6,rigid_velocity=1e-5)
    def count(labels):return dict(successes=int(labels.sum()),episodes=len(labels),by_motion=[int(labels[motion==m].sum()) for m in range(3)])
    counts['P0']=count(eval_packet['labels'][0])
    mean=cp['mean'].numpy();std=cp['std'].numpy()
    z=np.clip((eval_packet['features'].reshape(-1,len(mean))-mean)/std,-10,10)
    for arm in ARMS:
        v=z.copy();c,e=cp['common_dim'],cp['effect_dim']
        if arm in ('state','interaction'):v[:,c:c+e]=0
        if arm in ('state','effect'):v[:,c+e:]=0
        weights=cp['models'][arm]['model']
        for layer in (0,2,4):
            v=v@weights['network.%d.weight'%layer].numpy().T+weights['network.%d.bias'%layer].numpy()
            if layer!=4:v=np.maximum(v,0)
        probabilities=(1/(1+np.exp(-np.clip(v,-80,80)))).reshape(8,-1)
        neural_error=float(np.max(np.abs(probabilities-cp['scores'][arm])))
        choice=np.load(out/'fit'/(arm+'_choices.npy'));choices_match=np.array_equal(probabilities.argmax(0),choice)
        selected_options=options[choice]
        if not np.array_equal(selected_options,np.load(out/'fit'/(arm+'_options.npy'))):raise ValueError('selected actual executable requests')
        deploy=out/'deploy'/arm;features=descriptor(deploy,out/'eval/baseline',selected_options)
        counts[arm]=count(features['labels']);actual=load(deploy/'trace.pt')
        errors={key:0. for key in limits}
        for index in range(8):
            ids=np.flatnonzero(choice==index)
            if not len(ids):continue
            query=load(out/'eval'/('baseline' if index==0 else 'c%02d'%index)/'trace.pt')
            for key,slice_,name in (('native_q',slice(None),'native_q'),('native_dq',slice(None),'native_dq'),('object_root',slice(0,3),'object_position'),('object_root',slice(3,7),'object_quaternion'),('object_root',slice(7,13),'object_velocity'),('rigid_state',slice(0,3),'rigid_position'),('rigid_state',slice(3,7),'rigid_quaternion'),('rigid_state',slice(7,13),'rigid_velocity')):
                delta=(actual[key][DECISION:DECISION+HORIZON,ids,...,slice_]-query[key][DECISION:DECISION+HORIZON,ids,...,slice_]).abs()
                errors[name]=max(errors[name],float(delta.max()))
        wanted=eval_packet['features'][choice,np.arange(len(choice))]
        common_end=cp['common_dim']+cp['effect_dim']
        feature_errors=dict(common=float(np.max(np.abs(features['features'][:,:cp['common_dim']]-wanted[:,:cp['common_dim']]))),effect=float(np.max(np.abs(features['features'][:,cp['common_dim']:common_end]-wanted[:,cp['common_dim']:common_end]))),interaction=float(np.max(np.abs(features['features'][:,common_end:]-wanted[:,common_end:]))))
        pass_=all(errors[k]<=limits[k] for k in limits) and feature_errors['effect']<=1e-4 and feature_errors['interaction']<=1e-4 and neural_error<=5e-5 and choices_match
        valid &= pass_
        audit[arm]=dict(query_deployed_max_errors=errors,feature_max_errors=feature_errors,independent_numpy_probability_max_error=neural_error,independent_choices_match=choices_match,conditional_oracle_contract_pass=bool(pass_),choices=np.bincount(choice,minlength=8).tolist())
        for env in range(len(choice)):rows.append(dict(arm=arm,environment=env,motion=int(motion[env]),option=int(choice[env]),success=bool(features['labels'][env])))
    n=counts['joint']['episodes'];gates=dict(joint_vs_state_5pp=(counts['joint']['successes']-counts['state']['successes'])/n>=.05,joint_vs_effect_5pp=(counts['joint']['successes']-counts['effect']['successes'])/n>=.05,joint_at_least_P0=counts['joint']['successes']>=counts['P0']['successes'],oracle_contract_verified=bool(valid))
    conclusion='PROMISING' if all(gates.values()) else ('UNPROMISING' if valid else 'UNCLEAR')
    result=dict(run_status='COMPLETED',conclusion=conclusion,counts=counts,gates=gates,audit=audit,query_deployed_absolute_limits=limits,descriptor_error_limit=1e-4,numpy_probability_limit=5e-5,actual_optimizer_steps=6000,cm_updates=0,actor_updates=0,source_actor_weights_used=False,terminal_success_not_input_to_selection=True,one_shot_oracle_only=True,formal_upper_bound=False,all_motion0_included=True,input_sha256={str(out/'fit/selector.pt'):sha(out/'fit/selector.pt'),str(out/'fit/feature_packet.pt'):sha(out/'fit/feature_packet.pt')})
    with (out/'results.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    with (out/'rows.json').open('x') as f:f.write(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
