#!/usr/bin/env python3
"""Independent evaluated-policy feature/logit replay and physical utility gate."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
from scripts.audit_support_feature_update import feature_inputs,forward


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    root=a.directory;begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text())
    heads=root/'u12/policy_heads.pt';checkpoint=torch.load(heads,map_location='cpu',weights_only=False)
    if checkpoint['updates']!=12:raise ValueError('final-only macro')
    arrays=[];maximum=dict(feature=0.,logit=0.)
    for seed in (541,542):
        panel=root/f's{seed}';data=torch.load(panel/'audited.pt',map_location='cpu',weights_only=False)
        audit=json.loads((panel/'panel_audit.json').read_text());r=json.loads((panel/'results.json').read_text())
        if audit['run_status']!='COMPLETED' or sha(panel/'audited.pt')!=audit['dataset_sha256'] or r['macro_checkpoint_sha256']!=sha(heads):raise ValueError('audited evaluation provenance')
        choice=torch.load(panel/'decisions.pt',map_location='cpu',weights_only=False)
        features=feature_inputs(data['state'].numpy(),data['motion'].numpy(),Path(m['physical_source']))
        expected=np.zeros(768,dtype=np.int64)
        for index,variant in enumerate(('cm','state_only','global_motion_arm'),1):
            saved_input=choice['inputs'][variant].numpy();saved_logit=choice['logits'][variant].numpy()
            maximum['feature']=max(maximum['feature'],float(np.abs(features[variant]-saved_input).max()))
            logit,_=forward(saved_input,checkpoint['variants'][variant]['model'])
            maximum['logit']=max(maximum['logit'],float(np.abs(logit-saved_logit).max()))
            # Check the actual GPU argmax separately from numeric tolerance.
            if not np.array_equal(logit.argmax(-1),saved_logit.argmax(-1)):raise ValueError('independent argmax ambiguity')
            ids=data['policy_group'].numpy()==index;expected[ids]=logit[ids].argmax(-1)
        if not np.array_equal(expected,data['arm'].numpy()) or not np.array_equal(expected,choice['macro_action'].numpy()):raise ValueError('executed macro selection differs')
        arrays.append({k:data[k].numpy() for k in ('seed','motion','policy_group','physical105','arm')})
    if maximum['feature']>2e-5 or maximum['logit']>3e-5:raise ValueError(('independent decision replay',maximum))
    data={k:np.concatenate([d[k] for d in arrays]) for k in arrays[0]}
    def summarize(mask):
        output={}
        for group,name in enumerate(('unchanged','cm','state_only','global_motion_arm')):
            ids=mask & (data['policy_group']==group)
            output[name]=dict(n=int(ids.sum()),physical105_count=int(data['physical105'][ids].sum()),physical105_rate=float(data['physical105'][ids].mean()),
                chosen_macro_count=np.bincount(data['arm'][ids],minlength=8).tolist())
        return output
    pooled=summarize(np.ones(1536,dtype=bool));seeds={str(s):summarize(data['seed']==s) for s in (541,542)}
    motions={str(mo):summarize(data['motion']==mo) for mo in range(3)}
    diff={v:pooled['cm']['physical105_rate']-pooled[v]['physical105_rate'] for v in ('state_only','global_motion_arm')}
    gates=dict(pooled_gain5pp_both=all(v>=.05 for v in diff.values()),
               each_seed_no_worse_both=all(s['cm']['physical105_rate']>=s[v]['physical105_rate'] for s in seeds.values() for v in ('state_only','global_motion_arm')))
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',
        optimization_seed=752,updates_each=12,train_trajectories=9216,evaluation_trajectories=1536,
        shared_training_data=True,matched_update_budget=True,physical_pretraining_trajectories=4608,
        pooled=pooled,seeds=seeds,motions=motions,cm_minus_control=diff,gates=gates,
        independent_decision_maximum_error=maximum,final_checkpoint_sha256=sha(heads),
        boundary='single-seed reference-conditioned contextual macro policy Probe; no novel method, formal Validation, generalization or hardware')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'evaluation_audit.json').write_text(json.dumps(dict(run_status='COMPLETED',all1536_rows=True,feature_and_logit_replay=True,
        actual_native_actions_and_physical105_audited=True,maximum_error=maximum,wall_seconds=time.monotonic()-begin),indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
