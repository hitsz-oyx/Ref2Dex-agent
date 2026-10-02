#!/usr/bin/env python3
"""Independent force-derived factual labels and NumPy diagnostic arithmetic."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def run(args):
    started=time.monotonic()
    if args.output.exists():
        raise ValueError('unique audit artifact required')
    import torch
    torch.set_num_threads(2)
    manifest=json.loads((args.run/'run_manifest.json').read_text())
    result=json.loads((args.run/'results.json').read_text())
    if manifest['run_status']!='COMPLETED' or not manifest['parameters_frozen']:
        raise ValueError('complete frozen GPU replay required')
    if sha(args.run/'results.json')!=manifest['result_sha256'] or sha(args.run/'forecasts.pt')!=manifest['forecast_sha256']:
        raise ValueError('diagnostic artifact drift')
    if any(sha(Path(k))!=v for k,v in manifest['input_sha256'].items()):
        raise ValueError('pinned input drift')
    saved=torch.load(args.run/'forecasts.pt',map_location='cpu',weights_only=False)
    pred=saved['prediction'].numpy().astype(np.float64)
    if saved['future_used_as_model_input'] or pred.shape!=(4,3,306,8,52):
        raise ValueError('cal-only full forecast shape')
    source=json.loads((args.source/'run_manifest.json').read_text())
    records={p['seed']:torch.load(Path(p['directory'])/'records.pt',map_location='cpu',weights_only=False) for p in source['phases']}
    lookups={s:{(int(env),int(tick)):i for i,(env,tick) in enumerate(zip(b['env_id'],b['trigger']))} for s,b in records.items()}
    truth=[];arms=[];p=[];identities=set()
    for identity in saved['identities']:
        seed=identity['seed'];key=(identity['env_id'],identity['tick']);b=records[seed];i=lookups[seed][key]
        if (seed,*key) in identities or not 50<=int(b['split_group_bucket'][i])<70:
            raise ValueError('cal-only unique row required')
        identities.add((seed,*key))
        h=b['future_hand_force'][i].numpy().astype(np.float64)
        obj=b['future_object_force'][i].numpy().astype(np.float64)
        mg=float(b['mass_kg'][i])*b['gravity_magnitude']
        contact=np.stack((np.max(np.linalg.norm(h,axis=-1),axis=-1)/mg>.1,np.linalg.norm(obj,axis=-1)/mg>.1),-1)
        if not np.array_equal(contact,b['future_contact'][i].numpy()):
            raise ValueError('independent raw force labels')
        hand=bool(np.all(contact[-3:,0]));object_present=bool(np.all(contact[-3:,1]))
        joint=hand and object_present
        clear=b['future_clearance'][i].numpy()>=.002
        initial=bool(b['initial_clearance'][i]>=.002)
        ever=initial;loss=False
        for c in clear:
            loss=loss or (ever and not bool(c));ever=ever or bool(c)
        height=np.maximum(np.min(b['future_state'][i,-3:,38].numpy())-np.float32(b['rest_z'][i]),np.float32(0))
        support=joint and bool(np.all(clear[-3:]))
        truth.append([hand,object_present,joint,loss,support and height>=.03,height*support*1000])
        arms.append(int(b['assignment'][i]));p.append(float(b['propensity'][i]))
    truth=np.asarray(truth,dtype=np.float64);arms=np.asarray(arms);p=np.asarray(p)
    label_error=float(np.max(np.abs(truth-saved['factual_labels'].numpy())))
    if label_error>1e-4 or not np.array_equal(arms,saved['assignment'].numpy()) or not np.array_equal(p,saved['propensity'].numpy()):
        raise ValueError('factual diagnostic label/assignment mismatch')
    # Use the audited float32 label representation for exact downstream arithmetic.
    y=saved['factual_labels'].numpy().astype(np.float64)
    probability=1/(1+np.exp(-pred[..., [47,48,49,50]]))
    max_error=0.
    def compare(actual,expected,limit=3e-5):
        nonlocal max_error
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        max_error=max(max_error,error)
        if error>limit:raise ValueError('independent diagnostic arithmetic error '+str(error))
    for mode_index,mode in enumerate(saved['modes']):
        factual_prob=probability[mode_index][:,np.arange(306),arms].mean(0)
        factual_height=pred[mode_index][:,np.arange(306),arms,51].mean(0)*10
        metrics=result['factual_metrics'][mode]
        masks=dict(all=np.ones(306,dtype=bool),generated_pool=(arms>=2)&(arms<=4),random_pool=arms>=5)
        masks.update({'arm'+str(a):arms==a for a in range(8)})
        for name,mask in masks.items():
            m=metrics['by_arm'][name[3:]] if name.startswith('arm') else metrics[name]
            compare(int(mask.sum()),m['rows'],0)
            t=y[mask][:,[0,1,3,4]];fp=factual_prob[mask]
            for field,values in [('brier',((fp-t)**2).mean(0)),('observed_rates',t.mean(0)),('predicted_rates',fp.mean(0))]:
                compare(values,[m[field][k] for k in ('hand_presence','object_presence','geometric_loss','supported_lift')])
            compare(np.mean(np.abs(factual_height[mask]-y[mask,5])),m['supported_height_mae_mm'])
            compare(np.mean(factual_height[mask]-y[mask,5]),m['supported_height_bias_mm'])
        mean=probability[mode_index].mean(0)
        upper=np.min(mean[...,:2],axis=-1);lower=np.maximum(0,mean[...,0]+mean[...,1]-1)
        for arm in range(8):
            v=mean[:,arm,3]-upper[:,arm];r=result['consistency'][mode]['lift_subset_violation_by_arm'][str(arm)]
            compare(int(np.sum(v>1e-6)),r['rows'],0)
            compare(int(np.sum(v>.05)),r['over_5pp_rows'],0)
            compare(np.max(v),r['maximum_excess'])
        for name,mask in [('all',np.ones(306,dtype=bool)),('changed_cup_pd',saved['changed_cup_pd'].numpy())]:
            c=result['consistency'][mode]['cm_vs_cup'][name]
            compare(int(np.sum(upper[mask,2]<lower[mask,1])),c['frechet_definite_decline_rows'],0)
            compare(int(np.sum(upper[mask,2]+.05<lower[mask,1])),c['frechet_decline_over_5pp_rows'],0)
            compare(((pred[mode_index,:,:,2,51]-pred[mode_index,:,:,1,51])*10).mean(0)[mask].mean(),c['predicted_joint_height_gain_mm'])
    if result['parameter_fingerprint_before']!=result['parameter_fingerprint_after']:
        raise ValueError('parameter drift')
    out=dict(run_status='COMPLETED',audit_passed=True,cal_rows=306,raw_force_labels_verified=True,
        label_max_error=label_error,statistics_max_error=max_error,full_forecast_checks='frozen GPU replay and cached planning errors in result',
        result_sha256=sha(args.run/'results.json'),forecast_sha256=sha(args.run/'forecasts.pt'),
        auditor_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
        device='cpu',device_reason='independent label/arithmetic audit; no model computation',
        scope='HD03 diagnostic consistency only; no HF20 gate changes or utility claim')
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-optimized-contact-opportunity-source-r1')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
