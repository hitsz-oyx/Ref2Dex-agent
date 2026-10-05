#!/usr/bin/env python3
"""Registered fixed-range isolated-finger control Decision; no neural fits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from probe_amplitude_authority import audit_packet, targets, FAMILIES, ROOT
from probe_duration_response import current_design, residual_fit, randomization, sha
from audit_finger_amplitudes import finger_audit, markdown
from intervention import per_finger_residuals, PER_FINGER_ARM_NAMES


def own_interaction(beta, tails, halves, arm):
    body=(arm-1)//2
    if body>=4: body=4
    passed=[]
    for axis,threshold,name in ((5+body,.15,'log1p_force_norm'),(13+body,.003,'surface_distance_m')):
        effect=beta[arm-1,axis]; orientation=np.sign(effect)
        repeated=[float(b[arm-1,axis]*orientation) for b in halves]
        if abs(effect)>=threshold and tails[axis]<=.10 and min(repeated)>0:
            passed.append(dict(body=body,axis=axis,quantity=name,effect=float(effect),half_signed_effects=repeated))
    return passed


def candidates(beta,tails,halves):
    linked=[]; useful=[]
    for arm in range(1,13):
        own=own_interaction(beta,tails,halves,arm)
        if not own: continue
        task=[]
        for axis,threshold,orientation,name in ((1,.08,1,'late_contact'),(2,.10,-1,'late_height_failure')):
            effect=beta[arm-1,axis]*orientation; sign=np.sign(effect)
            repeated=[float(b[arm-1,axis]*orientation*sign) for b in halves]
            if abs(effect)>=threshold and tails[axis]<=.10 and min(repeated)>0:
                task.append(dict(quantity=name,orientation=int(sign),signed_benefit=float(effect),half_signed_effects=repeated))
        if task and len({t['orientation'] for t in task})==1:
            linked.append(dict(arm=arm,name=PER_FINGER_ARM_NAMES[arm],own_interaction=own,task=task,
                direction='beneficial' if task[0]['orientation']>0 else 'harmful'))
        if (beta[arm-1,1]>=.10 and beta[arm-1,2]<=-.10 and tails[1]<=.10 and tails[2]<=.10
                and all(b[arm-1,1]>0 and b[arm-1,2]<0 for b in halves)):
            useful.append(dict(arm=arm,name=PER_FINGER_ARM_NAMES[arm],own_interaction=own,
                late_contact_gain=float(beta[arm-1,1]),height_failure_reduction=float(-beta[arm-1,2])))
    return linked,useful


def plot(out,p,design,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tr=p['trajectory']; force=tr[:,:,48:63].reshape(-1,32,5,3).norm(dim=-1)
    values=torch.cat((tr[:,:,66:71]*1000,torch.log1p(force),tr[:,:,71:72],
        (tr[:,:,2:3]-p['rest_height'][:,None,None]<.02).float()),-1).numpy().astype(float)
    beta=residual_fit(design,labels,values.reshape(len(values),-1),14)[0].reshape(14,32,12)
    fig,axs=plt.subplots(6,4,figsize=(15,16),sharex=True)
    for finger in range(6):
        body=min(finger,4)
        for col,axis in enumerate((body,5+body,10,11)):
            ax=axs[finger,col]
            for row in (finger*2,finger*2+1,12,13):
                ax.plot(np.arange(1,33),beta[row,:,axis],label=p['arm_names'][row+1])
            ax.axvline(8,color='black',linestyle='--',linewidth=.7); ax.axhline(0,color='black',linewidth=.5)
            if finger==0: ax.set_title(('Own body surface distance (mm)','Own log1p force norm (N)','Global contact proxy','Physical height loss (<2cm)')[col])
            if col==0: ax.set_ylabel(('index','middle','pinky','ring','thumb yaw','thumb pitch')[finger])
            ax.legend(fontsize=6)
    for ax in axs[-1]: ax.set_xlabel('post-step')
    fig.suptitle('Fixed20% driver range, K8: adjusted isolated arms and same-cohort composite controls\nDescriptive force/proximity proxies; no certified contact load or slip')
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(out/'per_finger_response.png',dpi=150); plt.close(fig)
    return beta.tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=229)
    args=parser.parse_args(); args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); started=time.monotonic()
    def save(name,value): (args.run_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    manifest=dict(run_status='STARTED',command=__import__('sys').argv,dataset_sha256=sha(args.dataset),seed=args.seed,
        collection_manifest_sha256=sha(args.dataset.parent/'manifest.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={name:sha(Path(__file__).resolve().with_name(name)) for name in
            ('probe_per_finger_control.py','audit_finger_amplitudes.py','probe_amplitude_authority.py','probe_duration_response.py')},
        device='CPU file/OLS/randomization analysis; no neural models',created_at=datetime.now(timezone.utc).isoformat())
    save('manifest.json',manifest)
    try:
        p=torch.load(args.dataset,weights_only=False); collection=json.loads((args.dataset.parent/'manifest.json').read_text())
        assert collection['run_status']=='COMPLETED' and p['amplitude_levels']==[1.]
        assert tuple(p['arm_names'])==PER_FINGER_ARM_NAMES and p['finger_range_fraction']==.2
        assert tuple(p['native_q_units'])==('m',)*3+('rad',)*15
        audit=audit_packet(p,args.dataset,per_finger_residuals(.2)); save('engineering_audit.json',audit)
        envs=int(collection['command'][collection['command'].index('--num_envs')+1])
        waves=p['episode_id'].numpy()//envs; half=waves>=collection['waves']//2; labels=p['arm'].numpy()
        assert p['native_q'].shape==(len(labels),32,18) and p['fingertip_positions'].shape==(len(labels),32,5,3)
        assert tuple(p['fingertip_names'])==('index_tip','middle_tip','pinky_tip','ring_tip','thumb_tip')
        target,details,_,_=targets(p); design,groups=current_design(p,waves)
        half_counts=[[int(((labels==arm)&(half==side)).sum()) for arm in range(15)] for side in (False,True)]
        support=all(row['count']>=30 for row in audit['cells']) and min(min(c) for c in half_counts)>=12
        dose=all(row['dose_ratio'] is not None and row['dose_ratio']>=.90 for row in audit['cells'] if row['arm'])
        beta,tails,families,statistics=randomization(design,labels,target,groups,args.seed,FAMILIES,14)
        fits=[residual_fit(design[half==side],labels[half==side],target[half==side],14) for side in (False,True)]
        halves=[f[0] for f in fits]; linked,useful=candidates(beta,tails,halves)
        valid=support and dose and statistics['treatment_rank']==14 and all(f[3]==14 for f in fits)
        result=dict(status='PROMISING' if valid and linked else ('UNPROMISING' if valid else 'UNCLEAR'),
            useful_status='PROMISING' if valid and useful else ('UNPROMISING' if valid else 'UNCLEAR'),
            support_pass=bool(support),dose_pass=bool(dose),candidates=linked,useful_candidates=useful,
            families=families,statistics=statistics,half_counts=half_counts,half_treatment_ranks=[f[3] for f in fits],
            adjusted_cells=[dict(arm=arm,name=p['arm_names'][arm],effects=beta[arm-1].tolist(),half_effects=[b[arm-1].tolist() for b in halves]) for arm in range(1,15)],
            event_counts={k:int(v.sum()) for k,v in details.items() if v.dtype==torch.bool},neural_models_executed=False,
            elapsed_seconds=time.monotonic()-started,limits='Single-cohort fixed-range marginal contrasts; cannot identify additive cancellation, certified load-bearing contact or trained-policy Cm utility.')
        save('result.json',result)
        amplitudes=finger_audit(p,design,labels); save('finger_amplitudes.json',amplitudes)
        (args.run_dir/'finger_amplitudes.md').write_text(markdown(amplitudes))
        save('response_curves.json',dict(adjusted_arm_minus_zero=plot(args.run_dir,p,design,labels)))
        torch.save(dict(target=target,details=details,design=design,groups=groups,labels=labels,wave=waves,half=half,
            coefficient=beta,half_coefficient=halves,family_tails=tails),args.run_dir/'diagnostic.pt')
        print(json.dumps({k:result[k] for k in ('status','useful_status','support_pass','dose_pass','candidates','families','event_counts')},indent=2))
        manifest['run_status']='COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}'); raise
    finally:
        manifest['completed_at']=datetime.now(timezone.utc).isoformat(); save('manifest.json',manifest)


if __name__=='__main__': main()
