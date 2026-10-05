#!/usr/bin/env python3
"""No-fit raw replay of executed rolling oracle: choices, PD, actual path and Z."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from rolling_control import OFFSETS,TOLERANCE,control_gate


def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def raw_y(p,rows):
    height=p['height'][rows]-p['rest_height'][rows,None];contact=p['pair'][rows].bool()
    risk=(p['before'][rows,2]-p['rest_height'][rows]>=.03)&(p['before'][rows,71]>.5)
    lose=torch.zeros(len(rows),dtype=torch.long);fails=torch.zeros_like(contact)
    for t in range(32):
        lose=torch.where(contact[:,t],0,lose+1);fails[:,t]=((height[:,t]<.02)|(lose>=6))&risk
    label=[]
    for end in (16,32):
        label.extend([contact[:,8:end].float().mean(1),((height[:,8:end]>=.03)&contact[:,8:end]).float().mean(1),
                      fails[:,8:end].any(1).float()])
    label.extend([((height[:,8:]<.02).any(1)&risk).float(),(height[:,8:]>=.03).float().mean(1)])
    return torch.stack(label,1)


def raw_z(height,contact,rest):
    z=[];qualification=[];drop=[]
    for h,c,r in zip(height.tolist(),contact.tolist(),rest.tolist()):
        run=0;lost=0;first=-1;dropped=False
        for t,(v,pair) in enumerate(zip(h,c),1):
            previous=first>=0;run=run+1 if v-r>=.03 and pair else 0;lost=0 if pair else lost+1
            if first<0 and run>=45 and t<=60:first=t
            if previous and (v-r<.02 or lost>=6):dropped=True
        z.append(first>=0 and not dropped);qualification.append(first);drop.append(dropped)
    return z,qualification,drop


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve();out=root/'raw_audit.json'
    if out.exists():raise ValueError('audit result must be new')
    result=json.loads((root/'result.json').read_text()); groups=[];files={};rounds=0;pdmax=0.
    skipped=0
    for group in result['groups']:
        rows=torch.tensor(group['rows']);rest=torch.tensor(group['rest']);origin=group['origin']
        baseline=load(Path(group['baseline_dir'])/'panel.pt')
        z,q,d=raw_z(baseline['height'][rows],baseline['pair'][rows],rest)
        assert z==group['baseline']
        reference=Path(group['baseline_dir'])
        actual_prefix=load(reference/'trace.pt')
        for decision in group['decisions']:
            rounds+=1;offset=decision['offset'];clock=origin+offset
            prefix=f"s{group['seed']}-g{group['group']}-t{offset:02d}"
            plan_path=root/(prefix+'-plan.json');plan=json.loads(plan_path.read_text());files[str(plan_path)]=sha(plan_path)
            assert sha(reference/'trace.pt')==plan['source_prefix_sha256']
            choices=torch.tensor(plan['choices'])
            if plan.get('baseline_upper_bound_certificate'):
                skipped+=1;p=load(root/(prefix+'-k0')/'panel.pt');y=raw_y(p,rows)
                score=y[:,7]+.25*y[:,3]-y[:,6]
                assert torch.equal(score,torch.full_like(score,1.25)) and not choices.any()
                assert plan['y']==y[:,None].tolist() and plan['utility']==score[:,None].tolist()
            else:
                panels=[load(root/(prefix+f'-k{k}')/'panel.pt') for k in range(7)]
                yy=torch.stack([raw_y(p,rows) for p in panels],1)
                score=yy[:,:,7]+.25*yy[:,:,3]-yy[:,:,6]
                assert plan['y']==yy.tolist() and plan['utility']==score.tolist()
                assert torch.equal(choices[rows],score.argmax(1))
                for p in panels:
                    assert (p['full_world_prefix_errors']<=TOLERANCE).all()
                    assert p['valid_steps'][rows].all()
                    for key in ('before','history','actor_obs','hand_root'):
                        assert torch.allclose(p[key][rows],panels[0][key][rows],atol=1e-4,rtol=0)
            for path,h in plan['candidate_panel_sha256'].items():
                assert sha(Path(path))==h;files[path]=h
            folder=Path(decision['actual_dir']);p=load(folder/'panel.pt');trace=load(folder/'trace.pt')
            assert (p['full_world_prefix_errors']<=TOLERANCE).all()
            for key in ('physical','dof','root','action','done'):
                assert torch.equal(trace[key][:clock],actual_prefix[key][:clock]),'actual prefix changed: '+key
            expected=(p['base_actions'][rows]+p['delta'][choices[rows],None]).clamp(-1,1)
            assert torch.equal(expected,p['actions'][rows])
            assert torch.equal(trace['action'][clock:clock+8,rows].transpose(0,1),expected)
            assert torch.equal(trace['after_physical'][clock:clock+8,rows,2].transpose(0,1),p['height'][rows,:8])
            initial=load(folder/'initial_state.pt');a=expected.clone();a[:,:,6:]=(1+a[:,:,6:])/2
            pd=initial['tensors']['_pd_action_offset']+initial['tensors']['_pd_action_scale']*a
            pd[:,:,:6]+=trace['dof'][clock:clock+8,rows,:6].transpose(0,1)
            for target,source,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
                pd[:,:,target]=pd[:,:,source]*ratio
            delta=float((pd-p['pd_targets'][rows]).abs().max());pdmax=max(pdmax,delta);assert delta==0
            reference=folder;actual_prefix=trace
        assert [d['offset'] for d in group['decisions']]==list(OFFSETS)
        after=actual_prefix['after_physical'][origin:origin+90,rows].transpose(0,1)
        assert after.shape[1]==90 and not actual_prefix['done'][origin:origin+90,rows].any()
        z,q,d=raw_z(after[:,:,2],after[:,:,71]>.5,rest)
        assert z==group['rolling'] and q==group['rolling_qualification'] and d==group['rolling_dropped']
        groups.append(dict(seed=group['seed'],group=group['group'],rows=group['rows'],baseline=group['baseline'],rolling=z))
    base=np.concatenate([g['baseline'] for g in groups]);roll=np.concatenate([g['rolling'] for g in groups])
    motion=np.concatenate([g['motion'] for g in result['groups']]);gate=control_gate(base,roll,motion)
    for key,value in gate.items():assert result[key]==value
    report=dict(status='PASS',raw_Y_selection_PD_prefix_Z_and_statistics_exact=True,rounds=rounds,
                bounded_baseline_rounds=skipped,pd_max_abs=pdmax,gate=gate,groups=groups,input_hashes=files)
    with out.open('x') as f:json.dump(report,f,indent=2)
    np.savez_compressed(root/'raw_z_replay.npz',baseline=base,rolling=roll,motion=motion)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].bar(['Baseline','One-shot GT-Y','Rolling GT-Y'],[base.sum(),24,roll.sum()],color=['#8296a5','#b1b8c0','#25887b'])
    axes[0].set_ylim(0,32);axes[0].set_ylabel('Stable-grasp Z / 32');axes[0].set_title(gate['status']+' • mechanism Probe')
    for group in result['groups']:
        axes[1].plot(OFFSETS,[sum(c!=0 for c in d['choices']) for d in group['decisions']],marker='o',label=f"s{group['seed']} g{group['group']}")
    axes[1].set_xlabel('Replan offset (steps)');axes[1].set_ylabel('Nonzero choices');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(root/'control_result.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_hashes','groups')},indent=2))
if __name__=='__main__':main()
