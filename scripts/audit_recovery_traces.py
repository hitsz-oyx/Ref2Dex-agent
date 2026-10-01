#!/usr/bin/env python3
"""CPU artifact/label replay; no model inference, fitting, or new evaluation."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from src.task.CmResidual.physical_value_contract import HoldTracker


def audit_phase(directory,result):
    for name,key in [('cold_states.pt','cold_states_sha256'),('episode_traces.pt','episode_traces_sha256'),('decisions.pt','decisions_sha256')]:
        if sha(directory/name)!=result[key]:raise ValueError('artifact drift: '+name)
    cold=torch.load(directory/'cold_states.pt',map_location='cpu',weights_only=False)
    traces=torch.load(directory/'episode_traces.pt',map_location='cpu',weights_only=False)
    decisions=torch.load(directory/'decisions.pt',map_location='cpu',weights_only=False)
    if not len(cold)==len(traces)==len(decisions)==len(result['rollouts']):raise ValueError('rollout count')
    audits=[]
    for c,t,b,r in zip(cold,traces,decisions,result['rollouts']):
        steps=t['episode_steps'];n=len(steps);length=len(t['object_state'])
        if steps.tolist()!=r['episode_steps'] or length!=int(steps.max()):raise ValueError('episode budget drift')
        active=torch.arange(length)[:,None]<steps[None]
        expected=((t['object_state'][:,:,2]-c['rest'])/.03).clamp(0,1)*t['contact'].all(-1)*active
        reward_error=float((expected-t['training_reward']).abs().max())
        if reward_error>1e-6:raise ValueError('physical reward replay mismatch')
        first_done=t['done'].long().argmax(0)+1
        if not torch.equal(first_done,steps) or not t['done'].any(0).all():raise ValueError('first terminal mismatch')
        returns=torch.zeros(length,n);running=torch.zeros(n)
        for tick in reversed(range(length)):
            running=t['training_reward'][tick]*.01+.99*running;returns[tick]=running
        return_error=float((returns[b['tick'],b['env']]-b['returns']).abs().max())
        if return_error>2e-6:raise ValueError('actual MC return mismatch')
        row=torch.arange(len(b['selected']))
        if not torch.equal(b['candidate_actions'][row,b['selected']],b['chosen_command']):raise ValueError('selected bank mismatch')
        counts=b.get('executed_steps',torch.full_like(b['selected'],2))
        if not torch.equal(counts,torch.minimum(torch.full_like(counts,2),steps[b['env']]-b['tick'])):raise ValueError('terminal-prefix mismatch')
        for offset in range(2):
            take=counts>offset
            if not torch.equal(t['action'][b['tick'][take]+offset,b['env'][take]],b['chosen_command'][take]):raise ValueError('cached command mismatch')
        tracker=HoldTracker(n,'cpu');tracker.reset(torch.arange(n),c['rest'])
        acquired=torch.zeros(n,dtype=torch.bool);released=acquired.clone();lost=torch.zeros(n,dtype=torch.long)
        metrics={key:0 for key in r['metrics']}
        for tick in range(length):
            pair=t['contact'][tick].all(-1);z=t['object_state'][tick,:,2]
            tracker.step(z,pair);acquired|=active[tick]&(z-c['rest']>=.03)&pair
            lost=torch.where(acquired&~pair,lost+1,0)
            released|=active[tick]&acquired&((z-c['rest']<.02)|(lost>=6))
            end=steps==tick+1
            values=dict(stable_success=tracker.stable&~tracker.drop_after_success,ever_stable=tracker.stable,
                        drop_after_stable=tracker.drop_after_success,five_step_hold=tracker.max_run>=5/30-1e-6,
                        acquired_lift=acquired,post_lift_release=released)
            for key,value in values.items():metrics[key]+=int((value&end).sum())
        if metrics!=r['metrics']:raise ValueError('stable/drop label replay mismatch')
        audits.append(dict(episodes=n,steps=int(steps.sum()),decisions=len(row),terminal_one_step_prefixes=int((counts==1).sum()),
                           reward_max_error=reward_error,return_max_error=return_error,all_primary_labels_replayed=True,actual_commands_replayed=True))
    return audits,cold


def run(args):
    torch.set_num_threads(2);m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or not m.get('learnable_guide'):raise ValueError('terminal HF12 required')
    phases={};cold={}
    for p in m['phases']:
        if p['run_status']!='COMPLETED':raise ValueError('incomplete native phase')
        phases[p['name']],cold[p['name']]=audit_phase(Path(p['directory']),p['result'])
    comparisons={}
    for name,left in cold.items():
        if '_on' not in name:continue
        other=name.replace('_on','_off')
        if other not in cold:continue
        comparisons[name+' vs '+other]=[{key:dict(exact=torch.equal(a[key],b[key]),
             max_abs_difference=float((a[key].double()-b[key].double()).abs().max()))
             for key in ['root','dof','rigid_body','observation','target','contact_forces','object_forces','motion','start','rest','cpu_rng']}
             for a,b in zip(left,cold[other])]
    audit=dict(run_status='COMPLETED',artifact_and_label_contract_passed=True,phases=phases,cold_comparisons=comparisons,
               scope='CPU file/label/statistical audit only; no model fitting/inference; cold differences reported, no imposed outcome pairing')
    if args.output.exists():raise ValueError('audit output exists')
    args.output.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(contract_passed=True,phases=len(phases),terminal_one_step_prefixes=sum(q['terminal_one_step_prefixes'] for p in phases.values() for q in p))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
