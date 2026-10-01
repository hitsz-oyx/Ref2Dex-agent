#!/usr/bin/env python3
"""Terminal matched recovery-policy Probe; no checkpoint or threshold selection."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from src.task.CmResidual.trajectory_selector import effective_commands


def run(args):
    torch.set_num_threads(2)
    m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only']:raise ValueError('terminal scientific phases required')
    if len(m['phases'])!=12:raise ValueError('all fixed phases required')
    learnable_guide=m.get('learnable_guide',False);seeds=m.get('evaluation_seeds',list(range(391,395)));first_seed=seeds[0]
    phases={p['name']:p for p in m['phases']};records={};audit={}
    for name,p in phases.items():
        directory=Path(p['directory']);r=p['result']
        if not r['frozen_cm_experts'] or sha(directory/'policy.pt')!=r['policy_sha256'] or sha(directory/'decisions.pt')!=r['decisions_sha256']:
            raise ValueError('model/record drift')
        policy=torch.load(directory/'policy.pt',map_location='cpu',weights_only=False)
        buffers=torch.load(directory/'decisions.pt',map_location='cpu',weights_only=False)
        if policy['trajectory_sha256']!='027202015c32ba783aa1bbef5a0a3c501643bfa904e0b460cdf1877193971355':raise ValueError('physical input drift')
        if not all(torch.isfinite(v).all() for v in policy['policy'].values()):raise ValueError('nonfinite policy')
        for b in buffers:
            if not torch.isfinite(b['returns']).all() or not torch.isfinite(b['old_logprob']).all() or (b['old_logprob']>1e-6).any():raise ValueError('invalid return/log probability')
        learned_physical=0;nonbase=0;decisions=0;raw_changes=0;cached_frames=0;nonbase_cached_frames=0
        for b in buffers:
            row=torch.arange(len(b['selected']));selected=b['selected'];prior=b.get('reference_option',b['inputs']['log_prior'].argmax(-1))
            a=effective_commands(b['candidate_actions']) if 'candidate_actions' in b else b['inputs']['actions']
            raw_changes+=int((selected!=prior).sum())
            learned_physical+=int((a[row,selected]!=a[row,prior]).any(-1).sum())
            different=(a[row,selected]!=a[:,4]).any(-1)
            nonbase+=int(different.sum());decisions+=len(row)
            counts=b.get('executed_steps',torch.full_like(selected,2))
            cached_frames+=int(counts.sum());nonbase_cached_frames+=int(counts[different].sum())
        audit[name]=dict(episodes=sum(q['episodes'] for q in r['rollouts']),effective_first_episode_steps=r['native_env_steps'],
                        actual_batched_sim_steps=sum(q['episodes']*max(q['episode_steps']) for q in r['rollouts']),optimizer_updates=r['optimizer_updates'],
                        initial_fingerprint=policy['initial_fingerprint'],policy_parameters_changed=r['policy_parameters_changed'],
                        guide_weight=policy['policy'].get('guide_weight',torch.tensor(float('nan'))).item() if learnable_guide else None,
                        selected_raw_option_differs_from_prior=raw_changes,decisions=decisions,actual_nonbase_decisions=nonbase,selected_physical_differs_from_prior=learned_physical,
                        cached_candidate_frames=cached_frames,nonbase_candidate_frames=nonbase_cached_frames,
                        cached_frame_fraction=cached_frames/r['native_env_steps'],nonbase_candidate_frame_fraction=nonbase_cached_frames/r['native_env_steps'],
                        inference_ms_total=sum(r['batched_inference_ms']),inference_calls=len(r['batched_inference_ms']),
                        inference_ms_median=float(np.median(r['batched_inference_ms'])))
        records[name]=r
    trace_audit=None
    if learnable_guide:
        from audit_recovery_traces import audit_phase
        trace_audit={}
        for name,p in phases.items():trace_audit[name]=audit_phase(Path(p['directory']),p['result'])[0]
    on,off=records['train_on'],records['train_off']
    if len(on['rollouts'])!=4 or len(off['rollouts'])!=4 or on['native_env_steps']!=off['native_env_steps']:raise ValueError('training budget mismatch')
    if audit['train_on']['initial_fingerprint']!=audit['train_off']['initial_fingerprint']:raise ValueError('initial weights mismatch')
    def metrics(names):
        result={};episodes=0
        for name in names:
            for r in records[name]['rollouts']:
                episodes+=r['episodes']
                for key,value in r['metrics'].items():result[key]=result.get(key,0)+value
        return dict(episodes=episodes,counts=result,rates={k:v/episodes for k,v in result.items()},
                    conditional_release_after_acquisition=result['post_lift_release']/result['acquired_lift'] if result['acquired_lift'] else None)
    evaluated={mode:metrics([f'eval_{mode}_s{s}' for s in seeds]) for mode in ['on','off']}
    prior={mode:metrics([f'prior_{mode}_s{first_seed}']) for mode in ['on','off']}
    per_seed=[]
    for seed in seeds:
        left=metrics([f'eval_on_s{seed}']);right=metrics([f'eval_off_s{seed}'])
        per_seed.append(dict(seed=seed,on=left,off=right,stable_difference=left['rates']['stable_success']-right['rates']['stable_success']))
    differences=np.array([v['stable_difference'] for v in per_seed]);mean=float(differences.mean());se=float(differences.std(ddof=1)/2)
    interval=[mean-3.182446305284263*se,mean+3.182446305284263*se]
    conditional_on=evaluated['on']['conditional_release_after_acquisition'];conditional_off=evaluated['off']['conditional_release_after_acquisition']
    conditional_diff=conditional_on-conditional_off if conditional_on is not None and conditional_off is not None else None
    trained_first=metrics([f'eval_on_s{first_seed}']);learning_success_delta=trained_first['rates']['stable_success']-prior['on']['rates']['stable_success']
    physical_changes=sum(audit[f'eval_on_s{s}']['selected_physical_differs_from_prior'] for s in seeds)
    physical_fraction=physical_changes/sum(audit[f'eval_on_s{s}']['decisions'] for s in seeds)
    gates=dict(actual_policy_training=all(r['optimizer_updates']>0 and r['policy_parameters_changed'] for r in [on,off]),
               learned_physical_decisions=physical_fraction>=.05 if learnable_guide else physical_changes>0,
               stable_success_on_above_off=mean>=.02 if learnable_guide else mean>0,
               release_after_acquisition_noninferior=conditional_diff is not None and conditional_diff<=.02,
               stable_learning_beyond_frozen_prior=learning_success_delta>0)
    gates['passed']=all(gates.values());label='PROMISING' if gates['passed'] else 'UNPROMISING'
    output=dict(experiment_id=m['experiment_id'],run_status='COMPLETED',label=label,gate=gates,training_seed=m.get('training_seed',381),evaluated=evaluated,per_seed=per_seed,
                stable_difference=mean,descriptive_eval_seed_t95=interval,conditional_release_difference=conditional_diff,
                all_episode_release_difference=evaluated['on']['rates']['post_lift_release']-evaluated['off']['rates']['post_lift_release'],
                frozen_prior=prior,prior_comparison_seed=first_seed,stable_learning_difference_first_seed=learning_success_delta,
                learned_physical_evaluation_decisions=physical_changes,learned_physical_decision_fraction=physical_fraction,
                phase_audits=audit,trace_replay_audits=trace_audit,input_hashes_unchanged=all(sha(Path(k))==v for k,v in m['input_sha256'].items()),
                elapsed_seconds_including_smoke=m['cumulative_seconds'],bytes_including_smoke=m['output_bytes'],
                scope='one training seed; interval only over four evaluation seeds conditional on these checkpoints; matched configuration/initial weights/episode budgets; acquisition-conditioned release is descriptive, no common-prestate causal risk estimate; '+('cold physical inputs recorded and compared in separate trace audit; raw independent12 commands establish actual changes' if learnable_guide else 'exact cold physical-state identity not recorded; clipped normalized command inequality gives conservative physical-change evidence'),
                scientific_claim='C3 remains OPEN; engineering or single-seed positive Probe is not formal stable-grasp/Cm training utility support')
    if not output['input_hashes_unchanged']:raise ValueError('input/code drift')
    if args.output.exists():raise ValueError('analysis exists')
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['phase_audits','per_seed']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
