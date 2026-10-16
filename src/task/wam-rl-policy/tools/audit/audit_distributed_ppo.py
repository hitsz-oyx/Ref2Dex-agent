"""Gate completed native PPO gangs on saved weights, chunks and execution evidence."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--rollout-steps',type=int,required=True)
    parser.add_argument('--updates',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh audit output required')
    process=json.loads((args.run/'process.json').read_text());ranks=process['ranks']
    assert process['status']=='COMPLETED' and process['exit_codes']==[0]*len(ranks)
    packets=[torch.load(args.run/('rank%02d'%r)/'latest.pt',map_location='cpu',weights_only=False) for r in range(len(ranks))]
    chunks=[args.rollout_steps]*(199//args.rollout_steps)
    if 199%args.rollout_steps:chunks.append(199%args.rollout_steps)
    total_actor_steps=[];max_replay=0.;max_kl=0.
    for rank,packet in enumerate(packets):
        folder=args.run/('rank%02d'%rank)
        assert packet['update']==args.updates and packet['config']['rollout_steps']==args.rollout_steps
        frozen=json.loads((folder/'freeze-audit.json').read_text())
        assert frozen['status']=='PASS' and frozen['exact_state_dict_unchanged'] and frozen['gradients_absent']
        if packet['config'].get('wam_feature_checkpoint'):
            frozen_features=json.loads((folder/'feature-freeze-audit.json').read_text())
            assert frozen_features['status']=='PASS' and frozen_features['exact_state_dict_unchanged'] and frozen_features['gradients_absent']
        condition=packet['config'].get('observation_kind','native')
        if condition.startswith('wam-'):
            inputs=torch.load(folder/'first-wam-input.pt',map_location='cpu',weights_only=False)
            source=inputs['source_z'];supplied=inputs['supplied_z'];obs=inputs['observation']
            assert source.shape==supplied.shape==(len(obs),2304) and torch.isfinite(source).all()
            if condition=='wam-none':assert not supplied.any()
            elif condition=='wam-shuffled':assert torch.equal(supplied,source[inputs['donor_indices']])
            else:assert torch.equal(supplied,source)
            expected=torch.nn.functional.layer_norm(supplied.reshape(-1,3,768),(768,)).flatten(1)
            torch.testing.assert_close(obs[:,300:2604],expected,rtol=2e-5,atol=2e-5)
            assert torch.equal(obs[:,:300],inputs['current'].clamp(-5,5)) and torch.equal(obs[:,-24:],inputs['base'])
        for key in ('actor','critic'):
            assert all(torch.isfinite(value).all() and torch.equal(value,packets[0][key][name]) for name,value in packet[key].items())
        curve=json.loads((folder/'curve.json').read_text())
        assert len(curve)==args.updates and curve[-1]['update']==args.updates
        total=0
        for row in curve:
            segments=row.get('ppo_segments',[row['ppo']])
            assert [segment['rollout_steps'] for segment in segments]==chunks
            assert row['ppo_update_batches']==len(chunks)
            assert [segment['bootstrap_kind'] for segment in segments]==['actual-next-state-before-update']*(len(chunks)-1)+['terminal-zero']
            assert segments[-1]['bootstrap_mean']==0
            for segment in segments:
                assert 0<=segment['actor_steps']<=packet['config']['ppo_epochs']*packet['config'].get('ppo_minibatches',1)
                assert segment['behavior_logprob_error']<=1e-4 and segment['joint_kl']<=.020001
                max_replay=max(max_replay,segment['behavior_logprob_error']);max_kl=max(max_kl,segment['joint_kl'])
            steps=sum(segment['actor_steps'] for segment in segments)
            assert steps==row['actor_optimizer_steps_this_episode'];total+=steps
        total_actor_steps.append(total)
    assert len(set(total_actor_steps))==1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',run=str(args.run.resolve()),ranks=len(ranks),episodes=args.updates,
        rollout_steps=args.rollout_steps,chunk_lengths=chunks,optimizer_batches=args.updates*len(chunks),
        actor_optimizer_steps=total_actor_steps[0],max_behavior_replay_error=max_replay,max_joint_kl=max_kl,
        actor_critic_exact_across_ranks=True,frozen_reference_unchanged=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
