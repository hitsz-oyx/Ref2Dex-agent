#!/usr/bin/env python3
"""Capture missing reward/critic observations during unchanged cold fork replay."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT),str(Path(__file__).parent),str(ROOT/'src/task/cm-interaction-oracle/src')]
# The existing entry imports Isaac Gym before torch.
import collect_oracle_y_candidates as original
import torch
from critic_ranking import FrozenCritic
from src.task.CmResidual.paired_evaluation import fingerprint
EXPECTED = None


class CriticCandidatePlayer(original.CandidatePlayer):
    @torch.no_grad()
    def run(self):
        args = original.ARGS
        schedule = json.loads(args.anchor_schedule.read_text())
        triggers = torch.tensor(schedule['triggers'])
        groups = torch.tensor(schedule['groups'])
        self.critic_triggers = torch.where(groups == args.group_id,triggers,-1)
        n = len(triggers); self.critic_tick = 0
        self.critic_rewards = torch.full((n,8),float('nan'))
        self.critic_dones = torch.zeros(n,8,dtype=torch.bool)
        self.critic_seen = torch.zeros(n,8,dtype=torch.bool)
        self.critic_obs8 = torch.full((n,1442),float('nan'))
        self.critic_v8 = torch.full((n,),float('nan'))
        if self.normalize_value:
            raise ValueError('source_e260 contract requires normalize_value=false')
        if float(self.config['gamma']) != .99 or self.config['reward_shaper']['scale_value'] != 1:
            raise ValueError('PPO gamma/reward units drift')
        super().run()
        folder = args.run_dir
        panel = torch.load(folder/'panel.pt',map_location='cpu',weights_only=False)
        expected = torch.load(EXPECTED,map_location='cpu',weights_only=False)
        ids = (self.critic_triggers>=0).nonzero().flatten()
        if not self.critic_seen[ids].all() or self.critic_dones[ids].any():
            raise ValueError('missing reward/endpoint or terminal within block')
        errors = {}
        for key in ('before','actor_obs','history','actions','base_actions','pd_targets','trajectory','native_q','fingertip_positions','hand_base_pose'):
            errors[key] = float((panel[key][ids]-expected[key][ids]).abs().max())
        for key in ('height','pair','valid_steps'):
            if not torch.equal(panel[key][ids,:32],expected[key][ids,:32]):
                raise ValueError('old candidate outcome identity drift: '+key)
        if max(errors.values()) > 1e-4:
            raise ValueError('original candidate physical/action drift')
        raw_checkpoint = torch.load(self.critic_checkpoint,map_location='cpu',weights_only=False)
        independent = FrozenCritic(raw_checkpoint).to(self.device)
        values = independent(self.critic_obs8[ids].to(self.device)).cpu()
        replay_error = float((values-self.critic_v8[ids]).abs().max())
        if replay_error > 1e-5:
            raise ValueError('live player vs independently loaded critic mismatch')
        payload = dict(schema='ref2dex.critic_candidate.v1',candidate=args.candidate,group=args.group_id,
                       rows=ids,triggers=self.critic_triggers,observations8=self.critic_obs8[ids],
                       raw_rewards8=self.critic_rewards[ids],dones8=self.critic_dones[ids],
                       live_values8=self.critic_v8[ids],gamma=.99,reward_scale=1.,normalize_value=False,
                       original_candidate_path=str(EXPECTED.resolve()),original_candidate_sha256=original.sha(EXPECTED),
                       replay_physical_errors=errors,live_vs_independent_value_max_error=replay_error,
                       critic_model_fingerprint=fingerprint(self.model.state_dict()),
                       critic_rms_fingerprint=fingerprint(self.running_mean_std.state_dict()))
        torch.save(payload,folder/'critic.pt')
        (folder/'critic.json').write_text(json.dumps(dict(status='PASS',rows=ids.tolist(),
            expected_panel_sha256=original.sha(EXPECTED),critic_sha256=original.sha(folder/'critic.pt'),
            replay_errors=errors,value_replay_error=replay_error),indent=2)+'\n')

    @torch.no_grad()
    def env_step(self, env, actions):
        result = super().env_step(env,actions)
        age = self.critic_tick-self.critic_triggers
        ids = ((self.critic_triggers>=0)&(age>=0)&(age<8)).nonzero().flatten()
        if len(ids):
            raw,reward,done,_ = result
            offsets = age[ids]
            self.critic_rewards[ids,offsets] = reward.flatten()[ids.to(reward.device)].cpu()
            self.critic_dones[ids,offsets] = done.flatten()[ids.to(done.device)].bool().cpu()
            self.critic_seen[ids,offsets] = True
            endpoint = ids[offsets==7]
            if len(endpoint):
                obs = raw['obs'] if isinstance(raw,dict) else raw
                copied = obs[endpoint.to(obs.device)].to(self.device).clone()
                self.critic_obs8[endpoint] = copied.cpu()
                processed = self._preproc_obs(copied)
                value = self.model.a2c_network.eval_critic(processed).flatten()
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite live critic')
                self.critic_v8[endpoint] = value.cpu()
        self.critic_tick += 1
        return result


def main():
    global EXPECTED
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--expected-panel',type=Path,required=True)
    args, remaining = ap.parse_known_args()
    EXPECTED = args.expected_panel
    checkpoint = Path(remaining[remaining.index('--checkpoint')+1])
    CriticCandidatePlayer.critic_checkpoint = checkpoint
    original.CandidatePlayer = CriticCandidatePlayer
    sys.argv = [sys.argv[0],*remaining]
    original.main()


if __name__ == '__main__':
    main()
