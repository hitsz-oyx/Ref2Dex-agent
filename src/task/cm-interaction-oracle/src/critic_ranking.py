"""Frozen source_e260 V and eight-step PPO reward ranking, no fitted model."""
import numpy as np
import torch
from torch import nn
from oracle_y_utility import paired_bootstrap
from ranking_tolerance import same_state_ranking


class FrozenCritic(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        weights = {k.replace('_orig_mod.', ''):v for k,v in checkpoint['model'].items()}
        layers = []
        for index in (0,2,4,6):
            prefix = 'a2c_network.critic_mlp.'+str(index)
            w,b = weights[prefix+'.weight'], weights[prefix+'.bias']
            layer = nn.Linear(w.shape[1],w.shape[0]); layer.weight.data.copy_(w); layer.bias.data.copy_(b)
            layers.extend((layer,nn.ReLU()))
        w,b = weights['a2c_network.value.weight'],weights['a2c_network.value.bias']
        layer = nn.Linear(w.shape[1],w.shape[0]); layer.weight.data.copy_(w); layer.bias.data.copy_(b)
        layers.append(layer); self.net = nn.Sequential(*layers)
        rms = checkpoint['running_mean_std']
        self.register_buffer('mean',rms['running_mean'].float())
        self.register_buffer('variance',rms['running_var'].float())
        self.eval(); self.requires_grad_(False)

    def forward(self, raw):
        normalized = ((raw-self.mean)/torch.sqrt(self.variance+1e-5)).clamp(-5,5)
        return self.net(normalized).flatten()


def bootstrapped_score(rewards, values, dones, gamma=.99):
    rewards = np.asarray(rewards,dtype=np.float64)
    values = np.asarray(values,dtype=np.float64)
    dones = np.asarray(dones,dtype=bool)
    if rewards.shape[-1] != 8 or dones.shape != rewards.shape or values.shape != rewards.shape[:-1]:
        raise ValueError('eight reward/done steps and endpoint V required')
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError('nonfinite critic inputs')
    # This experiment's inherited cohort has no terminal within the block.
    # Abort rather than bootstrap through reset observations.
    if dones.any():
        raise ValueError('terminal within eight-step candidate block')
    reward_part = (rewards*gamma**np.arange(8)).sum(-1)
    value_part = gamma**8*values
    return reward_part+value_part,reward_part,value_part


def selection_summary(scores,z,y_scores):
    scores,z,y_scores = [np.asarray(v) for v in (scores,z,y_scores)]
    if scores.shape != z.shape or scores.shape != y_scores.shape or scores.shape[1] != 7:
        raise ValueError('matched seven-candidate panels required')
    rows = np.arange(len(z)); chosen = scores.argmax(-1)
    selected = z[rows,chosen].astype(bool); baseline = z[:,0].astype(bool)
    ychosen = y_scores.argmax(-1); yz = z[rows,ychosen].astype(bool)
    return dict(selection=chosen.tolist(),selected_z=selected.tolist(),successes=int(selected.sum()),
                rescued=int((selected & ~baseline).sum()),harmed=int((~selected & baseline).sum()),
                baseline_successes=int(baseline.sum()),gt_y_successes=int(yz.sum()),
                gain_vs_baseline=paired_bootstrap(selected.astype(float)-baseline,seed=281),
                gain_vs_gt_y=paired_bootstrap(selected.astype(float)-yz,seed=281),
                agrees_with_gt_y=int((chosen==ychosen).sum()),
                selection_counts=np.bincount(chosen,minlength=7).tolist(),
                ranking_vs_Z=same_state_ranking(z,scores,np.ones_like(z,dtype=bool)),
                ranking_vs_Y=same_state_ranking(y_scores,scores,np.ones_like(z,dtype=bool)))
