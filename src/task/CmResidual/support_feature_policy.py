"""Return-trained macro policy with frozen physical probability features.

This module neither chooses a model-greedy action nor substitutes predictions for
the actual reward. The score-loss importance ratio is deliberately detached.
"""
import torch

SCHEMA = 'support69-probability8-return-trained-macro-v1'


def actor(seed=752):
    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(77,64), torch.nn.ReLU(),
        torch.nn.Linear(64,64), torch.nn.ReLU(),
        torch.nn.Linear(64,8))
    with torch.no_grad():
        network[-1].weight.zero_()
        network[-1].bias.zero_()
        network[-1].bias[0] = 2
    return network


def policy_input(normalized_state, physical_probability):
    if normalized_state.ndim != 2 or normalized_state.shape[1] != 69:
        raise ValueError('69 current normalized features required')
    if physical_probability.shape != (len(normalized_state),8):
        raise ValueError('eight frozen physical probabilities required')
    if not torch.isfinite(normalized_state).all() or not torch.isfinite(physical_probability).all():
        raise ValueError('finite features')
    if (physical_probability < 0).any() or (physical_probability > 1).any():
        raise ValueError('physical probability range')
    return torch.cat((normalized_state.clamp(-10,10),2*physical_probability-1),dim=-1).detach()


def offpolicy_score_loss(logits, action, reward, behavior_probability, state_baseline):
    n = len(logits)
    if logits.shape != (n,8) or any(v.shape != (n,) for v in (action,reward,behavior_probability,state_baseline)):
        raise ValueError('macro score schema')
    if action.dtype != torch.long or (action < 0).any() or (action >= 8).any():
        raise ValueError('eight executable actions')
    if not all(torch.isfinite(v).all() for v in (logits,reward,behavior_probability,state_baseline)):
        raise ValueError('finite gradient inputs')
    if (behavior_probability <= 0).any() or (behavior_probability > 1).any():
        raise ValueError('known positive behavior probability')
    log_probability = torch.log_softmax(logits,dim=-1).gather(1,action[:,None]).squeeze(1)
    importance = (log_probability.exp()/behavior_probability).detach()
    # state_baseline must be fixed before assignment and constant across actions.
    # The caller supplies this explicitly; action-dependent nuisance values are
    # not justified by this loss and require a different estimator.
    loss = -(importance*(reward-state_baseline).detach()*log_probability).mean()
    if not torch.isfinite(loss):
        raise ValueError('finite score loss')
    return loss
