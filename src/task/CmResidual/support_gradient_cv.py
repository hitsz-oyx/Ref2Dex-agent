"""Finite-action gradient control variate; a standard unbiased correction."""
import torch

def corrected_logit_gradient(probability,behavior,action,reward,prediction):
    if prediction.shape!=probability.shape or behavior.shape!=probability.shape or action.shape!=reward.shape or action.shape!=probability.shape[:1]:raise ValueError('gradient schema')
    if (probability<=0).any() or (behavior<=0).any() or not torch.allclose(probability.sum(-1),torch.ones_like(reward)) or not torch.allclose(behavior.sum(-1),torch.ones_like(reward)):raise ValueError('positive normalized action probabilities')
    ids=torch.arange(len(action),device=action.device);score=torch.nn.functional.one_hot(action,probability.shape[-1]).to(probability)-probability
    importance=probability[ids,action]/behavior[ids,action]
    residual=importance[:,None]*score*(reward-prediction[ids,action])[:,None]
    analytic=probability*(prediction-(probability*prediction).sum(-1,keepdim=True))
    result=residual+analytic
    if not torch.isfinite(result).all():raise ValueError('finite gradient')
    return result
