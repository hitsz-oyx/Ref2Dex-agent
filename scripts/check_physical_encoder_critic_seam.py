"""Tiny CPU transfer and decoder-removal engineering seam, no task data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.option_model_policy import initialized_network
from src.task.CmResidual.physical_encoder_critic import transferred_critic,ENCODER_KEYS
physical=initialized_network('dynamics');critic=transferred_critic(physical.state_dict());head=initialized_network('value')
for key in ENCODER_KEYS:assert torch.equal(critic.state_dict()[key],physical.state_dict()[key])
for key in ('4.weight','4.bias'):assert torch.equal(critic.state_dict()[key],head.state_dict()[key])
assert critic[4].out_features==1 and all(p.requires_grad for p in critic.parameters())
x=torch.randn(3,164,requires_grad=True);pred=critic(x);assert pred.shape==(3,1)
g=torch.autograd.grad(pred.sum(),x)[0];assert torch.isfinite(g).all() and g[:,152:].abs().max()>0
print('PASS exact encoder transfer/common scalar head/physical decoder removal and actual-action Q gradient. CPU engineering only.')
