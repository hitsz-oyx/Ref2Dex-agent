"""The v2 masked objective, normalized over all DDP ranks' valid points."""
import torch
import torch.distributed as dist
from ptv3_shared_wam_v2.objectives import forecast_loss as local_loss, forecast_metrics


def loss_denominators(batch, config):
    hand = batch['trajectory_valid'] & batch['start_hand_valid'][:,None]
    scene = batch['scene_valid'] & batch['object_valid'][:,:,None,None]
    motion = torch.where(scene[...,None], batch['scene_target'].float(), 0.).norm(dim=-1)
    weight = config['motion_floor']+(1-config['motion_floor'])*torch.sigmoid(5*(motion/config['motion_tau_m']-1))
    return torch.stack([hand.float().sum(),torch.where(scene,weight,0.).sum()])


def forecast_loss(prediction,batch,config,normalizer,distributed=False):
    total,terms=local_loss(prediction,batch,config,normalizer)
    if not distributed or not dist.is_initialized():return total,terms
    local=loss_denominators(batch,config)
    global_counts=local.detach().clone()
    dist.all_reduce(global_counts)
    # DDP averages gradients; compensate world size to recover sum(numerators)/sum(weights).
    factors=dist.get_world_size()*local/global_counts.clamp_min(1e-8)
    terms={name:terms[name]*factors[i] for i,name in enumerate(('hand','scene'))}
    return config['hand_weight']*terms['hand']+config['scene_weight']*terms['scene'],terms
