"""Masked v4 objective: v3 forecasts plus a voxel-derived interaction field."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ptv3_shared_wam_v2.objectives import forecast_loss as base_loss, forecast_metrics as base_metrics
from .interaction import interaction_target


def _interaction_loss(prediction, batch, config):
    target, valid = interaction_target(batch)
    logits = prediction["interaction_logits"]
    contact = target[..., 0]
    edge_mask = valid
    bce = F.binary_cross_entropy_with_logits(logits[..., 0], contact, reduction="none")
    bce = torch.where(edge_mask, bce, torch.zeros_like(bce))
    part_target = target[..., 1:]
    part_logprob = F.log_softmax(logits[..., 1:], dim=-1)
    part_loss = -(part_target * part_logprob).sum(-1)
    positive = edge_mask & (contact > 0)
    part_loss = torch.where(positive, part_loss, torch.zeros_like(part_loss))
    denom = edge_mask.float().sum().clamp_min(1.)
    positive_denom = positive.float().sum().clamp_min(1.)
    return bce.sum() / denom + part_loss.sum() / positive_denom, denom.detach()


def forecast_loss(prediction, batch, config, normalizer, distributed=False):
    total, terms = base_loss(prediction, batch, config, normalizer)
    c_loss, local_count = _interaction_loss(prediction, batch, config)
    if distributed and dist.is_initialized():
        count = local_count.clone()
        dist.all_reduce(count)
        c_loss = c_loss * (dist.get_world_size() * local_count / count.clamp_min(1.))
    terms = dict(terms, interaction=c_loss, interaction_consistency=c_loss.new_zeros(()))
    total = total + config.get("interaction_weight", 1.0) * c_loss
    return total, terms


def metrics(prediction, batch, config):
    result = dict(base_metrics(prediction, batch, config))
    target, valid = interaction_target(batch)
    error = (prediction["interaction"] - target).abs()
    mask = valid[..., None].expand_as(error)
    result["interaction/mae"] = dict(
        sum_mm=float(torch.where(mask, error, torch.zeros_like(error)).sum()), count=int(mask.sum()))
    confidence = prediction["interaction"][..., 0]
    contact_mask = valid & (target[..., 0] > config.get("contact_metric_threshold", .25))
    result["interaction/contact_mae"] = dict(
        sum_mm=float(torch.where(contact_mask, (confidence - target[..., 0]).abs(),
                                  torch.zeros_like(confidence)).sum()), count=int(contact_mask.sum()))
    return result


forecast_metrics = metrics
