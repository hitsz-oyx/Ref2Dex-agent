"""Masked v4 objective: v3 forecasts plus a dynamic interaction graph."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ptv3_shared_wam_v2.objectives import forecast_loss as base_loss, forecast_metrics as base_metrics
from .interaction import interaction_target


def _interaction_loss(prediction, batch, config):
    target, valid = interaction_target(batch,
        sigma_m=config.get("contact_sigma_m", .010), fps=config.get("fps", 30.0))
    scale = prediction["interaction_normalized"].new_tensor(config.get(
        "interaction_scales", (1., .05, .05, .05, 1., 1., 1., .5, .5)))
    edge_mask = valid
    logits = prediction["interaction_logits"][..., 0]
    contact = target[..., 0]
    bce = F.binary_cross_entropy_with_logits(logits, contact, reduction="none")
    bce = torch.where(edge_mask, bce, torch.zeros_like(bce))
    geometry_error = F.smooth_l1_loss(
        prediction["interaction_normalized"][..., 1:], target[..., 1:] / scale[1:],
        reduction="none", beta=1.)
    weight = contact[..., None] + float(config.get("interaction_background_weight", .05))
    geometry_error = torch.where(edge_mask[..., None], geometry_error * weight,
                                 torch.zeros_like(geometry_error))
    denom = edge_mask.float().sum().clamp_min(1.)
    return bce.sum() / denom + geometry_error.sum() / denom, denom.detach()


def _interaction_consistency_loss(prediction, batch, config):
    """Keep predicted contact edges near their predicted hand/object geometry."""
    hand = batch["start_hand"].float()[:, None] + prediction["trajectory"].float()
    objects = batch["scene_points"].float()[:, :, None] + prediction["scene"].float()
    pair = objects[:, :, :, None, None, :, :] - hand[:, None, :, :, :, None, :]
    distance = pair.square().sum(-1).sqrt()
    confidence = prediction["interaction"][..., 0]
    valid = prediction["interaction_valid"]
    hinge = torch.relu(distance - float(config.get("contact_sigma_m", .010))) * confidence
    return torch.where(valid, hinge, torch.zeros_like(hinge)).sum() / valid.float().sum().clamp_min(1.)


def forecast_loss(prediction, batch, config, normalizer, distributed=False):
    total, terms = base_loss(prediction, batch, config, normalizer)
    c_loss, local_count = _interaction_loss(prediction, batch, config)
    if distributed and dist.is_initialized():
        count = local_count.clone()
        dist.all_reduce(count)
        c_loss = c_loss * (dist.get_world_size() * local_count / count.clamp_min(1.))
    terms = dict(terms, interaction=c_loss)
    consistency = _interaction_consistency_loss(prediction, batch, config)
    terms["interaction_consistency"] = consistency
    total = total + config.get("interaction_weight", 1.0) * c_loss
    total = total + config.get("interaction_consistency_weight", .05) * consistency
    return total, terms


def metrics(prediction, batch, config):
    result = dict(base_metrics(prediction, batch, config))
    target, valid = interaction_target(batch,
        sigma_m=config.get("contact_sigma_m", .010), fps=config.get("fps", 30.0))
    error = (prediction["interaction"] - target).abs()
    mask = valid[..., None].expand_as(error)
    result["interaction/mae"] = dict(
        sum_mm=float(torch.where(mask, error, torch.zeros_like(error)).sum()),
        count=int(mask.sum()))
    confidence = prediction["interaction"][..., 0]
    contact_mask = valid & (target[..., 0] > config.get("contact_metric_threshold", .25))
    result["interaction/contact_mae"] = dict(
        sum_mm=float(torch.where(contact_mask, (confidence - target[..., 0]).abs(),
                                  torch.zeros_like(confidence)).sum()),
        count=int(contact_mask.sum()))
    return result


forecast_metrics = metrics
