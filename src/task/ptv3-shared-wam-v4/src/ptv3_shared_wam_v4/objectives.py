"""Masked v4 objective: v3 forecasts plus a voxel-derived interaction field."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ptv3_shared_wam_v2.objectives import forecast_loss as base_loss, forecast_metrics as base_metrics
from .interaction import interaction_target


def _interaction_loss(prediction, batch, config, distributed=False):
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
    edge_count = edge_mask.float().sum()
    positive_count = positive.float().sum()
    # DDP averages gradients across ranks.  Scale each rank's local numerator
    # by world size and normalize with the corresponding global count; using
    # the BCE denominator for both terms makes the semantic CE depend on how
    # positive contacts happen to be split across GPUs.
    # Evaluation runs on rank0 while the other ranks wait at a barrier.
    # Only training calls made by every rank may reduce label counts.
    if distributed and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        global_edge = edge_count.detach().clone()
        global_positive = positive_count.detach().clone()
        dist.all_reduce(global_edge)
        dist.all_reduce(global_positive)
        scale = float(dist.get_world_size())
        c_loss = (scale * bce.sum() / global_edge.clamp_min(1.)
                  + scale * part_loss.sum() / global_positive.clamp_min(1.))
    else:
        c_loss = bce.sum() / edge_count.clamp_min(1.) + part_loss.sum() / positive_count.clamp_min(1.)
    return c_loss, edge_count.detach(), positive_count.detach()


def forecast_loss(prediction, batch, config, normalizer, distributed=False):
    total, terms = base_loss(prediction, batch, config, normalizer)
    c_loss, _, _ = _interaction_loss(prediction, batch, config, distributed=distributed)
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
    result["interaction/contact_mae_all"] = dict(
        sum_mm=float(torch.where(valid, (confidence - target[..., 0]).abs(),
                                  torch.zeros_like(confidence)).sum()), count=int(valid.sum()))
    positive_parts = valid & (target[..., 0] > config.get("contact_metric_threshold", .25))
    if positive_parts.any():
        target_part = target[..., 1:].argmax(-1)
        predicted_part = prediction["interaction"][..., 1:].argmax(-1)
        correct = positive_parts & (target_part == predicted_part)
        log_prob = torch.log_softmax(prediction["interaction_logits"][..., 1:], -1)
        part_ce = -(target[..., 1:] * log_prob).sum(-1)
        result["interaction/part_accuracy"] = dict(sum_mm=float(correct.sum()),
                                                     count=int(positive_parts.sum()))
        result["interaction/part_ce"] = dict(
            sum_mm=float(torch.where(positive_parts, part_ce, torch.zeros_like(part_ce)).sum()),
            count=int(positive_parts.sum()))
    else:
        result["interaction/part_accuracy"] = dict(sum_mm=0., count=0)
        result["interaction/part_ce"] = dict(sum_mm=0., count=0)
    return result


forecast_metrics = metrics
