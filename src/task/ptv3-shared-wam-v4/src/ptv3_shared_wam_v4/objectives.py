"""Masked v4 objective: legacy hand/scene forecast plus interaction C."""
import torch
import torch.distributed as dist
from ptv3_shared_wam_v2.objectives import forecast_loss as base_loss, forecast_metrics as base_metrics
from .interaction import interaction_target


def _interaction_loss(prediction, batch, config):
    target, valid = interaction_target(batch,
        contact_threshold_m=config.get("contact_threshold_m", .012),
        contact_temperature_m=config.get("contact_temperature_m", .003),
        fps=config.get("fps", 30.0))
    scale = prediction["interaction_normalized"].new_tensor(config.get(
        "interaction_scales", (1., .05, .05, .05, .05, .5, .5, .5)))
    target = target / scale
    error = torch.nn.functional.smooth_l1_loss(
        prediction["interaction_normalized"], target, reduction="none", beta=1.)
    mask = valid[..., None].expand_as(error)
    denom = mask.float().sum().clamp_min(1.)
    return torch.where(mask, error, torch.zeros_like(error)).sum() / denom, denom.detach()


def forecast_loss(prediction, batch, config, normalizer, distributed=False):
    total, terms = base_loss(prediction, batch, config, normalizer)
    c_loss, local_count = _interaction_loss(prediction, batch, config)
    if distributed and dist.is_initialized():
        count = local_count.clone()
        dist.all_reduce(count)
        c_loss = c_loss * (dist.get_world_size() * local_count / count.clamp_min(1.))
    terms = dict(terms, interaction=c_loss)
    return total + config.get("interaction_weight", 1.0) * c_loss, terms


def metrics(prediction, batch, config):
    result = dict(base_metrics(prediction, batch, config))
    target, valid = interaction_target(batch,
        contact_threshold_m=config.get("contact_threshold_m", .012),
        contact_temperature_m=config.get("contact_temperature_m", .003),
        fps=config.get("fps", 30.0))
    error = (prediction["interaction"] - target).abs()
    mask = valid[..., None].expand_as(error)
    result["interaction/mae"] = dict(sum=float(torch.where(mask, error, torch.zeros_like(error)).sum()),
        count=int(mask.sum()))
    return result


# Training keeps the v3 naming convention for evaluation hooks.
forecast_metrics = metrics
