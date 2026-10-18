"""Masked v4 objective: v3 forecasts plus a voxel-derived interaction field."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ptv3_shared_wam_v2.objectives import forecast_metrics as base_metrics
from .interaction import interaction_target


OBJECTIVE_SEMANTICS = "v4.global-masked-components.v2"
EVALUATION_SEMANTICS = "v4.source-macro-sufficient-statistics.v2"
COMPONENT_NAMES = ("hand", "scene", "contact", "part")


def _interaction_components(prediction, batch, reduction_dtype=None):
    target, valid = interaction_target(batch)
    logits = prediction["interaction_logits"]
    contact = target[..., 0]
    edge_mask = valid & batch.get("object_valid", torch.ones_like(valid[:, :, 0, 0]))[:, :, None, None]
    if "interaction_prediction_valid" in prediction:
        edge_mask = edge_mask & prediction["interaction_prediction_valid"]
    bce = F.binary_cross_entropy_with_logits(logits[..., 0], contact, reduction="none")
    bce = torch.where(edge_mask, bce, torch.zeros_like(bce))
    part_target = target[..., 1:]
    part_logprob = F.log_softmax(logits[..., 1:], dim=-1)
    part_loss = -(part_target * part_logprob).sum(-1)
    positive = edge_mask & (contact > 0)
    part_loss = torch.where(positive, part_loss, torch.zeros_like(part_loss))
    return dict(contact=(bce.sum(dtype=reduction_dtype), edge_mask.float().sum(dtype=reduction_dtype)),
                part=(part_loss.sum(dtype=reduction_dtype), positive.float().sum(dtype=reduction_dtype)))


def loss_components(prediction, batch, config, normalizer, reduction_dtype=None):
    """Unnormalized sufficient statistics, shared by training and evaluation."""
    result = {}
    hand_valid = batch["trajectory_valid"] & batch["start_hand_valid"][:, None]
    scene_valid = batch["scene_valid"] & batch["object_valid"][:, :, None, None]
    for name, key, mask in (("hand", "trajectory", hand_valid), ("scene", "scene", scene_valid)):
        target = torch.where(mask[..., None], batch[key + "_target"].float(), 0.)
        normalized = normalizer.normalize(name, target)
        error = torch.where(mask[..., None], prediction[key + "_normalized"] - normalized, 0.).square().mean(-1)
        weight = torch.ones_like(error)
        if name == "scene":
            motion = target.norm(dim=-1)
            tau, floor = config["motion_tau_m"], config["motion_floor"]
            if tau <= 0 or not 0 < floor <= 1:
                raise ValueError("positive motion scale and nonzero static weight required")
            weight = floor + (1-floor) * torch.sigmoid(5*(motion/tau-1))
        weight = torch.where(mask, weight, 0.)
        result[name] = ((error*weight).sum(dtype=reduction_dtype), weight.sum(dtype=reduction_dtype))
    result.update(_interaction_components(prediction, batch, reduction_dtype))
    return result


def normalized_components(components, distributed=False):
    denominators = torch.stack([pair[1].detach() for pair in components.values()])
    scale = 1
    if distributed and dist.is_available() and dist.is_initialized():
        scale = dist.get_world_size()
        if scale > 1:
            # DDP averages gradients; each local numerator needs world/global_Z.
            dist.all_reduce(denominators)
    return {key: scale*pair[0]/denominators[i].clamp_min(1e-8)
            for i, (key, pair) in enumerate(components.items())}


def combine_components(means, config):
    interaction = means["contact"] + means["part"]
    terms = dict(hand=means["hand"], scene=means["scene"], interaction=interaction,
                 contact=means["contact"], part=means["part"])
    return (config["hand_weight"]*means["hand"] + config["scene_weight"]*means["scene"]
            + config.get("interaction_weight", 1.)*interaction), terms


def _interaction_loss(prediction, batch, config, distributed=False):
    """Compatibility helper for C-only gradient probes."""
    components = _interaction_components(prediction, batch)
    means = normalized_components(components, distributed)
    return means["contact"]+means["part"], components["contact"][1].detach(), components["part"][1].detach()


def forecast_loss(prediction, batch, config, normalizer, distributed=False):
    components = loss_components(prediction, batch, config, normalizer)
    total, terms = combine_components(normalized_components(components, distributed), config)
    terms["components"] = components
    return total, terms


def metrics(prediction, batch, config):
    result = {key:dict(sum=value["sum_mm"], count=value["count"], unit="mm")
              for key, value in base_metrics(prediction, batch, config).items()}
    target, valid = interaction_target(batch)
    valid = valid & batch["object_valid"][:, :, None, None]
    if "interaction_prediction_valid" in prediction:
        valid = valid & prediction["interaction_prediction_valid"]
    confidence = prediction["interaction"][..., 0]
    contact_mask = valid & (target[..., 0] > config.get("contact_metric_threshold", .25))
    result["interaction/contact_mae"] = dict(
        sum=float(torch.where(contact_mask, (confidence - target[..., 0]).abs(),
                             torch.zeros_like(confidence)).sum()), count=int(contact_mask.sum()), unit="1")
    result["interaction/contact_mae_all"] = dict(
        sum=float(torch.where(valid, (confidence - target[..., 0]).abs(),
                             torch.zeros_like(confidence)).sum()), count=int(valid.sum()), unit="1")
    positive_parts = valid & (target[..., 0] > config.get("contact_metric_threshold", .25))
    if positive_parts.any():
        target_part = target[..., 1:].argmax(-1)
        predicted_part = prediction["interaction"][..., 1:].argmax(-1)
        correct = positive_parts & (target_part == predicted_part)
        log_prob = torch.log_softmax(prediction["interaction_logits"][..., 1:], -1)
        part_ce = -(target[..., 1:] * log_prob).sum(-1)
        result["interaction/part_accuracy"] = dict(sum=float(correct.sum()),
                                                  count=int(positive_parts.sum()), unit="1")
        result["interaction/part_ce"] = dict(
            sum=float(torch.where(positive_parts, part_ce, torch.zeros_like(part_ce)).sum()),
            count=int(positive_parts.sum()), unit="nats")
    else:
        result["interaction/part_accuracy"] = dict(sum=0., count=0, unit="1")
        result["interaction/part_ce"] = dict(sum=0., count=0, unit="nats")
    return result


forecast_metrics = metrics
