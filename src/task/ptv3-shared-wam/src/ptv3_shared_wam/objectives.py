"""Separately normalized masked hand/scene MSE in metric displacement units."""
import torch


def masked_mean(values, mask, weights=None):
    if values.shape != mask.shape:
        raise ValueError("loss and validity shapes differ")
    if weights is None:
        weights = torch.ones_like(values)
    weights = torch.where(mask, weights, 0.)
    return (torch.where(mask, values, 0.) * weights).sum() / weights.sum().clamp_min(1e-8)


def forecast_loss(prediction, batch, config):
    scale = float(config.get("loss_scale_m", .05))
    if scale <= 0:
        raise ValueError("positive metric loss scale required")
    hand_valid = batch["trajectory_valid"] & batch["start_hand_valid"][:, None]
    scene_valid = batch["scene_valid"] & batch["object_valid"][:, :, None, None]
    hand_delta = torch.where(hand_valid[..., None],
                             prediction["trajectory"] - batch["trajectory_target"], 0.)
    scene_delta = torch.where(scene_valid[..., None], prediction["scene"] - batch["scene_target"], 0.)
    hand_error = (hand_delta / scale).square().mean(-1)
    scene_error = (scene_delta / scale).square().mean(-1)
    motion = batch["scene_target"].float().norm(dim=-1)
    tau, floor = float(config.get("motion_tau_m", .002)), float(config.get("motion_floor", .1))
    if tau <= 0 or not 0 < floor <= 1:
        raise ValueError("positive motion scale and nonzero static weight required")
    weight = floor + (1. - floor) * torch.sigmoid(5. * (motion / tau - 1.))
    hand = masked_mean(hand_error, hand_valid)
    scene = masked_mean(scene_error, scene_valid, weight)
    total = float(config.get("hand_weight", 1.)) * hand + float(config.get("scene_weight", 1.)) * scene
    return total, dict(hand=hand, scene=scene)


def forecast_metrics(prediction, batch, config):
    """Point-count aggregation; missing categories return None, never fake zero."""
    hand = (prediction["trajectory"] - batch["trajectory_target"]).float().norm(dim=-1) * 1000
    scene = (prediction["scene"] - batch["scene_target"]).float().norm(dim=-1) * 1000
    valid = batch["scene_valid"] & batch["object_valid"][:, :, None, None]
    moving = batch["scene_target"].float().norm(dim=-1).amax((2, 3)) >= config.get("motion_tau_m", .002)
    masks = dict(hand_first8=(hand[:, :8], batch["trajectory_valid"][:, :8]),
                 hand_all24=(hand, batch["trajectory_valid"]),
                 scene_all24=(scene, valid), scene_h24=(scene[:, :, -1], valid[:, :, -1]),
                 scene_moving=(scene, valid & moving[:, :, None, None]),
                 scene_static=(scene, valid & ~moving[:, :, None, None]))
    result = {}
    for name, (values, mask) in masks.items():
        result[name] = dict(sum_mm=float(values[mask].sum().detach()), count=int(mask.sum()))
    return result
