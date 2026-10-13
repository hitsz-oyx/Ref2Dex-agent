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


def target_object_mask(batch):
    """Use the observed target identity, never select by future motion."""
    objects = batch["object_valid"]
    slots = batch["goal_object_slot"]
    if slots.shape != (len(objects),) or slots.dtype != torch.long:
        raise ValueError("one integer target-object slot per row required")
    if ((slots < 0) | (slots >= objects.shape[1])).any():
        raise ValueError("target-object slot outside scene")
    rows = torch.arange(len(objects), device=objects.device)
    if not objects[rows, slots].all():
        raise ValueError("target-object slot must identify an observed object")
    return torch.arange(objects.shape[1], device=objects.device)[None] == slots[:, None]


def goal_only_scene(batch):
    """Untrained baseline: linearly interpolate target delta; other objects stay.

    Point pairing is checked against current observations, not future labels.
    Missing goal points are excluded from the target baseline metrics.
    """
    points = batch["scene_points"]
    target = target_object_mask(batch)
    valid = batch["object_goal_valid"].bool()
    delta = batch["goal_object_delta"]
    if delta.shape != (len(points), points.shape[2], 3) or valid.shape != delta.shape[:-1]:
        raise ValueError("goal and scene baseline require the same paired point count")
    rows = torch.arange(len(points), device=points.device)
    current = points[rows, batch["goal_object_slot"]]
    if not torch.allclose(current[valid], batch["start_object_points"][valid], atol=1e-6, rtol=1e-5):
        raise ValueError("goal-only baseline requires matching current scene/goal point identities")
    delta = torch.where(valid[..., None], delta.float(), 0.)
    if not torch.isfinite(delta).all():
        raise ValueError("finite valid goal delta required")
    fractions = torch.arange(1, 25, device=points.device, dtype=delta.dtype) / 24.
    prediction = delta[:, None, None] * fractions[None, None, :, None, None]
    prediction = prediction * target[:, :, None, None, None]
    supported = (~target[:, :, None, None] | valid[:, None, None]).expand(-1, -1, 24, -1)
    return prediction, supported


def scene_forecast_metrics(prediction, batch, config, prediction_valid=None):
    """EPE in mm, with explicit temporal and target/non-target partitions."""
    scene = (prediction - batch["scene_target"]).float().norm(dim=-1) * 1000
    valid = batch["scene_valid"] & batch["object_valid"][:, :, None, None]
    if prediction_valid is not None:
        valid = valid & prediction_valid
    target = target_object_mask(batch)[:, :, None, None]
    moving = batch["scene_target"].float().norm(dim=-1).amax((2, 3)) >= config.get("motion_tau_m", .002)
    masks = dict(scene_moving=(scene, valid & moving[:, :, None, None]),
                 scene_static=(scene, valid & ~moving[:, :, None, None]))
    windows = dict(h1=slice(0, 1), first8=slice(0, 8), middle=slice(8, 16),
                   h24=slice(23, 24), all24=slice(0, 24))
    for group, mask in (("scene", valid), ("scene_target", valid & target),
                        ("scene_non_target", valid & ~target)):
        for window, interval in windows.items():
            masks[group + "_" + window] = (scene[:, :, interval], mask[:, :, interval])
    return metric_sums(masks)


def metric_sums(masks):
    """Point-count aggregation; empty categories retain count=0."""
    result = {}
    for name, (values, mask) in masks.items():
        result[name] = dict(sum_mm=float(values[mask].sum().detach()), count=int(mask.sum()))
    return result


def forecast_metrics(prediction, batch, config):
    hand = (prediction["trajectory"] - batch["trajectory_target"]).float().norm(dim=-1) * 1000
    hand_valid = batch["trajectory_valid"] & batch["start_hand_valid"][:, None]
    result = metric_sums(dict(hand_first8=(hand[:, :8], hand_valid[:, :8]),
                              hand_all24=(hand, hand_valid)))
    result.update(scene_forecast_metrics(prediction["scene"], batch, config))
    baseline, supported = goal_only_scene(batch)
    result.update({"goal_only/" + key: value for key, value in
                   scene_forecast_metrics(baseline, batch, config, supported).items()})
    return result
