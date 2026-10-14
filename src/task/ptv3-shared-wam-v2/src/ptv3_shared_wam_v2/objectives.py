"""Loss in shared horizon-normalized units; metrics consume metre outputs."""
import torch
from . import paths
from ptv3_shared_wam.objectives import masked_mean, forecast_metrics


def forecast_loss(prediction, batch, config, normalizer):
    hand_valid = batch["trajectory_valid"] & batch["start_hand_valid"][:, None]
    scene_valid = batch["scene_valid"] & batch["object_valid"][:, :, None, None]
    terms = {}
    for name, key, mask in (("hand", "trajectory", hand_valid), ("scene", "scene", scene_valid)):
        target = torch.where(mask[..., None], batch[key + "_target"].float(), 0.)
        target = normalizer.normalize(name, target)
        error = torch.where(mask[..., None], prediction[key + "_normalized"] - target, 0.).square().mean(-1)
        weight = None
        if name == "scene":
            tau, floor = config["motion_tau_m"], config["motion_floor"]
            motion = torch.where(mask[..., None], batch["scene_target"].float(), 0.).norm(dim=-1)
            weight = floor + (1. - floor) * torch.sigmoid(5. * (motion / tau - 1.))
        terms[name] = masked_mean(error, mask, weight)
    return config["hand_weight"] * terms["hand"] + config["scene_weight"] * terms["scene"], terms
