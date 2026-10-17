"""Object-centric interaction contract for WAM-v4.

C is deliberately embodiment-light: contact likelihood, hand/object proximity,
relative approach and relative velocity.  It is computed from existing hand and
object future labels, so adding the head does not require a second data corpus.
"""
import torch

C_NAMES = ("contact", "proximity_m", "relative_x_m", "relative_y_m", "relative_z_m",
           "relative_vx_mps", "relative_vy_mps", "relative_vz_mps")
C_DIM = len(C_NAMES)


def _masked_mean(value, mask, dim):
    weight = mask.to(value.dtype)
    return (value * weight.unsqueeze(-1)).sum(dim) / weight.sum(dim).clamp_min(1).unsqueeze(-1)


def interaction_target(batch, contact_threshold_m=0.012, contact_temperature_m=0.003,
                       fps=30.0):
    """Build C labels from the same current-anchor hand/object supervision."""
    start_hand = batch["start_hand"].float()                 # B,2,11,3
    hand_delta = batch["trajectory_target"].float()         # B,T,2,11,3
    hand_valid = batch["start_hand_valid"].bool()            # B,2,11
    trajectory_valid = batch.get("trajectory_valid", torch.ones_like(hand_delta[..., 0], dtype=torch.bool)).bool()
    start_hand = torch.where(hand_valid[..., None], start_hand, torch.zeros_like(start_hand))
    hand_delta = torch.where(trajectory_valid[..., None], hand_delta, torch.zeros_like(hand_delta))
    hand_future = start_hand[:, None] + hand_delta
    hand_mask = (hand_valid[:, None] & trajectory_valid).expand_as(hand_future[..., 0])
    hand_center = _masked_mean(hand_future, hand_mask, dim=3)  # B,T,2,3
    side_valid = hand_valid.any(-1)[:, None].expand(hand_center.shape[:3])
    hand_center = _masked_mean(hand_center, side_valid, dim=2)

    points = batch["scene_points"].float()                   # B,O,P,3
    object_valid = batch["object_valid"].bool()
    point_valid = batch.get("scene_valid")
    if point_valid is None:
        point_valid = torch.ones(object_future.shape[:-1], dtype=torch.bool, device=points.device)
    point_valid = point_valid.bool() & object_valid[:, :, None, None]
    points = torch.where(object_valid[:, :, None, None, None], points[:, :, None], torch.zeros_like(points[:, :, None]))
    scene_delta = torch.where(point_valid[..., None], batch["scene_target"].float(),
                              torch.zeros_like(batch["scene_target"].float()))
    object_future = points + scene_delta                           # B,O,T,P,3
    object_center = _masked_mean(object_future, point_valid, dim=3)  # B,O,T,3

    hand_points = hand_future.reshape(hand_future.shape[0], hand_future.shape[1], -1, 3)
    hand_point_valid = hand_mask.reshape(hand_mask.shape[0], hand_mask.shape[1], -1)
    pair = object_future[:, :, :, None, :, :] - hand_points[:, None, :, :, None, :]
    pair_distance = pair.square().sum(-1).sqrt()
    pair_valid = point_valid[:, :, :, None, :] & hand_point_valid[:, None, :, :, None]
    pair_distance = torch.where(pair_valid, pair_distance, torch.full_like(pair_distance, float("inf")))
    nearest = pair_distance.amin(dim=(-1, -2))
    nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
    contact = torch.sigmoid((contact_threshold_m - nearest) / max(contact_temperature_m, 1e-6))

    relative = object_center - hand_center[:, None]
    relative_velocity = torch.zeros_like(relative)
    relative_velocity[:, :, 1:] = (relative[:, :, 1:] - relative[:, :, :-1]) * float(fps)
    target = torch.cat((contact[..., None], nearest[..., None], relative, relative_velocity), dim=-1)
    has_hand = hand_valid.any(dim=(-1, -2))
    valid = object_valid[:, :, None] & point_valid.any(-1) & has_hand[:, None, None]
    return torch.where(valid[..., None], target, torch.zeros_like(target)), valid


def interaction_scales(config=None, device=None, dtype=torch.float32):
    values = (1.0, 0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
    if config is not None:
        values = tuple(config.get("interaction_scales", values))
    if len(values) != C_DIM or any(v <= 0 for v in values):
        raise ValueError("interaction_scales must contain eight positive values")
    return torch.tensor(values, device=device, dtype=dtype)
