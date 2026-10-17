"""Dynamic object-centric hand--object interaction graph for WAM-v4.

The graph is independent of a robot hand.  For every forecast time,
canonical hand keypoint and object query point it predicts a small edge state.
The first implementation uses the existing 32 object query points; a dense
surface view can replace them without changing the edge contract.
"""
import torch
import torch.nn.functional as F


C_NAMES = (
    "contact_confidence",
    "surface_x_m", "surface_y_m", "surface_z_m",
    "normal_x", "normal_y", "normal_z",
    "approach_velocity_mps", "slip_velocity_mps",
)
C_DIM = len(C_NAMES)
C_HORIZON = 24
# Compatibility alias; v4 now emits every horizon step instead of only 8/16/24.
C_ANCHORS = C_HORIZON
C_STEPS = tuple(range(1, C_HORIZON + 1))


def _masked_mean(value, mask, dim):
    weight = mask.to(value.dtype)
    # ``value`` has one trailing channel axis that ``mask`` does not.
    mask_dim = dim if dim >= 0 else dim + 1
    return (value * weight.unsqueeze(-1)).sum(dim) / weight.sum(mask_dim).clamp_min(1).unsqueeze(-1)


def _future_valid(batch, points):
    B, O, P, _ = points.shape
    scene_valid = batch.get("scene_valid")
    if scene_valid is None:
        scene_valid = torch.ones(B, O, C_HORIZON, P, dtype=torch.bool, device=points.device)
    return scene_valid.bool() & batch["object_valid"].bool()[:, :, None, None]


def interaction_target(batch, sigma_m=0.010, fps=30.0):
    """Build ``[B,O,T,2,11,P,9]`` dynamic interaction labels.

    Contact is a smooth distance label.  Surface position and radial normal
    are expressed in the object query frame.  Approach and slip use relative
    hand/object point velocity.  Radial normals are the documented 32-point
    approximation and are masked when a query point or hand keypoint is invalid.
    """
    if sigma_m <= 0 or fps <= 0:
        raise ValueError("positive interaction sigma and fps required")
    start_hand = batch["start_hand"].float()                 # B,2,11,3
    hand_delta = batch["trajectory_target"].float()          # B,T,2,11,3
    hand_valid = batch["start_hand_valid"].bool()            # B,2,11
    trajectory_valid = batch.get(
        "trajectory_valid", torch.ones_like(hand_delta[..., 0], dtype=torch.bool)
    ).bool()
    hand_valid_future = trajectory_valid & hand_valid[:, None]
    hand_future = start_hand[:, None] + torch.where(
        hand_valid_future[..., None], hand_delta, torch.zeros_like(hand_delta)
    )

    points = batch["scene_points"].float()                   # B,O,P,3
    object_valid = batch["object_valid"].bool()              # B,O
    point_valid = _future_valid(batch, points)                # B,O,T,P
    scene_delta = torch.nan_to_num(batch["scene_target"].float())
    scene_delta = torch.where(point_valid[..., None], scene_delta, torch.zeros_like(scene_delta))
    object_future = points[:, :, None] + scene_delta          # B,O,T,P,3
    object_center = _masked_mean(object_future, point_valid, dim=-2)  # B,O,T,3
    radial = object_future - object_center[..., None, :]
    normals = F.normalize(radial, dim=-1, eps=1e-6)

    pair = object_future[:, :, :, None, None, :, :] - hand_future[:, None, :, :, :, None, :]
    distance = pair.square().sum(-1).sqrt()                  # B,O,T,2,11,P
    pair_valid = (point_valid[:, :, :, None, None, :] &
                  hand_valid_future[:, None, :, :, :, None])
    distance = torch.where(pair_valid, distance, torch.full_like(distance, float("inf")))
    contact = torch.exp(-distance.square() / (2.0 * float(sigma_m) ** 2))
    contact = torch.where(pair_valid, contact, torch.zeros_like(contact))

    surface = object_future - object_center[..., None, :]
    surface = surface[:, :, :, None, None, :, :].expand_as(pair)
    normal = normals[:, :, :, None, None, :, :].expand_as(pair)

    object_velocity = torch.zeros_like(object_future)
    object_velocity[:, :, 0] = (object_future[:, :, 0] - points) * float(fps)
    object_velocity[:, :, 1:] = (object_future[:, :, 1:] - object_future[:, :, :-1]) * float(fps)
    hand_velocity = torch.zeros_like(hand_future)
    hand_velocity[:, 0] = (hand_future[:, 0] - start_hand) * float(fps)
    hand_velocity[:, 1:] = (hand_future[:, 1:] - hand_future[:, :-1]) * float(fps)
    relative_velocity = (hand_velocity[:, None, :, :, :, None, :] -
                         object_velocity[:, :, :, None, None, :, :])
    approach = -(relative_velocity * normal).sum(-1)
    tangential = relative_velocity + approach[..., None] * normal
    slip = tangential.square().sum(-1).sqrt()

    target = torch.cat((contact[..., None], surface, normal,
                        approach[..., None], slip[..., None]), dim=-1)
    return torch.where(pair_valid[..., None], target, torch.zeros_like(target)), pair_valid


def interaction_scales(config=None, device=None, dtype=torch.float32):
    values = (1.0, 0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 0.5, 0.5)
    if config is not None:
        values = tuple(config.get("interaction_scales", values))
    if len(values) != C_DIM or any(v <= 0 for v in values):
        raise ValueError("interaction_scales must contain nine positive values")
    return torch.tensor(values, device=device, dtype=dtype)
