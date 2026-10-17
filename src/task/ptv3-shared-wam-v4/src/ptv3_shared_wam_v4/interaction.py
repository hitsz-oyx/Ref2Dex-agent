"""Structured object-centric interaction targets for WAM-v4.

The reference contract uses three horizon anchors (8, 16 and 24), eleven
canonical hand points, and nine interpretable channels per point:
contact confidence, approximate object-surface position, surface normal,
normal approach velocity and tangential slip speed.  The current v4 probe
uses the audited 32-point object query as a bounded approximation.  A future
dense-surface data view can replace the query points without changing the
head contract.
"""
import torch


C_NAMES = (
    "contact_confidence",
    "surface_x_m", "surface_y_m", "surface_z_m",
    "normal_x", "normal_y", "normal_z",
    "approach_velocity_mps", "slip_velocity_mps",
)
C_DIM = len(C_NAMES)
C_ANCHORS = 3
C_ANCHOR_STEPS = (8, 16, 24)
C_ANCHOR_INDICES = tuple(step - 1 for step in C_ANCHOR_STEPS)


def _masked_mean(value, mask, dim):
    weight = mask.to(value.dtype)
    value_dim = dim if dim >= 0 else value.ndim + dim
    # The mask has no trailing channel dimension, but all spatial/time axes
    # keep the same indices as the value tensor.
    mask_dim = value_dim
    return ((value * weight.unsqueeze(-1)).sum(dim)
            / weight.sum(mask_dim).clamp_min(1).unsqueeze(-1))


def _approximate_normals(points, valid):
    """Estimate normals from the current 32-point object query."""
    center = _masked_mean(points, valid, dim=-2)
    radial = points - center.unsqueeze(-2)
    normal = torch.nn.functional.normalize(radial, dim=-1, eps=1e-6)
    return torch.where(valid[..., None], normal, torch.zeros_like(normal))


def interaction_target(batch, sigma_m=0.010, fps=30.0):
    """Build ``[B,O,3,11,9]`` C labels from current-anchor supervision.

    For each anchor and canonical keypoint, the nearest valid query point over
    the two hand sides is selected.  The selected query point is centered by
    the future object centroid, giving an approximate object-surface position.
    Normals are radial because the v3 32-point contract has no dense CAD normal
    field.  The velocity channels use relative hand/object-centroid motion.
    """
    if sigma_m <= 0 or fps <= 0:
        raise ValueError("positive interaction sigma and fps required")
    start_hand = batch["start_hand"].float()                  # B,2,11,3
    hand_delta = batch["trajectory_target"].float()           # B,T,2,11,3
    hand_valid = batch["start_hand_valid"].bool()             # B,2,11
    trajectory_valid = batch.get(
        "trajectory_valid", torch.ones_like(hand_delta[..., 0], dtype=torch.bool)
    ).bool()
    hand_valid_future = trajectory_valid & hand_valid[:, None]
    start_hand = torch.where(hand_valid[..., None], start_hand, torch.zeros_like(start_hand))
    hand_delta = torch.where(hand_valid_future[..., None], hand_delta, torch.zeros_like(hand_delta))
    hand_future = start_hand[:, None] + hand_delta              # B,T,2,11,3

    points = batch["scene_points"].float()                      # B,O,P,3
    object_valid = batch["object_valid"].bool()                 # B,O
    scene_valid = batch.get("scene_valid")
    if scene_valid is None:
        scene_valid = torch.ones(batch["scene_target"].shape[:-1], dtype=torch.bool,
                                 device=points.device)
    scene_valid = scene_valid.bool() & object_valid[:, :, None, None]
    scene_delta = torch.nan_to_num(batch["scene_target"].float())
    scene_delta = torch.where(scene_valid[..., None], scene_delta, torch.zeros_like(scene_delta))
    object_future = points[:, :, None] + scene_delta              # B,O,T,P,3
    point_valid = scene_valid
    object_center = _masked_mean(object_future, point_valid, dim=-2)  # B,O,T,3
    normals = _approximate_normals(object_future, point_valid)

    # Pair each object query with both hands and all canonical keypoints.
    pair = object_future[:, :, :, None, None, :, :] - hand_future[:, None, :, :, :, None, :]
    distance = pair.square().sum(-1).sqrt()                     # B,O,T,2,11,P
    pair_valid = point_valid[:, :, :, None, None, :] & hand_valid_future[:, None, :, :, :, None]
    distance = torch.where(pair_valid, distance, torch.full_like(distance, float("inf")))
    point_distance, point_id = distance.amin(dim=-1), distance.argmin(dim=-1)
    nearest_distance, side = point_distance.amin(dim=3), point_distance.argmin(dim=3)
    point_id = point_id.gather(3, side.unsqueeze(3)).squeeze(3)  # B,O,T,11

    # Gather the selected object query and matching hand side.
    object_index = point_id.unsqueeze(-1).expand(-1, -1, -1, -1, 3)
    object_for_gather = object_future.unsqueeze(3).expand(-1, -1, -1, 11, -1, -1)
    selected_object = object_for_gather.gather(4, object_index.unsqueeze(3)).squeeze(3)
    normal_for_gather = normals.unsqueeze(3).expand(-1, -1, -1, 11, -1, -1)
    selected_normal = normal_for_gather.gather(4, object_index.unsqueeze(3)).squeeze(3)
    selected_object = selected_object - object_center[:, :, :, None, :]
    selected_normal = torch.nn.functional.normalize(selected_normal, dim=-1, eps=1e-6)
    contact = torch.exp(-nearest_distance.square() / (2.0 * float(sigma_m) ** 2))
    contact = torch.where(torch.isfinite(nearest_distance), contact, torch.zeros_like(contact))

    hand_side = side.unsqueeze(3).unsqueeze(-1).expand(-1, -1, -1, 1, 11, 3)
    hand_for_gather = hand_future[:, None].expand(-1, points.shape[1], -1, -1, -1, -1)
    selected_hand = hand_for_gather.gather(3, hand_side).squeeze(3)
    object_velocity = torch.zeros_like(object_center)
    object_velocity[:, :, 0] = (object_center[:, :, 0] - points.mean(2)) * float(fps)
    object_velocity[:, :, 1:] = (object_center[:, :, 1:] - object_center[:, :, :-1]) * float(fps)
    hand_velocity = torch.zeros_like(selected_hand)
    hand_velocity[:, :, 0] = (selected_hand[:, :, 0] - start_hand.mean(1)[:, None]) * float(fps)
    hand_velocity[:, :, 1:] = (selected_hand[:, :, 1:] - selected_hand[:, :, :-1]) * float(fps)
    relative_velocity = hand_velocity - object_velocity[:, :, :, None]
    normal_velocity = -(relative_velocity * selected_normal).sum(-1)
    tangential = relative_velocity - normal_velocity[..., None].neg() * selected_normal
    slip_velocity = tangential.square().sum(-1).sqrt()

    take = torch.as_tensor(C_ANCHOR_INDICES, device=points.device)
    take4 = take.view(1, 1, C_ANCHORS, 1)
    take5 = take.view(1, 1, C_ANCHORS, 1, 1)
    contact = contact.gather(2, take4.expand(contact.shape[0], contact.shape[1], -1, 11))
    selected_object = selected_object.gather(2, take5.expand(selected_object.shape[0], selected_object.shape[1], -1, 11, 3))
    selected_normal = selected_normal.gather(2, take5.expand(selected_normal.shape[0], selected_normal.shape[1], -1, 11, 3))
    normal_velocity = normal_velocity.gather(2, take4.expand(normal_velocity.shape[0], normal_velocity.shape[1], -1, 11))
    slip_velocity = slip_velocity.gather(2, take4.expand(slip_velocity.shape[0], slip_velocity.shape[1], -1, 11))
    valid = (object_valid[:, :, None, None]
             & point_valid[:, :, C_ANCHOR_INDICES].any(-1)[:, :, :, None]
             & hand_valid_future[:, C_ANCHOR_INDICES].any(2)[:, None, :, :])
    target = torch.cat((contact[..., None], selected_object, selected_normal,
                        normal_velocity[..., None], slip_velocity[..., None]), dim=-1)
    return torch.where(valid[..., None], target, torch.zeros_like(target)), valid


def interaction_scales(config=None, device=None, dtype=torch.float32):
    values = (1.0, 0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 0.5, 0.5)
    if config is not None:
        values = tuple(config.get("interaction_scales", values))
    if len(values) != C_DIM or any(v <= 0 for v in values):
        raise ValueError("interaction_scales must contain nine positive values")
    return torch.tensor(values, device=device, dtype=dtype)
