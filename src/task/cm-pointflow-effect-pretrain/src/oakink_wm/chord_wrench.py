"""CHORD-style contact-wrench support utilities.

This module implements the geometric part of NVIDIA CHORD's contact-wrench
space comparison.  It deliberately accepts object-frame contact points and
normals rather than pretending that a point-flow or normalized tactile signal
contains those quantities.  The caller remains responsible for obtaining
contacts from a calibrated simulator or a separately audited reconstruction.
"""
from __future__ import annotations

import math

import torch


def contact_wrench_rays(
    points: torch.Tensor,
    normals: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    friction: float = 0.5,
    ray_count: int = 8,
    body_radius: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct normalized friction-cone wrench rays.

    Parameters are batched as ``[B,C,3]`` in the object COM frame.  The result
    has shape ``[B,C*(ray_count+1),6]`` and a matching boolean validity mask;
    ``ray_count`` is the number of friction-cone edge rays and one additional
    pure-normal ray is appended per contact, matching the released CHORD
    implementation.  The first three wrench components are force and the last three are the
    lever-arm torque divided by ``body_radius`` as in the released CHORD code.
    """
    if points.ndim != 3 or normals.shape != points.shape or points.shape[-1] != 3:
        raise ValueError("points and normals must both have shape [B,C,3]")
    if contact_mask.shape != points.shape[:2]:
        raise ValueError("contact_mask must have shape [B,C]")
    if ray_count < 1 or friction < 0 or body_radius <= 0:
        raise ValueError("ray_count >= 1, friction >= 0 and body_radius > 0 required")

    dtype = points.dtype
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-8
    n = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(eps)
    # Pick a reference axis that is not parallel to the normal, then form an
    # orthonormal tangent basis.  The branch is deterministic and finite for
    # every nonzero normal.
    ref_x = torch.zeros_like(n)
    ref_x[..., 0] = 1.0
    ref_y = torch.zeros_like(n)
    ref_y[..., 1] = 1.0
    ref = torch.where(n[..., :1].abs() < 0.9, ref_x, ref_y)
    tangent_1 = torch.linalg.cross(n, ref, dim=-1)
    tangent_1 = tangent_1 / torch.linalg.vector_norm(tangent_1, dim=-1, keepdim=True).clamp_min(eps)
    tangent_2 = torch.linalg.cross(n, tangent_1, dim=-1)
    angles = torch.arange(ray_count, device=points.device, dtype=dtype) * (2.0 * math.pi / ray_count)
    tangent = (
        tangent_1.unsqueeze(-2) * angles.cos().view(1, 1, ray_count, 1)
        + tangent_2.unsqueeze(-2) * angles.sin().view(1, 1, ray_count, 1)
    )
    edge_force = n.unsqueeze(-2) + float(friction) * tangent
    edge_force = edge_force / torch.linalg.vector_norm(edge_force, dim=-1, keepdim=True).clamp_min(eps)
    force = torch.cat((edge_force, n.unsqueeze(-2)), dim=-2)
    torque = torch.linalg.cross(
        points.unsqueeze(-2).expand_as(force), force, dim=-1
    ) / float(body_radius)
    rays = torch.cat((force, torque), dim=-1).reshape(points.shape[0], -1, 6)
    valid = contact_mask.to(torch.bool).unsqueeze(-1).expand(
        -1, -1, ray_count + 1
    ).reshape(points.shape[0], -1)
    return rays, valid


def support_vector(
    wrench_rays: torch.Tensor,
    ray_valid: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Project wrench rays onto fixed basis directions and max-reduce.

    ``wrench_rays`` is ``[B,R,6]`` and ``basis`` is ``[6,D]``.  Samples with
    no valid contacts return a zero support vector instead of ``-inf``.
    """
    if wrench_rays.ndim != 3 or wrench_rays.shape[-1] != 6:
        raise ValueError("wrench_rays must have shape [B,R,6]")
    if ray_valid.shape != wrench_rays.shape[:2]:
        raise ValueError("ray_valid must have shape [B,R]")
    if basis.ndim != 2 or basis.shape[0] != 6:
        raise ValueError("basis must have shape [6,D]")
    basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True).clamp_min(1e-8)
    projected = torch.einsum("brk,kd->brd", wrench_rays, basis)
    neg_inf = torch.finfo(projected.dtype).min
    projected = projected.masked_fill(~ray_valid.to(torch.bool).unsqueeze(-1), neg_inf)
    support = projected.max(dim=1).values
    return torch.where(ray_valid.any(dim=1, keepdim=True), support, torch.zeros_like(support))


def chord_support_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    *,
    tolerance: float = 0.1,
    variance: float = 1.0,
    min_support: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CHORD's directional support loss and exponential reward.

    The returned loss is the mean of the two squared, tolerance-adjusted
    support violations.  The reward is ``exp(-loss_per_sample)`` and includes
    one unit penalty for unintended and missed active directions, matching the
    released implementation's contact-presence terms.
    """
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("predicted and reference must both have shape [B,D]")
    if not 0 <= tolerance < 1 or variance <= 0:
        raise ValueError("tolerance must be in [0,1) and variance must be positive")
    low = torch.relu((1.0 - tolerance) * reference - predicted).square()
    high = torch.relu(predicted - (1.0 + tolerance) * reference).square()
    per_sample = (low + high).mean(dim=-1) / float(variance)
    predicted_active = predicted > float(min_support)
    reference_active = reference > float(min_support)
    unintended = (predicted_active & ~reference_active).float().mean(dim=-1)
    missed = (reference_active & ~predicted_active).float().mean(dim=-1)
    per_sample = per_sample + unintended + missed
    reward = torch.exp(-per_sample)
    return per_sample.mean(), reward
