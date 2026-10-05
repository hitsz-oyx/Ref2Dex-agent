"""Bounded surface-relative finger actions; no future physical measurements."""
import torch
from execution_geometry import endpoint_flows
from geometric_consequence import standardize_fit, normalize

RADIUS = .02


def nearest_surface(points, object_points, object_normals):
    """Nearest sampled surface; normals are proxies, never certified contacts."""
    n, arms, count, _ = points.shape
    obj = object_points.expand(n, -1, -1)
    normals = object_normals.expand(n, -1, -1)
    distance, ids = torch.cdist(points.flatten(1, 2), obj).min(-1)
    closest = torch.gather(obj, 1, ids[..., None].expand(-1, -1, 3))
    normal = torch.gather(normals, 1, ids[..., None].expand(-1, -1, 3))
    normal = torch.nn.functional.normalize(normal, dim=-1)
    signed = ((points.flatten(1, 2)-closest)*normal).sum(-1)
    return distance.reshape(n, arms, count), normal.reshape_as(points), signed.reshape(n, arms, count)


def surface_state(points, object_points, object_normals):
    d, _, signed = nearest_surface(points[:, None], object_points, object_normals)
    d = (d[:, 0]/RADIUS).reshape(-1, 5, 24)
    s = (signed[:, 0]/RADIUS).reshape_as(d).clamp(-5, 5)
    clipped = d.clamp(max=5)
    fields = (clipped.mean(-1), clipped.min(-1).values,
              torch.exp(-d.square()).mean(-1), s.mean(-1),
              (d<1).float().mean(-1), clipped.square().mean(-1).sqrt())
    return torch.stack(fields, -1).flatten(1)


def surface_action(points, object_points, object_normals):
    """Six scalars/finger, relative to arm0; all24 points retain identity."""
    d, normals, _ = nearest_surface(points, object_points, object_normals)
    flow = points-points[:, :1]
    signed = (flow*normals[:, :1]).sum(-1)/RADIUS
    tangent = (flow.square().sum(-1)/RADIUS**2-signed.square()).clamp_min(0).sqrt()
    weight = torch.exp(-(d[:, :1]/RADIUS).square())
    occupancy = torch.exp(-(d/RADIUS).square())
    depth = (1-d/RADIUS).clamp_min(0)
    gap = (d/RADIUS).clamp(max=5)
    fields = (weight*signed, weight*(-signed).clamp_min(0), weight*tangent,
              occupancy-occupancy[:, :1], depth-depth[:, :1], gap-gap[:, :1])
    grouped = [x.reshape(*x.shape[:2], 5, 24).mean(-1) for x in fields]
    value = torch.stack(grouped, -1).flatten(2).clamp(-5, 5)
    return torch.nn.functional.pad(value, (0, 2))


@torch.no_grad()
def contact_inputs(p, bridge, geometry, q):
    """Freeze candidate wrist at CURRENT q; compare predicted finger endpoints.

    Current object/root and predicted OOF/full-source finger q only. Local
    contact context deliberately excludes uncertain future wrist/object motion.
    """
    wrist = p['history'][:, -1, :6].to(q.device)
    causal = {key:p[key].to(q.device) for key in ('history','hand_root','before')}
    state, action = [], []
    ids = torch.arange(len(q), device=q.device)
    for chunk in ids.split(64):
        local = {key:value[chunk] for key,value in causal.items()}
        shared = q[chunk].clone()
        shared[..., :6] = wrist[chunk, None]
        flow, _ = endpoint_flows(local, bridge, {'points':geometry['points'][chunk]}, shared)
        endpoints = geometry['points'][chunk, None]+flow
        obj, normals = geometry['obj_points'], geometry['obj_normals']
        state.append(torch.cat((surface_state(geometry['points'][chunk], obj, normals),
                                surface_state(endpoints[:, 0], obj, normals)), -1))
        action.append(surface_action(endpoints, obj, normals))
    return torch.cat(state), torch.cat(action)


def action_normalize_fit(action, train):
    matrix = action.flatten(0, 1)
    ids = (train[:, None]*15+torch.arange(15, device=action.device)).flatten()
    return standardize_fit(matrix, ids, floor=.1)


def bounded_design(common, action, norm):
    return torch.cat((common.clamp(-8, 8), normalize(action, norm).clamp(-5, 5)), -1)
