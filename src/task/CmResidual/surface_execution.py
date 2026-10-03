"""Causal execution input qualification, not a deployable policy claim."""
import numpy as np
import torch
from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
from src.task.CmResidual.v118_planner import QUERY_LINKS

FLOW_MODES = ('stationary', 'pd_target', 'velocity_only', 'action_velocity', 'oracle')


def episode_split(motion, arm):
    generator = np.random.default_rng(4001)
    train, held = [], []
    for m in range(3):
        for a in range(4):
            ids = np.flatnonzero((motion == m) & (arm == a)); assert len(ids) == 64
            ids = generator.permutation(ids)
            train.extend(ids[:32]); held.extend(ids[32:])
    return np.sort(train), np.sort(held)


def selected_rows(initial, trace, environments):
    ticks = np.linspace(1, 200, 16, dtype=int)
    env = np.repeat(environments, len(ticks)); tick = np.tile(ticks, len(environments))
    previous_obj = torch.cat((initial['object_root'][None], trace['object_root']), 0)
    result = dict(env=env, tick=tick, q=trace['native_q'][tick-1, env].numpy(),
                  dq=trace['native_dq'][tick-1, env].numpy(), target=trace['target'][tick, env].numpy(),
                  next_q=trace['native_q'][tick, env].numpy(),
                  current_obj=trace['object_root'][tick-1, env].numpy(),
                  previous_obj=previous_obj[tick-1, env].numpy(), next_obj=trace['object_root'][tick, env].numpy(),
                  hand_root=trace['hand_root'][tick-1, env].numpy(),
                  sdk_next_positions=trace['hand_body_position'][tick, env].numpy(),
                  sdk_next_quaternions=trace['hand_body_quaternion'][tick, env].numpy(),
                  motion=initial['motion'][env].numpy(), arm=initial['arm_assignment'][env].numpy())
    assert np.all(np.diff(ticks) > 1)
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def fit_execution(rows, device):
    # Closed-form statistics; all scale estimates from train episodes only.
    gap = torch.from_numpy(rows['target']-rows['q']).to(device, dtype=torch.float64)
    velocity = torch.from_numpy(rows['dq']/30).to(device, dtype=torch.float64)
    signals = torch.stack((gap, velocity), -1)
    scale = signals.square().mean(0).sqrt().clamp_min(1e-4)
    y = torch.from_numpy(rows['next_q']-rows['q']).to(device, dtype=torch.float64)
    fitted = {}
    for mode in ('velocity_only', 'action_velocity'):
        x = signals/scale
        if mode == 'velocity_only': x = x.clone(); x[..., 0] = 0
        x = torch.cat((x, torch.ones_like(x[..., :1])), -1).transpose(0, 1)
        yy = y.T[..., None]
        gram = x.transpose(-1,-2) @ x + 1e-6*torch.eye(3,device=device,dtype=torch.float64)[None]
        coefficients = torch.linalg.solve(gram, x.transpose(-1,-2) @ yy)[..., 0]
        fitted[mode] = coefficients.cpu().numpy()
    return dict(scale=scale.cpu().numpy(), coefficients=fitted, ridge=1e-6)


def predict_execution(rows, fitted, lower, upper):
    q = rows['q'].astype(np.float64)
    signals = np.stack((rows['target']-rows['q'], rows['dq']/30), -1).astype(np.float64)/fitted['scale']
    result = dict(stationary=q.copy(), pd_target=rows['target'].astype(np.float64), oracle=rows['next_q'].astype(np.float64))
    for mode, coefficients in fitted['coefficients'].items():
        x = signals.copy()
        if mode == 'velocity_only': x[..., 0] = 0
        design = np.concatenate((x,np.ones_like(x[..., :1])), -1)
        predicted = q + (design*coefficients[None]).sum(-1)
        # Predict all18measured joints; do not assume realized mimic ratios exact.
        predicted[:, 6:] = np.clip(predicted[:,6:], lower[6:], upper[6:])
        result[mode] = predicted
    return {mode:value.astype(np.float32) for mode,value in result.items()}


def area_hand_samples(urdf, count=10135, seed=2024):
    # Reuse mesh loading only; global area sampling and barycentric interpolation
    # are explicit. No legacy per-link sampling, or mesh vertex resampling.
    import trimesh
    loader = object.__new__(_surface_geometry_class())
    triangles, link_ids = [], []
    for name, mesh, transform, scale in loader._visuals(trimesh, urdf):
        assert name in QUERY_LINKS
        vertices = np.asarray(mesh.vertices, np.float64)*scale.astype(np.float64)
        vertices = vertices @ transform[:3,:3].astype(np.float64).T + transform[:3,3]
        tri = vertices[np.asarray(mesh.faces,np.int64)]
        areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)/2
        tri = tri[areas>1e-12]
        triangles.append(tri); link_ids.append(np.full(len(tri),QUERY_LINKS.index(name),np.int64))
    triangles = np.concatenate(triangles); links = np.concatenate(link_ids)
    cross = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]); areas = np.linalg.norm(cross,axis=-1)/2
    rng = np.random.default_rng(seed); ids = rng.choice(len(triangles),count,p=areas/areas.sum())
    u = np.sqrt(rng.random(count)); v = rng.random(count)
    barycentric = np.stack((1-u,u*(1-v),u*v),-1)
    points = (triangles[ids]*barycentric[...,None]).sum(1)
    normals = cross[ids]/np.linalg.norm(cross[ids],axis=-1,keepdims=True)
    return dict(points=points.astype(np.float32), normals=normals.astype(np.float32), links=links[ids],
                triangle_vertices=triangles[ids], barycentric=barycentric, seed=seed,
                correspondence_not_asserted_identical_to_external_cache=True)


def execution_gates(hand, prior, persistence):
    full = hand['action_velocity']['episode_epe_mm']
    bridge = full <= .5*hand['stationary']['episode_epe_mm'] and full <= .75*hand['velocity_only']['episode_epe_mm'] and full <= 5.
    gates = {'execution_bridge':bridge}
    for name, reports in prior.items():
        oracle = reports['oracle']['parent_epe_mm']; causal = reports['action_velocity']['parent_epe_mm']
        gates[name+'_oracle'] = oracle <= .9*persistence and oracle <= .9*reports['stationary']['parent_epe_mm']
        gates[name+'_causal'] = causal <= 1.1*oracle and causal <= .9*persistence and causal <= .9*reports['velocity_only']['parent_epe_mm']
    promising = bridge and any(gates[name+'_causal'] and gates[name+'_oracle'] for name in prior)
    label = 'PROMISING' if promising else ('UNCLEAR' if bridge or any(gates[name+'_oracle'] for name in prior) else 'UNPROMISING')
    return gates,label
