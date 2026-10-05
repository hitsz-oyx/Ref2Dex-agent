"""Frozen physical-action diagnostics; no future consequence targets."""
import torch
from geometric_consequence import ridge_fit, ridge_predict
from execution_geometry import endpoint_flows


@torch.no_grad()
def intrinsic_finger_flow(p, bridge, geometry, q):
    """Finger-only endpoint differences in recipient zero-arm hand-base frame."""
    shared = q.clone()
    shared[..., :6] = q[:, :1, :6]
    flow, links = endpoint_flows(p, bridge, geometry, shared)
    # Current object frame -> world -> baseline endpoint hand-base frame.
    from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
    obj = dexplore_root_pose(p['before'][:, :13].to(q.device))
    rotation = obj[:, :3, :3].transpose(-1, -2) @ links[:, 0, 0, :3, :3]
    return torch.einsum('bknj,bji->bkni', flow-flow[:, :1], rotation)


def current_finger_support(geometry):
    distances = torch.cdist(geometry['points'], geometry['obj_points'].expand(len(geometry['points']), -1, -1))
    fingers = distances.min(-1).values.lt(.02).reshape(-1, 5, 24).any(-1)
    return fingers[:, torch.tensor([0,0,1,1,2,2,3,3,4,4,4,4], device=distances.device)]


@torch.no_grad()
def extract_stages(model, h, geometry, flow, ids, batch_size=64):
    """All recipient candidates, matching the frozen adapter exactly."""
    candidates = {name: [] for name in ('LocalFlow', 'Contact', 'Tokens', 'Fused', 'Output')}
    for arm in range(15):
        chunks = {name: [] for name in candidates}
        for chunk in ids.split(batch_size):
            batch = dict(obj_points=geometry['obj_points'].expand(len(chunk), -1, -1),
                obj_normals=geometry['obj_normals'].expand(len(chunk), -1, -1),
                hand_points=geometry['points'][chunk], hand_normals=geometry['normals'][chunk],
                hand_flow=flow[chunk, arm],
                delta_time_s=torch.full((len(chunk),), 8/30, device=h.device))
            output = model.spatial(batch)
            indices = output['edge_indices']
            valid = output['edge_valid_mask']
            expanded = batch['hand_flow'][:, None].expand(-1, indices.shape[1], -1, -1)
            selected = torch.gather(expanded, 2, indices[..., None].expand(-1, -1, -1, 3))
            local = (selected * valid[..., None]).sum(2) / valid.sum(-1, keepdim=True).clamp_min(1)
            points = batch['obj_points']
            scale = (points-points.mean(1, keepdim=True)).square().sum(-1).mean(1).sqrt()
            tokens = torch.cat((output['tokens'], output['token_anchors']/scale[:, None, None],
                output['token_normals'], torch.log(output['token_mass'].clamp_min(1e-8))[..., None]), -1)
            head = model.head(torch.cat((h[chunk], output['fused_feature'],
                torch.zeros(len(chunk), 58, device=h.device)), -1))
            values = dict(LocalFlow=local.flatten(1)/.02, Contact=output['contact_features'].flatten(1),
                Tokens=tokens.flatten(1), Fused=output['fused_feature'], Output=head)
            for name, value in values.items():
                if not torch.isfinite(value).all():raise ValueError('nonfinite frozen '+name)
                chunks[name].append(value)
        for name in candidates:candidates[name].append(torch.cat(chunks[name]))
    return {name:torch.stack(values, 1) for name, values in candidates.items()}


def decoder_features(value, norm):
    score = (value.flatten(0, 1)-norm['mean']) @ norm['projection']
    return (score-norm['score_mean']) / norm['score_scale']


def fit_decoder(value, target, fit_windows):
    """Source-only PCA/normalizer/ridge; no held-out target inspection."""
    rows = (fit_windows[:, None]*15+torch.arange(15, device=value.device)).flatten()
    matrix = value.flatten(0, 1)
    mean = matrix[rows].mean(0)
    _, _, v = torch.linalg.svd(matrix[rows]-mean, full_matrices=False)
    projection = v[:min(32, value.shape[-1])].T
    score = (matrix-mean) @ projection
    norm = dict(mean=mean, projection=projection, score_mean=score[rows].mean(0),
                score_scale=score[rows].std(0, unbiased=False).clamp_min(1e-6))
    x = decoder_features(value, norm)
    model = ridge_fit(x, target.flatten(0, 1), rows)
    prediction = ridge_predict(x, model).reshape_as(target)
    if not torch.isfinite(prediction).all():raise ValueError('nonfinite decoder')
    return prediction, dict(norm=norm, model=model)


def replay_decoder(value, state):
    return ridge_predict(decoder_features(value, state['norm']), state['model']).reshape(value.shape[0], 15, 12)
