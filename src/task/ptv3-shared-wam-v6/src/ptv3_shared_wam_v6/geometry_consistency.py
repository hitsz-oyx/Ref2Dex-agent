"""Differentiable contact correspondences over decoded hand/object futures.

Observed skin is attached to the observed joint21 bones. The same skinning
approximation evaluates predicted and GT trajectories: GT-relative targets
cancel proxy bias, rather than forcing joint centres onto the object skin.
No future labels are added to model inputs. This is a training loss branch,
not a new independently predicted contact head or a complete collision solver.
"""
import torch
import torch.nn.functional as F

GEOMETRY_ENCODING = 'v6.observed-skin-bone-frames.affine-object.contact-relative.v1'
BONES = tuple((0 if j == 0 else base+j-1, base+j)
              for base in (1, 5, 9, 13, 17) for j in range(4))


def validate_geometry_config(config):
    if not config.get('geometry_consistency', False):
        return
    if (config.get('geometry_encoding') != GEOMETRY_ENCODING
            or not 0 < config.get('geometry_weight', 0) <= .1
            or not 0 <= config.get('geometry_clearance_weight', -1) <= 1
            or not 1 <= config.get('geometry_warmup_updates', 0) <= 1000
            or not 2 <= config.get('geometry_queries', 0) <= 256
            or not 1 <= config.get('geometry_query_chunk', 0) <= 64
            or not .001 <= config.get('geometry_scale_m', 0) <= .05
            or not 0 <= config.get('geometry_clearance_tolerance_m', -1) <= .01
            or not .01 <= config.get('geometry_proxy_max_gt_gap_m', 0) <= .1):
        raise ValueError('explicit bounded geometry-consistency configuration required')
    ticks = config.get('geometry_ticks', [])
    if not ticks or len(ticks) > 12 or sorted(set(ticks)) != ticks or min(ticks) < 1 or max(ticks) > 24:
        raise ValueError('unique ordered geometry ticks within K24 required')


def frames(joints):
    """Bone frames with palm-normal roll reference; equivariant to rigid SE3."""
    a = torch.tensor([x for x, _ in BONES], device=joints.device)
    b = torch.tensor([y for _, y in BONES], device=joints.device)
    first, last = joints.index_select(-2, a), joints.index_select(-2, b)
    x = F.normalize(last-first, dim=-1, eps=1e-6)
    normal = torch.cross(joints[..., 5, :]-joints[..., 0, :],
                         joints[..., 17, :]-joints[..., 0, :], dim=-1)
    normal = F.normalize(normal, dim=-1, eps=1e-6)[..., None, :]
    z = F.normalize(normal-(normal*x).sum(-1, keepdim=True)*x, dim=-1, eps=1e-6)
    y = torch.cross(z, x, dim=-1)
    return first, last, torch.stack((x, y, torch.cross(x, y, dim=-1)), -1)


def skin_attachment(start, vertices, joint_valid, vertex_valid):
    """Discrete observed-only binding; it never reads predicted or GT futures."""
    with torch.no_grad():
        first, last, axes = frames(start)
        delta = last-first
        alpha = ((vertices[..., None, :]-first[..., None, :, :])*delta[..., None, :, :]).sum(-1)
        alpha = (alpha/delta.square().sum(-1)[..., None, :].clamp_min(1e-10)).clamp(0, 1)
        closest = first[..., None, :, :]+alpha[..., None]*delta[..., None, :, :]
        distance = (vertices[..., None, :]-closest).square().sum(-1)
        a = torch.tensor([x for x, _ in BONES], device=start.device)
        b = torch.tensor([y for _, y in BONES], device=start.device)
        palm = joint_valid[..., [0, 5, 17]].all(-1)
        palm &= torch.cross(start[..., 5, :]-start[..., 0, :],
                            start[..., 17, :]-start[..., 0, :], dim=-1).norm(dim=-1) > 1e-8
        known = joint_valid[..., a] & joint_valid[..., b] & palm[..., None]
        known &= delta.norm(dim=-1) > 1e-6
        distance = distance.masked_fill(~known[..., None, :], float('inf'))
        minimum, index = distance.min(-1)
        alpha = alpha.gather(-1, index[..., None]).squeeze(-1)
        gather = index[..., None].expand(*index.shape, 3)
        first = first.gather(-2, gather); last = last.gather(-2, gather)
        selected = axes.gather(-3, index[..., None, None].expand(*index.shape, 3, 3))
        offset = vertices-(first+alpha[..., None]*(last-first))
        local = (selected.transpose(-1, -2)@offset[..., None]).squeeze(-1)
        return index, alpha, local, vertex_valid & torch.isfinite(minimum)


def skin_future(joints, attachment):
    index, alpha, local, _ = attachment
    first, last, axes = frames(joints)
    # joints B,T,H,J,3; binding B,H,V.
    chosen = index[:, None].expand(joints.shape[0], joints.shape[1], *index.shape[1:])
    first = first.gather(-2, chosen[..., None].expand(*chosen.shape, 3))
    last = last.gather(-2, chosen[..., None].expand(*chosen.shape, 3))
    axes = axes.gather(-3, chosen[..., None, None].expand(*chosen.shape, 3, 3))
    return (first+alpha[:, None, ..., None]*(last-first)
            +(axes@local[:, None, ..., None]).squeeze(-1))


def affine_query_weights(points, queries):
    """Constant geometry interpolation: exactly reproduces rigid flow if full rank.

    Pseudoinverse is taken only over observed CAD points, under no_grad. Rank-
    deficient CADs get the finite minimum-norm extension; no predicted SVD.
    """
    with torch.no_grad():
        center = points.mean(-2, keepdim=True)
        scale = (points-center).square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-4)
        x = torch.cat(((points-center)/scale, torch.ones_like(points[..., :1])), -1)
        q = torch.cat(((queries-center[:, :, None])/scale[:, :, None], torch.ones_like(queries[..., :1])), -1)
        return q@torch.linalg.pinv(x, rtol=1e-5)[:, :, None]


def gather_vertices(vertices, index):
    return vertices.gather(-2, index[..., None].expand(*index.shape, 3))


def geometry_components(prediction, batch, config, reduction_dtype=None):
    validate_geometry_config(config)
    device = prediction['trajectory'].device
    # Float32 geometry under BF16 training; nearest-index searches are detached,
    # then selected distances/correspondences are recomputed with gradients.
    with torch.autocast(device_type=device.type, enabled=False):
        ticks = torch.tensor(config['geometry_ticks'], device=device)-1
        valid = (batch['interaction_valid'] & prediction['interaction_prediction_valid']).index_select(2, ticks)
        confidence = batch['contact_target'].float().index_select(2, ticks)
        confidence = torch.where(valid, confidence, 0.)
        total = confidence.shape[-1]; count = min(config['geometry_queries'], total)
        priority = count//2
        top = confidence.topk(priority, dim=-1).indices
        uniform = torch.linspace(0, total-1, count-priority, device=device).long()
        uniform = uniform.expand(*confidence.shape[:-1], len(uniform))
        query_ids = torch.cat((top, uniform), -1)
        confidence = confidence.gather(-1, query_ids); valid = valid.gather(-1, query_ids)
        parts = batch['part_target'].index_select(2, ticks).gather(-1, query_ids)
        points = batch['interaction_points'].float()[:, :, None].expand(-1, -1, len(ticks), -1, -1)
        points = points.gather(-2, query_ids[..., None].expand(*query_ids.shape, 3))
        if points.shape[1] != 1:
            raise ValueError('geometry consistency currently binds one target object')
        weights = affine_query_weights(batch['scene_points'].float(), points)
        scene_valid = batch['scene_valid'].index_select(2, ticks)
        scene_target = torch.where(scene_valid[..., None], batch['scene_target'].index_select(2, ticks).float(), 0.)
        scene_pred = torch.where(scene_valid[..., None], prediction['scene'].index_select(2, ticks).float(), 0.)
        pred_queries = points+weights@scene_pred
        gt_queries = points+weights@scene_target
        valid &= scene_valid.all(-1)[..., None] & batch['object_valid'][:, :, None, None]

        start = torch.where(batch['start_hand_valid'][..., None], batch['start_hand'].float(), 0.)
        dense = torch.where(batch['dense_hand_valid'][..., None], batch['dense_hand'].float(), 0.)
        attachment = skin_attachment(start, dense, batch['start_hand_valid'], batch['dense_hand_valid'])
        future_valid = batch['trajectory_valid'].index_select(1, ticks) & batch['start_hand_valid'][:, None]
        target = torch.where(future_valid[..., None], batch['trajectory_target'].index_select(1, ticks).float(), 0.)
        predicted = torch.where(future_valid[..., None], prediction['trajectory'].index_select(1, ticks).float(), 0.)
        pred_skin = skin_future(start[:, None]+predicted, attachment).flatten(2, 3)
        with torch.no_grad():
            gt_skin = skin_future(start[:, None]+target, attachment).flatten(2, 3)
            a = torch.tensor([x for x, _ in BONES], device=device)
            b = torch.tensor([y for _, y in BONES], device=device)
            bones_known = future_valid[..., a] & future_valid[..., b] & future_valid[..., [0, 5, 17]].all(-1)[..., None]
            vertex_known = bones_known.gather(-1, attachment[0][:, None].expand(-1, len(ticks), -1, -1))
            vertex_known &= attachment[3][:, None]
            vertex_known = vertex_known.flatten(2, 3)
            vertex_part = batch['dense_hand_part'].flatten(1, 2)
        pred_queries, gt_queries = pred_queries[:, 0], gt_queries[:, 0]
        confidence, valid, parts = confidence[:, 0], valid[:, 0], parts[:, 0]
        positive = valid & (confidence > .25); negative = valid & (confidence < .01)
        selected_gt, selected_pred, available = [], [], []
        n = pred_skin.shape[-2]
        for start_q in range(0, count, config['geometry_query_chunk']):
            end_q = min(start_q+config['geometry_query_chunk'], count)
            with torch.no_grad():
                gt_distance = torch.cdist(gt_queries[:, :, start_q:end_q].flatten(0, 1), gt_skin.flatten(0, 1))
                gt_distance = gt_distance.reshape(*gt_queries.shape[:2], end_q-start_q, n)
                allowed = vertex_known[..., None, :].expand_as(gt_distance)
                match = vertex_part[:, None, None] == parts[:, :, start_q:end_q, None]
                allowed = allowed & (~positive[:, :, start_q:end_q, None] | match)
                minimum, closest_gt = gt_distance.masked_fill(~allowed, float('inf')).min(-1)
                pred_distance = torch.cdist(pred_queries[:, :, start_q:end_q].detach().flatten(0, 1), pred_skin.detach().flatten(0, 1))
                pred_distance = pred_distance.reshape_as(gt_distance).masked_fill(~vertex_known[..., None, :], float('inf'))
                closest_pred = pred_distance.argmin(-1)
                selected_gt.append(closest_gt); selected_pred.append(closest_pred); available.append(torch.isfinite(minimum))
        closest_gt, closest_pred = torch.cat(selected_gt, -1), torch.cat(selected_pred, -1)
        available = torch.cat(available, -1)
        gt_anchor = gather_vertices(gt_skin, closest_gt)
        pred_anchor = gather_vertices(pred_skin, closest_gt)
        gt_gap = gt_queries-gt_anchor
        pred_gap = pred_queries-pred_anchor
        positive &= available & (gt_gap.norm(dim=-1) <= config['geometry_proxy_max_gt_gap_m'])
        negative &= available
        relative_error = pred_gap-gt_gap.detach()
        scale = config['geometry_scale_m']
        contact = F.smooth_l1_loss(relative_error/scale, torch.zeros_like(relative_error), reduction='none').mean(-1)
        # Signed GT reference gives a separating direction even at exact overlap.
        # Euclidean gate prevents pulling already-separated noncontact points.
        nearest_gap = pred_queries-gather_vertices(pred_skin, closest_pred)
        margin = (gt_gap.norm(dim=-1).clamp(max=.033)-config['geometry_clearance_tolerance_m']).clamp_min(0.)
        direction = F.normalize(gt_gap.detach(), dim=-1, eps=1e-6)
        projection = (nearest_gap*direction).sum(-1)
        near = nearest_gap.detach().norm(dim=-1) < margin
        violation = torch.where(near, (margin-projection).clamp_min(0.), 0.)
        clearance = F.smooth_l1_loss(violation/scale, torch.zeros_like(violation), reduction='none')
        result = dict(geometry_contact=(torch.where(positive, contact, 0.).sum(dtype=reduction_dtype), positive.sum(dtype=reduction_dtype)),
                      geometry_clearance=(torch.where(negative, clearance, 0.).sum(dtype=reduction_dtype), negative.sum(dtype=reduction_dtype)))
        def metric(value, mask, unit):
            return dict(sum=float(torch.where(mask, value.detach(), 0.).sum()), count=int(mask.sum()), unit=unit)
        diagnostics = {}
        if reduction_dtype is not None:
            diagnostics = {
                'geometry/contact_relative': metric(relative_error.norm(dim=-1)*1000, positive, 'mm'),
                'geometry/contact_proxy_gap': metric(pred_gap.norm(dim=-1)*1000, positive, 'mm'),
                'geometry/contact_gt_proxy_gap': metric(gt_gap.norm(dim=-1)*1000, positive, 'mm'),
                'geometry/noncontact_violation': metric(violation*1000, negative, 'mm'),
                'geometry/positive_coverage': metric(positive.float(), valid & (confidence > .25), 'fraction'),
            }
        return result, diagnostics
