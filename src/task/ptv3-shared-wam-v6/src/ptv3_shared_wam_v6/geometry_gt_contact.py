"""GT dense-object proximity supervises predicted hand geometry, without scene flow.

Nearest indices are detached, then selected distances are differentiated. BCE
is evaluated from log probability so distant positives retain a pulling gradient.
The3mm plateau is smoothed over1mm and the training tail is never truncated.
"""
import torch
import torch.nn.functional as F
from .geometry_consistency import (
    BONES, GEOMETRY_ENCODING, skin_attachment, skin_future, gather_vertices,
)


def proximity_bce(distance, target, config):
    radius = config['geometry_contact_radius_m']
    falloff = config['geometry_contact_falloff_m']
    smooth = config['geometry_contact_smoothing_m']
    negative_log_probability = F.softplus((distance-radius)/smooth)*(smooth/falloff)
    probability = torch.exp(-negative_log_probability)
    loss = (target*negative_log_probability
            -(1-target)*torch.log(-torch.expm1(-negative_log_probability)))
    return loss, probability


def nearest_indices(queries, skin, known, chunk):
    chosen = []
    with torch.no_grad():
        for piece in queries.split(chunk, dim=2):
            distance = torch.cdist(piece.flatten(0, 1), skin.detach().flatten(0, 1))
            distance = distance.reshape(*piece.shape[:3], skin.shape[-2])
            chosen.append(distance.masked_fill(~known[:, :, None], float('inf')).argmin(-1))
    return torch.cat(chosen, dim=2)


def gt_object_contact_components(prediction, batch, config, reduction_dtype=None):
    device = prediction['trajectory'].device
    with torch.autocast(device_type=device.type, enabled=False):
        ticks = torch.tensor(config['geometry_ticks'], device=device)-1
        if 'geometry_object_future_gt' not in batch:
            raise ValueError('recorded-pose GT2048 object label required; no32-flow fallback')
        objects = batch['geometry_object_future_gt']
        if objects.shape[1:] != (1, 24, 2048, 3):
            raise ValueError('complete ordered GT object labels must be B,1,24,2048,3')
        valid = (batch['interaction_valid'] & prediction['interaction_prediction_valid']).index_select(2, ticks)[:, 0]
        valid = valid & batch['object_valid'][:, :1, None]
        queries = torch.where(valid[..., None], objects[:, 0].index_select(1, ticks).detach().float(), 0.)
        if not torch.isfinite(queries).all():
            raise ValueError('nonfinite valid GT object geometry')
        confidence = torch.where(valid, batch['contact_target'].index_select(2, ticks)[:, 0].float(), 0.)
        start_valid = batch['start_hand_valid']
        start = torch.where(start_valid[..., None], batch['start_hand'].float(), 0.)
        dense = torch.where(batch['dense_hand_valid'][..., None], batch['dense_hand'].float(), 0.)
        attachment = skin_attachment(start, dense, start_valid, batch['dense_hand_valid'])
        future_valid = batch['trajectory_valid'].index_select(1, ticks) & start_valid[:, None]
        predicted = torch.where(future_valid[..., None], prediction['trajectory'].index_select(1, ticks).float(), 0.)
        pred_skin = skin_future(start[:, None]+predicted, attachment).flatten(2, 3)
        with torch.no_grad():
            target = torch.where(future_valid[..., None], batch['trajectory_target'].index_select(1, ticks).float(), 0.)
            gt_skin = skin_future(start[:, None]+target, attachment).flatten(2, 3)
            a = torch.tensor([x for x, _ in BONES], device=device)
            b = torch.tensor([y for _, y in BONES], device=device)
            known = future_valid[..., a] & future_valid[..., b] & future_valid[..., [0, 5, 17]].all(-1)[..., None]
            known = known.gather(-1, attachment[0][:, None].expand(-1, len(ticks), -1, -1))
            known = (known & attachment[3][:, None]).flatten(2, 3)
        valid = valid & known.any(-1)[..., None]
        predicted_indices = nearest_indices(queries, pred_skin, known, config['geometry_query_chunk'])
        gt_indices = nearest_indices(queries, gt_skin, known, config['geometry_query_chunk'])
        predicted_gap = queries-gather_vertices(pred_skin, predicted_indices)
        gt_gap = queries-gather_vertices(gt_skin, gt_indices)
        distance = predicted_gap.norm(dim=-1)
        bce, probability = proximity_bce(distance, confidence, config)
        margin = (gt_gap.norm(dim=-1).clamp(max=.033)-config['geometry_clearance_tolerance_m']).clamp_min(0.)
        projection = (predicted_gap*F.normalize(gt_gap.detach(), dim=-1, eps=1e-6)).sum(-1)
        violation = torch.where(distance.detach() < margin, (margin-projection).clamp_min(0.), 0.)
        clearance = F.smooth_l1_loss(violation/config['geometry_scale_m'], torch.zeros_like(violation), reduction='none')
        negative = valid & (confidence < .01)
        result = dict(geometry_contact=(torch.where(valid, bce, 0.).sum(dtype=reduction_dtype), valid.sum(dtype=reduction_dtype)),
                      geometry_clearance=(torch.where(negative, clearance, 0.).sum(dtype=reduction_dtype), negative.sum(dtype=reduction_dtype)))
        diagnostics = {}
        if reduction_dtype is not None:
            def metric(values, mask, unit):
                return dict(sum=float(torch.where(mask, values.detach(), 0.).sum()), count=int(mask.sum()), unit=unit)
            _, gt_probability = proximity_bce(gt_gap.norm(dim=-1), confidence, config)
            diagnostics = {
                'geometry_gt/contact_mae': metric((probability-confidence).abs(), valid, '1'),
                'geometry_gt/contact_mae_near_gt025': metric((probability-confidence).abs(), valid & (confidence > .25), '1'),
                'geometry_gt/gt_tau_contact_mae': metric((gt_probability-confidence).abs(), valid, '1'),
                'geometry_gt/positive_coverage': metric(valid.float(), (batch['contact_target'].index_select(2, ticks)[:, 0] > .25)
                                                      & batch['interaction_valid'].index_select(2, ticks)[:, 0], 'fraction'),
                'geometry_gt/noncontact_violation': metric(violation*1000, negative, 'mm'),
            }
            # Keep the previous predicted-hand/predicted-object diagnostic for
            # matched DEV comparisons, separate from the new GT-object objective.
            from .geometry_consistency import geometry_components
            old_config = dict(config, geometry_object_source='predicted', geometry_encoding=GEOMETRY_ENCODING,
                              geometry_queries=128)
            _, previous_diagnostics = geometry_components(prediction, batch, old_config, reduction_dtype)
            diagnostics.update(previous_diagnostics)
        return result, diagnostics
