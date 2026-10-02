"""Pre-allocation contact-risk intervention protocol; no future arguments."""
ALLOCATION_TO_OPTION = (0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4)
ARM_PROBABILITIES = (.0625, .0625, .375, .375, .125)
NUISANCE_COLUMNS = ('score_mm', 'any_contact_loss', 'geometric_loss', 'joint_presence')


def pre_forecasts(generator, inputs, bank, candidates, weights, offset, scale):
    import torch
    from .support_preserving_consequence import current_features, normalize
    n = len(bank)
    before_height = (inputs['history'][:, -1, 38]-inputs['rest']).clamp_min(0)*1000
    forecasts = []
    with torch.no_grad():
        for option in range(8):
            features = current_features(**inputs, bank=bank, cup_action=candidates[:, 1],
                candidate_action=candidates[:, option], weights=weights[:, option],
                law=torch.full((n,), float(option != 0), device=bank.device), offset=offset, scale=scale)
            normalized = normalize(features, generator.norm)
            forecasts.append(torch.stack([torch.stack((pred['supported_height']*10-before_height,
                pred['contact_loss_probability'], pred['loss_probability'], pred['joint_probability']), -1)
                for pred in [m(**normalized) for m in generator.models['cm']]], 1))
    return torch.stack(forecasts, 2)


def binding_mask(unguarded, guarded):
    difference = (unguarded['predicted_risk']-unguarded['reference_risk']).mean(0)
    unsafe = (difference[:, 0] < -.02) | (difference[:, 1] > .02)
    changed = (guarded['initial_pd_targets']-unguarded['initial_pd_targets']).abs().amax(-1) > 1e-5
    return unsafe & changed, unsafe, changed


def subset_report(report, mask):
    member_keys = {'predicted_score_mm', 'predicted_risk', 'reference_score_mm', 'reference_risk'}
    row_keys = {'initial_command', 'initial_pd_targets', 'context_ood', 'candidate_ood', 'changed_reference'}
    return {key: (value[:, mask] if key in member_keys or key == 'trace' else
                  value[mask] if key in row_keys else value) for key, value in report.items()}
