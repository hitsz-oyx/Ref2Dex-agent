"""Balanced randomized local-control protocol; inference is pre-allocation only."""

ALLOCATION_TO_OPTION = (0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4)
ARM_PROBABILITIES = (.125, .25, .25, .1875, .1875)


def pre_forecasts(generator, inputs, bank, candidates, weights, offset, scale):
    """Return [state, member, option, (score_mm, geometric_loss, joint_presence)].

    Base uses its actual free-rotation command and law=0. All other options use
    their executable anchored command and law=1. No outcome argument exists.
    The nuisance ensemble is fixed on old fit data, including for the controls.
    """
    import torch
    from .structured_contact_consequence import current_features, normalize

    n = len(bank)
    before_height = (inputs['history'][:, -1, 38] - inputs['rest']).clamp_min(0)*1000
    options = []
    with torch.no_grad():
        for option in range(8):
            features = current_features(**inputs, bank=bank, cup_action=candidates[:, 1],
                candidate_action=candidates[:, option], weights=weights[:, option],
                law=torch.full((n,), float(option != 0), device=bank.device),
                offset=offset, scale=scale)
            normalized = normalize(features, generator.norm)
            members = []
            for model in generator.models['cm']:
                pred = model(**normalized)
                members.append(torch.stack((pred['supported_height']*10-before_height,
                    pred['loss_probability'],
                    pred['event_probability'][:, 6:8].sum(-1)), -1))
            options.append(torch.stack(members, 1))
    return torch.stack(options, 2)
