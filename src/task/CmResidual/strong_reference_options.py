"""Bounded feedback corrections around rotation-anchored cup control."""
from .orientation_anchored_options import orientation_anchored_action

OPTION_NAMES = ['half_balanced', 'half_duck', 'half_mixed12', 'half_base',
                'reference_cup', 'half_train5', 'half_base_translation',
                'half_base_fingers']


def feedback_candidates(bank):
    """All corrected channels stay in the coordinate range of expert commands."""
    cup = bank[:, 1]
    values = [(cup + bank[:, index]) * .5 for index in (0, 2, 3, 4)]
    values += [cup.clone(), (cup + bank[:, 5]) * .5]
    translation = cup.clone()
    translation[:, :3] = (cup[:, :3] + bank[:, 4, :3]) * .5
    fingers = cup.clone()
    fingers[:, 6:] = (cup[:, 6:] + bank[:, 4, 6:]) * .5
    values += [translation, fingers]
    import torch
    return torch.stack(values, 1)


def executable_candidates(bank, anchor, position, offset, scale):
    feedback = feedback_candidates(bank)
    import torch
    return torch.stack([orientation_anchored_action(feedback[:, index], anchor,
                        position, offset, scale) for index in range(8)], 1)
