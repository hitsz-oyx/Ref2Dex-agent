"""Observation-feedback translation programs around the rotation-held cup expert.

The native URDF has three zero-origin world-axis prismatic joints BEFORE any
rotation. A constant environment translation cancels from relative displacement;
the program uses that virtual wrist origin, not the rotating palm body COM.
"""
import xml.etree.ElementTree as ET

import torch

from src.task.CmResidual.executable_contact_options import hold_action

OPTION_NAMES = ('base', 'rotation_cup', 'relative_velocity', 'relative_position')
ALLOCATION_TO_OPTION = (0, 0, 1, 1, 2, 3)
GAIN = .5


def verify_translation_asset(path):
    joints = ET.parse(path).getroot().findall('joint')
    for index, (parent, child, axis) in enumerate((
            ('link0', 'link1', '1 0 0'), ('link1', 'link2', '0 1 0'),
            ('link2', 'link3', '0 0 1'))):
        joint = joints[index]
        if (joint.get('type') != 'prismatic'
                or joint.find('parent').get('link') != parent
                or joint.find('child').get('link') != child
                or joint.find('axis').get('xyz').split() != axis.split()
                or any(float(v) != 0 for key in ('xyz', 'rpy')
                       for v in joint.find('origin').get(key).split())):
            raise ValueError('unsupported native world translation chain')
    return dict(translation_dofs=[0, 1, 2], world_axes=['x', 'y', 'z'],
                frame='virtual wrist translation origin; constant reset origin cancels')


def relative_anchor(state):
    return state[..., 36:39] - state[..., :3]


def feedback_candidates(bank, state, rotation_anchor, initial_relative, offset, scale, dt):
    """Only current observation and the fragment's initial anchor are consumed.

    Cup XYZ/fingers stay the nominal feedback. All non-base programs hold wrist
    rotation. Requested metric correction is converted by the native PD scale,
    then bounded by the existing actuator command domain. Position feedback can
    agree with Cup at step zero while defining a different multi-step program.
    """
    if (bank.ndim != 3 or bank.shape[1:] != (6, 18) or state.shape != (len(bank), 49)
            or rotation_anchor.shape != (len(bank), 18)
            or initial_relative.shape != (len(bank), 3)):
        raise ValueError('relative feedback input dimensions')
    if abs(dt - 1 / 30) > 1e-8 or (scale[:6] <= 0).any():
        raise ValueError('native time or scale')
    for tensor in (bank, state, rotation_anchor, initial_relative, offset, scale):
        if not torch.isfinite(tensor).all():
            raise ValueError('nonfinite feedback input')
    cup = bank[:, 1].clone()
    cup[:, 3:6] = hold_action(rotation_anchor, state[:, :18], offset, scale)[:, 3:6]
    correction = torch.stack((torch.zeros_like(initial_relative),
        GAIN * dt * (state[:, 43:46] - state[:, 18:21]),
        GAIN * (relative_anchor(state) - initial_relative)), 1)
    programmes = cup[:, None].expand(-1, 3, -1).clone()
    programmes[:, :, :3] = (programmes[:, :, :3] + correction / scale[None, None, :3]).clamp(-1, 1)
    return torch.cat((bank[:, 4, None], programmes), 1), correction
