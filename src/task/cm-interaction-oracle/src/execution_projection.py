"""Physical forecast projection, separate from PD scaling conventions."""
import torch
import xml.etree.ElementTree as ET
from src.task.CmResidual.v118_planner import NATIVE_TO_URDF


def continuous_mask(urdf, device):
    joints = [j for j in ET.parse(urdf).getroot().findall('joint')
              if j.get('type', 'fixed') != 'fixed']
    if len(joints) != 18:
        raise ValueError('expected 18 movable native DOFs')
    mask = torch.tensor([j.get('type') == 'continuous' for j in joints], device=device)
    return mask[torch.tensor(NATIVE_TO_URDF, device=device)]


def project_execution(q, lower, upper, continuous):
    # ±pi in native_joint_limits is a PD scaling fallback, not a physical
    # absolute limit for continuous joints. Preserve the simulator branch.
    bounded = torch.maximum(torch.minimum(q, upper), lower)
    return torch.where(continuous, q, bounded)
