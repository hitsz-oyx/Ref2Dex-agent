import importlib.util
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parents[1] / 'tools/run/probe_pointflow_g.py'
SPEC = importlib.util.spec_from_file_location('pointflow_g_contract', PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_pose_effect_is_quaternion_sign_invariant_and_handles_pi():
    effect = torch.zeros(3, 1, 13)
    effect[:, :, :3] = torch.tensor([.01, .02, -.03])
    effect[:, :, 3:7] = torch.tensor([[0., 0., 0., 1.], [1., 0., 0., 0.], [0., 0., 1e-9, 1.]])[:, None]
    result = MOD.pose_effect(effect)
    effect[:, :, 3:7] *= -1
    # At exactly pi either sign represents the same rotation; compare magnitude.
    flipped = MOD.pose_effect(effect)
    assert torch.allclose(result[..., :3], flipped[..., :3])
    assert torch.allclose(result[..., 3:].norm(dim=-1), flipped[..., 3:].norm(dim=-1))
    assert torch.isfinite(result).all()
    assert torch.allclose(result[1, 0, 3:].norm(), torch.tensor(torch.pi))
    assert torch.allclose(result[0, 0, 3:], torch.zeros(3))


def test_subsample_preserves_every_source_episode_without_duplicate_rows():
    source = {'source_run': torch.tensor([0, 0, 0, 0, 1, 1]),
              'episode_id': torch.tensor([1, 1, 1, 2, 1, 1])}
    idx = MOD.subset_rows(source, 2)
    assert idx.tolist() == [0, 2, 3, 4, 5]


def test_observed_hand_roots_do_not_broadcast_between_samples():
    links = torch.eye(4).expand(2, 5, 4, 4).clone()
    roots = torch.eye(4).expand(2, 4, 4).clone()
    roots[0, :3, 3] = torch.tensor([0., .02, 0.])
    roots[1, :3, 3] = torch.tensor([.03, 0., 0.])
    result = MOD.world_link_poses(links, roots)
    assert result.shape == (2, 5, 4, 4)
    assert torch.allclose(result[0, :, :3, 3], roots[0, :3, 3].expand(5, 3))
    assert torch.allclose(result[1, :, :3, 3], roots[1, :3, 3].expand(5, 3))
