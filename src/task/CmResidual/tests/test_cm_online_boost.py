import torch

from src.task.CmResidual.cm_online_boost import apply_wrist_z_boost


def test_contact_boost_changes_only_selected_wrist_z_and_clamps():
    action = torch.zeros(3, 18)
    action[0, 2] = .95
    action[1, 2] = -.2
    output = apply_wrist_z_boost(action, torch.tensor([True, False, True]), .1)
    assert output[0, 2] == 1
    assert output[1, 2] == action[1, 2]
    assert torch.isclose(output[2, 2], torch.tensor(.1))
    torch.testing.assert_close(output[:, :2], action[:, :2])
    torch.testing.assert_close(action[0, 2], torch.tensor(.95))
