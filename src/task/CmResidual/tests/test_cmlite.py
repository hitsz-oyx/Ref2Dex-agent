import torch

from src.task.CmResidual.cmlite import (
    CmLite, compact_features, local_to_world_translation, local_translation_target,
)
from src.task.CmResidual.tools.train_cmlite import prepare


def _state(batch):
    state = torch.zeros(batch, 13)
    state[:, 6] = 1
    return state


def test_cmlite_contract_and_parameter_budget():
    model = CmLite()
    output = model(torch.randn(7, 49))
    assert output["delta_local_normalized"].shape == (7, 3)
    assert output["contact_logit"].shape == (7,)
    assert sum(p.numel() for p in model.parameters()) < 120_000


def test_compact_features_are_49d_and_quaternion_sign_invariant():
    q, action, state = torch.randn(2, 18), torch.randn(2, 18), _state(2)
    first = compact_features(q, action, state)
    state[:, 3:7] *= -1
    torch.testing.assert_close(compact_features(q, action, state), first)
    assert first.shape == (2, 49)


def test_local_world_translation_round_trip():
    state = _state(2)
    state[1, 5:7] = torch.tensor([2 ** -0.5, 2 ** -0.5])
    world = torch.tensor([[0.01, -0.02, 0.03], [0.02, 0.01, -0.04]])
    nxt = state.clone()
    nxt[:, :3] += world
    local = local_translation_target(state, nxt)
    torch.testing.assert_close(local_to_world_translation(state, local), world, atol=1e-6, rtol=1e-6)


def test_training_contact_target_is_flat():
    state = _state(3)
    features, target, contact = prepare({
        "q": torch.zeros(3, 18),
        "action": torch.zeros(3, 18),
        "object_state": state,
        "next_object_state": state.clone(),
        "hand_contact": torch.tensor([[True], [True], [False]]),
        "object_contact": torch.tensor([[True], [False], [True]]),
    })
    assert features.shape == (3, 49)
    assert target.shape == (3, 3)
    assert contact.shape == (3,)
