import torch
from src.task.CmResidual.physical_value_contract import private_initialization
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, dynamics_output


def test_real_and_predicted_geometry_use_same_fk_and_quaternion_contract():
    features = Features(torch.zeros(55), torch.ones(55), torch.zeros(8), torch.ones(8))
    states = torch.zeros(2, 16, 55)
    states[..., 42] = 1
    states[..., 38] = .2
    actions = torch.zeros(2, 16, 18)
    mask = torch.ones(2, 16, 1)
    model = OutcomeNetwork(8, 18, 53, 9283)
    output, contact, reward, terminal = dynamics_output(model, features, states, actions, mask,
                                                       torch.zeros(2, 8), torch.zeros(2, 18))
    assert output.shape == (2, 55) and contact.shape == (2, 2)
    torch.testing.assert_close(output[:, 39:43].norm(dim=-1), torch.ones(2))
    actual = features.state(output)
    equivalent = output.clone()
    equivalent[:, 39:43] *= -1
    # FK geometry depends on q and object position, not quaternion sign.
    torch.testing.assert_close(actual[:, 55:], features.state(equivalent)[:, 55:])
    assert torch.isfinite(actual).all() and reward.shape == terminal.shape == (2,)


def test_network_creation_preserves_cuda_and_cpu_streams():
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    OutcomeNetwork(8, 18, 53, 9283)
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
