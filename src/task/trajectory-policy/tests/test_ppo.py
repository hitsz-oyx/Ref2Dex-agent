import torch

from trajectory_policy.ppo import advantages, task_reward


def test_variable_prefix_gae_cuts_terminal_and_bootstraps_rollout():
    reward = torch.tensor([[2.], [3.]])
    value = torch.tensor([[.5], [.8]])
    duration = torch.tensor([[8], [6]])
    done = torch.tensor([[False], [True]])
    a, returns = advantages(reward, value, done, duration, torch.tensor([999.]))
    last = 3.-.8
    first = 2.+.99**8*.8-.5+(.99*.95)**8*last
    torch.testing.assert_close(a, torch.tensor([[first], [last]]))
    torch.testing.assert_close(returns, a+value)
    a, _ = advantages(reward[:1], value[:1], done[:1], duration[:1], torch.tensor([1.]))
    torch.testing.assert_close(a, torch.tensor([[2.+.99**8-.5]]))


def test_task_reward_distinguishes_table_support_and_loss_without_references():
    reward, held, terms = task_reward(torch.tensor([.001, .001, .03]),
        torch.tensor([.05, .05, .05]), torch.tensor([False, True, False]),
        torch.tensor([False, False, True]), torch.zeros(3, dtype=torch.bool), torch.zeros(3, 18))
    assert held.tolist() == [True, False, False]
    assert reward[0]-reward[1] == 1
    assert terms['loss'][2] == -.5
    assert reward[0] > reward[1] > reward[2]


def test_lambda_one_returns_discounted_prefix_returns_independent_of_intermediate_value():
    reward = torch.tensor([[2., 2.], [3., 3.]])
    duration = torch.tensor([[8, 8], [6, 6]])
    value = torch.tensor([[.5, .5], [100., -100.]])
    bootstrap = torch.tensor([7., 7.])
    for terminal in (False, True):
        done = torch.tensor([[False, False], [terminal, terminal]])
        _, returns = advantages(reward, value, done, duration, bootstrap, lam=1.)
        last = 3. + (0. if terminal else .99**6 * 7.)
        expected = torch.tensor([[2. + .99**8 * last]*2, [last]*2])
        torch.testing.assert_close(returns, expected, atol=2e-5, rtol=1e-5)
