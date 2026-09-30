import torch
from src.task.CmResidual.physical_value_contract import (
    HoldTracker, HistoryBuffer, advance_events, canonical_quaternion, quaternion_loss,
    make_candidates, teacher_label, actor_supervision, private_initialization,
)


def test_transient_bounce_does_not_count_and_drop_only_penalized_once():
    tracker = HoldTracker(2, "cpu")
    tracker.reset(torch.arange(2), torch.zeros(2))
    for i in range(45):
        tracker.step(torch.tensor([.04 if i < 5 else 0., .04]), torch.ones(2, dtype=torch.bool))
    assert tracker.stable.tolist() == [False, True]
    reward, drop = tracker.step(torch.zeros(2), torch.ones(2, dtype=torch.bool))
    assert drop.tolist() == [False, True]
    assert reward[1] == -1
    assert tracker.drop_after_success.tolist() == [False, True]
    reward, drop = tracker.step(torch.zeros(2), torch.ones(2, dtype=torch.bool))
    assert not drop.any() and reward[1] == 0
    tracker.reset(torch.tensor([1]), torch.tensor([.2]))
    assert not tracker.stable[1] and not tracker.events[1].any()


def test_contact_loss_requires_six_steps_and_reward_events_match_predicted_update():
    tracker = HoldTracker(1, "cpu")
    tracker.reset(torch.tensor([0]), torch.zeros(1))
    for _ in range(30):
        tracker.step(torch.tensor([.04]), torch.ones(1, dtype=torch.bool))
    for i in range(6):
        before = tracker.events.clone()
        expected, reward, drop = advance_events(before, torch.tensor([.04]), torch.zeros(1, dtype=torch.bool), torch.zeros(1))
        actual_reward, actual_drop = tracker.step(torch.tensor([.04]), torch.zeros(1, dtype=torch.bool))
        torch.testing.assert_close(tracker.events, expected)
        torch.testing.assert_close(reward, actual_reward)
        assert bool(actual_drop[0]) == (i == 5)


def test_reset_history_cannot_leak_previous_episode():
    buffer = HistoryBuffer(2, "cpu")
    buffer.append(torch.ones(2, 55), torch.ones(2, 18))
    buffer.reset(torch.tensor([0]))
    buffer.append(torch.zeros(2, 55), torch.zeros(2, 18))
    assert buffer.mask[0].sum() == 1
    assert buffer.mask[1].sum() == 2
    assert not buffer.state[0].any() and not buffer.action[0].any()


def test_candidates_retain_original_and_mask_clipped_duplicates():
    mean = torch.ones(2, 18)
    candidates, valid = make_candidates(mean)
    assert candidates.shape == (2, 37, 18)
    assert valid[:, 0].all() and not valid[:, 1:19].any()
    scores = torch.ones(2, 37)
    label, active = teacher_label(mean, scores, candidates, valid)
    torch.testing.assert_close(label, mean)
    assert not active.any()
    scores[:, 19] = 2
    label, active = teacher_label(mean, scores, candidates, valid)
    assert active.all() and torch.allclose(label[:, 0], torch.full((2,), .99))


def test_supervision_has_actor_gradient_but_no_teacher_gradient():
    mean = torch.zeros(2, 18, requires_grad=True)
    label = torch.ones(2, 18, requires_grad=True)
    loss = actor_supervision(mean, torch.ones_like(mean), label, torch.tensor([True, False])).mean()
    loss.backward()
    assert mean.grad[0].abs().sum() > 0 and not mean.grad[1].any()
    assert label.grad is None


def test_quaternion_sign_and_initialization_do_not_change_random_stream():
    q = torch.randn(8, 4)
    torch.testing.assert_close(canonical_quaternion(q), canonical_quaternion(-q))
    assert quaternion_loss(q, -q).max() == 0
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    with private_initialization(9283):
        torch.nn.Linear(8, 4)
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
