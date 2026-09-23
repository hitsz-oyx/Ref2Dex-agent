from __future__ import annotations

import torch

from src.task.CmResidual.cmlite_policy_select import (
    ProposalConfig,
    proposal_actions,
    select_cmlite_action,
    select_cmlite_candidates,
)


class _FakeCm:
    def predict(self, q, action, object_state):
        # Candidate 4 has the only useful vertical progress and confident contact.
        delta = torch.zeros((q.shape[0], 3), dtype=q.dtype, device=q.device)
        delta[:, 2] = action[:, 2].clamp_min(0) * 0.1
        contact = torch.where(action[:, 6] > 0.05,
                              torch.full((q.shape[0],), 0.9, device=q.device),
                              torch.full((q.shape[0],), 0.1, device=q.device))
        return {"delta_world": delta, "contact_probability": contact}


def test_proposals_keep_policy_action_as_candidate_zero():
    base = torch.zeros(2, 18)
    candidates = proposal_actions(base)
    assert candidates.shape == (2, 5, 18)
    torch.testing.assert_close(candidates[:, 0], base)
    assert torch.all(candidates.abs() <= 1)


def test_selector_requires_confident_contact_and_returns_a_candidate():
    q = torch.zeros(1, 18)
    base = torch.zeros(1, 18)
    state = torch.zeros(1, 13)
    state[:, 3] = 1
    goal = torch.zeros(1, 3)
    goal[:, 2] = 0.1
    selected, scores, ids = select_cmlite_action(
        _FakeCm(), q, base, state, goal,
        actual_contact=torch.ones(1, dtype=torch.bool),
        config=ProposalConfig(contact_threshold=0.25, finger_delta=0.12),
    )
    assert selected.shape == (1, 18)
    assert scores.shape == (1, 5)
    assert ids.shape == (1,)
    assert int(ids.item()) == 4
    torch.testing.assert_close(
        selected, proposal_actions(base, ProposalConfig(finger_delta=0.12))[:, 4])


def test_selector_falls_back_to_policy_when_model_is_not_confident():
    class Uncertain(_FakeCm):
        def predict(self, q, action, object_state):
            result = super().predict(q, action, object_state)
            result["contact_probability"].fill_(0.0)
            return result

    q = torch.zeros(1, 18)
    base = torch.full((1, 18), 0.2)
    state = torch.zeros(1, 13)
    state[:, 3] = 1
    selected, _, ids = select_cmlite_action(
        Uncertain(), q, base, state, torch.zeros(1, 3),
        actual_contact=torch.ones(1, dtype=torch.bool))
    assert int(ids.item()) == 0
    torch.testing.assert_close(selected, base)


def test_candidate_selector_can_choose_an_external_expert_action():
    q = torch.zeros(2, 18)
    candidates = torch.zeros(2, 3, 18)
    candidates[:, 1, 2] = 0.5
    candidates[:, 1, 6] = 0.2
    state = torch.zeros(2, 13)
    state[:, 3] = 1
    goal = torch.zeros(2, 3)
    goal[:, 2] = 0.1
    selected, scores, ids = select_cmlite_candidates(
        _FakeCm(), q, candidates, state, goal,
        actual_contact=torch.ones(2, dtype=torch.bool),
        config=ProposalConfig(contact_threshold=0.25),
    )
    assert scores.shape == (2, 3)
    assert ids.tolist() == [1, 1]
    torch.testing.assert_close(selected, candidates[:, 1])


def test_candidate_selector_uses_candidate_zero_as_safe_fallback():
    q = torch.zeros(1, 18)
    candidates = torch.full((1, 2, 18), 0.2)
    state = torch.zeros(1, 13)
    state[:, 3] = 1
    selected, _, ids = select_cmlite_candidates(
        _FakeCm(), q, candidates, state, torch.zeros(1, 3),
        actual_contact=torch.zeros(1, dtype=torch.bool),
    )
    assert ids.tolist() == [0]
    torch.testing.assert_close(selected, candidates[:, 0])
