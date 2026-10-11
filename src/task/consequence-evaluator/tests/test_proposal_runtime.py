import numpy as np
import pytest
import torch
from consequence_evaluator.proposal_runtime import retrieve_rows, choose_generated


def test_retrieval_deduplicates_episode_without_future_labels():
    query = torch.tensor([[0., 0.]])
    bank = torch.tensor([[0., 0.], [.01, 0.], [1., 0.], [2., 0.]])
    episode = torch.tensor([0, 0, 1, 2])
    rows, distance = retrieve_rows(query, bank, episode, k=2)
    assert rows.tolist() == [[0, 2]]
    assert distance.tolist() == [[0., .5]]


def test_observed_gt_cannot_enter_generated_selector():
    scores = np.zeros((3, 11)); scores[:, 10] = 100
    with pytest.raises(ValueError, match='ten'):
        choose_generated(scores)
    np.testing.assert_array_equal(choose_generated(scores[:, :10]), [0, 0, 0])
