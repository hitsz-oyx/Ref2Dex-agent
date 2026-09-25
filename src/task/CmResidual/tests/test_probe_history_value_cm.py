import numpy as np
import torch

from src.task.CmResidual.tools.probe_history_value_cm import (
    HistoryValueCm, OBS_DIM, STEP_DIM, rank_report,
)


def test_history_value_cm_has_separate_probabilistic_heads():
    model = HistoryValueCm(history_len=5)
    history = torch.zeros(7, 5, STEP_DIM)
    mask = torch.ones(7, 5)
    output = model(history, mask)
    assert output["continuous_mean"].shape == (7, 4)
    assert output["continuous_logvar"].shape == (7, 4)
    assert output["final_contact_logit"].shape == (7,)


def test_rank_report_uses_stratified_randomized_effects():
    score = np.tile(np.linspace(0.0, 1.0, 64), 4)
    assignment = np.tile(np.array([-1, 1] * 32), 4)
    step = np.repeat(np.arange(4), 64)
    env_id = np.tile(np.arange(64), 4)
    outcome = assignment * score * 10.0
    report = rank_report(score, outcome, assignment, step, env_id,
                         bootstraps=100, seed=240999)
    assert report["status"] == "COMPLETED"
    assert report["actual_high_minus_low"] > 5.0
    assert report["cluster_95ci"][0] > 0.0
