import numpy as np

from src.task.CmResidual.tools.probe_cm_cate_ranking import ranking_report


def test_ranking_report_detects_randomized_treatment_heterogeneity():
    score = np.tile(np.linspace(0, 1, 64), 4)
    assignment = np.tile(np.array([-1, 1] * 32), 4)
    step = np.repeat([50, 60, 70, 80], 64)
    env_id = np.tile(np.arange(64), 4)
    outcome = assignment * score * 5
    report = ranking_report(score, outcome, assignment, step, env_id,
                            seed=24, bootstraps=100)
    assert report["observed_ate_high_minus_low_mm"] > 5
    assert report["cluster_bootstrap_95ci_mm"][0] > 0
