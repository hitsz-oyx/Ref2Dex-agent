import numpy as np

from src.task.CmResidual.tools.run_cm_effect_validation import (
    ARMS, EVAL_SEEDS, TRAIN_SEEDS, analyze, bootstrap_ci, eval_output,
    evaluation_jobs, train_command, training_jobs,
)


def test_frozen_job_matrix_and_modes():
    assert len(training_jobs()) == 12
    assert len(evaluation_jobs()) == 72
    assert len({eval_output(*job) for job in evaluation_jobs()}) == 72
    for arm, mode in (("effect_rank", "effect_rank"),
                      ("action_shuffled", "effect_action_shuffled_rank")):
        command = train_command(TRAIN_SEEDS[0], arm, 5)
        assert command[command.index("--cm-actor-weight-component") + 1] == mode
    assert "--cm-actor-weight-component" not in train_command(TRAIN_SEEDS[0], "off", 5)


def test_crossed_bootstrap_and_gates():
    rng = np.random.default_rng(1)
    assert bootstrap_ci(np.full((4, 6), 8), rng) == [0.125, 0.125]
    records = []
    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            for arm in ARMS:
                records.append({"train_seed": train_seed, "eval_seed": eval_seed,
                                "arm": arm, "successes":
                                40 if arm == "effect_rank" else 30})
    result = analyze({"completed_evaluations": records})
    assert result["conclusion"] == "SUPPORTED"
    assert result["totals"]["effect_rank"] == 40 * 24
    assert result["stable_runs_ge58"] == 0
