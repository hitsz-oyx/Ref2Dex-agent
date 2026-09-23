"""Check the frozen V1.46 evaluation command identity."""

from src.task.CmResidual.tools.run_v146_eval_matrix import command


def test_on_off_commands_share_strict_evaluator_and_use_distinct_runs():
    on, on_output = command("on", 104, 0, 5)
    off, off_output = command("off", 104, 0, 6)
    assert on[1] == off[1]
    assert on[on.index("--epochs") + 1] == off[off.index("--epochs") + 1] == "300"
    assert on_output != off_output
    assert "agent_v146_cmon" in str(on_output)
    assert "agent_v146_cmoff" in str(off_output)
