"""Exercise the same absolute-tick executor used by native ACT behavior."""
import math

import pytest
import torch

from consequence_evaluator.action_chunk import ActionChunkExecutor


def chunk(value):
    return torch.full((2, 24, 18), float(value))


def test_overlap8_keeps_old_tail_at_actual_dispatch_boundary():
    runner = ActionChunkExecutor('overlap8')
    old = chunk(.2)
    old[:, 8] = .4
    runner.add(0, old)
    runner.add(8, chunk(.8))
    expected = (.4 + math.exp(-.01) * .8) / (1 + math.exp(-.01))
    assert torch.allclose(runner.action(8), torch.full((2, 18), expected))
    runner.add(16, chunk(0))  # Zero is a valid action, not an empty slot.
    expected = (.2 + math.exp(-.01) * .8) / sum(math.exp(-.01*i) for i in range(3))
    assert torch.allclose(runner.action(16), torch.full((2, 18), expected))


def test_every_step_uses_absolute_horizon_and_expires_after24():
    runner = ActionChunkExecutor('temporal1')
    for tick in range(26):
        plan = chunk(0)
        # Every prediction agrees on the absolute action time, across batches.
        plan[:] = torch.arange(tick, tick+24).view(1, 24, 1) / 100
        runner.add(tick, plan)
        assert torch.allclose(runner.action(tick), torch.full((2,18), tick/100))
        assert runner.active_count == min(tick+1, 24)
    runner.reset()
    with pytest.raises(RuntimeError):
        runner.action(0)
    runner.add(0, chunk(0))
    assert torch.equal(runner.action(0), torch.zeros(2,18))


@pytest.mark.parametrize('mode,period', [('open_loop24',24), ('receding8',8), ('receding1',1)])
def test_legacy_modes_preserve_hard_switch(mode, period):
    runner = ActionChunkExecutor(mode)
    for tick in range(49):
        assert runner.should_query(tick) == (tick % period == 0)
        if runner.should_query(tick):
            plan = torch.arange(24).view(1,24,1).expand(2,24,18)/100 + tick/100
            runner.add(tick, plan)
        assert torch.equal(runner.action(tick), plan[:,tick % period])


def test_executor_rejects_future_stale_duplicate_and_mutated_input():
    runner = ActionChunkExecutor('overlap8')
    plan = chunk(.2)
    runner.add(0, plan)
    plan.zero_()
    assert torch.equal(runner.action(0), chunk(.2)[:,0])
    with pytest.raises(ValueError):
        runner.add(0, chunk(.3))
    with pytest.raises(ValueError):
        runner.add(1, chunk(.3))
    with pytest.raises(RuntimeError):
        runner.action(24)
    runner.reset()
    runner.add(8, chunk(.3))
    with pytest.raises(RuntimeError):
        runner.action(7)
