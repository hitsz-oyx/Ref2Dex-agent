"""CPU assertions at the real clock adapter and live runtime input seam."""
import numpy as np
import pytest

from ptv3_shared_wam_v6 import paths  # install the frozen shared-v5 dependency
from ptv3_shared_wam_v6.executor_bridge import world_tau_to_executor
from ptv3_shared_wam_v6.tau_contract import (
    first_loss_evidence, future_times, gt_reference_tau, reference_coverage,
)


def test_first_row_is_future_and_interpolation_origin_is_separate():
    # Nonlinear reference catches first-row/current-frame confusion and pooling.
    ref = np.broadcast_to(np.arange(40)[:, None, None]**2, (40, 11, 3)).copy()
    tau, valid = gt_reference_tau(ref, 3)
    assert valid
    np.testing.assert_array_equal(tau[:, 0, 0], np.arange(4, 12)**2 - 9)
    assert tau[0, 0, 0] == 7  # first future, never tau(0)=0
    world = np.broadcast_to((np.arange(1,25)**2)[:,None,None], (24,11,3)).copy()
    inspire = world_tau_to_executor(world, native_hz=30)
    shadow = world_tau_to_executor(world, native_hz=60)
    np.testing.assert_array_equal(inspire, world[:8])
    assert shadow[0,0,0] == .5  # midpoint of inserted zero and first future
    assert shadow[1,0,0] == 1
    assert shadow[2,0,0] == 2.5  # future1/future2 midpoint, not a time mean
    assert shadow[-1,0,0] == 16
    np.testing.assert_array_equal(world[:,0,0], np.arange(1,25)**2)  # no mutation


def test_eight_steps_are_not_the_same_duration_across_hands():
    assert future_times(30)[-1] == pytest.approx(8/30)
    assert future_times(60)[-1] == pytest.approx(8/60)
    assert future_times(30)[-1] == 2*future_times(60)[-1]
    assert len(future_times(30)) == len(future_times(60)) == 8


def test_gt_reference_clock_startup_and_tail_have_exact_boundaries():
    ref = np.zeros((81,11,3))
    valid = [gt_reference_tau(ref,t)[1] for t in range(81)]
    assert np.flatnonzero(valid).tolist() == list(range(3,73))
    assert sum(valid) == 70
    assert not gt_reference_tau(ref,64,encoder_available=False)[1]
    np.testing.assert_array_equal(gt_reference_tau(ref,73)[0],0)
    full = reference_coverage(205,166)
    assert full['valid_tau_controls'] == 194 and full['last_valid_tick'] == 196
    assert full['post_endpoint_states'] == 38
    assert full['valid_post_endpoint_elapsed_s'] == 1
    assert full['endpoint_plus_1s_tau_covered']
    assert not reference_coverage(204,166)['endpoint_plus_1s_tau_covered']


@pytest.mark.parametrize('last_held_tick', [64,66])
def test_first_loss_precedes_gt_coverage_expiry(last_held_tick):
    tick=last_held_tick+1
    held = np.zeros(127, bool);held[49:tick] = True
    valid = np.zeros(127,bool);valid[3:73] = True
    row = first_loss_evidence(held,valid,81)
    assert row['first_loss_tick'] == tick
    assert row['last_held_tick_before_first_loss'] == last_held_tick
    assert row['loss_while_gt_valid']
    assert row['coverage_expiry_explains_first_loss'] is False
    valid[tick] = False  # Actual masking must override purely index-based availability.
    assert not first_loss_evidence(held,valid,81)['loss_while_gt_valid']


def test_no_loss_and_expired_gt_are_not_early_channel_failure():
    assert first_loss_evidence([False,False],[False,False],81)['first_loss_tick'] is None
    held=np.zeros(90,bool);held[50:74]=True
    row=first_loss_evidence(held,np.zeros(90,bool),81)
    assert row['first_loss_tick']==74 and not row['reference_future_available']
    assert row['coverage_expiry_explains_first_loss'] is None


def test_live_wam_reconditions_while_gt_is_reference_indexed():
    # Use production predict/collate/online and capture the actual model inputs.
    # A fixed output spy checks conditioning, not whether a random model uses it.
    from test_v6_executor_wiring import runtime, inputs
    import torch
    rt=runtime();seen=[]
    def capture(batch):
        seen.append({k:v.clone() for k,v in batch.items()})
        return torch.zeros((1,24,2,21,3))
    rt._trajectory=capture
    objects,hands,goals,events,codes=inputs()
    args=[objects[:1],hands[:1],goals[:1],events[:1],codes[:1]]
    rt.predict(*args)
    args[1]=args[1].copy();args[1][0,-1,0,0]+=.01
    args[2]=args[2].copy();args[2][0,0,3]+=.02
    args[3]=[dict(events[0],event_type=1,event_delta_steps=6)]
    rt.predict(*args)
    assert rt.calls == 2
    assert not torch.equal(seen[0]['start_hand'],seen[1]['start_hand'])
    assert not torch.equal(seen[0]['goal_object_delta'],seen[1]['goal_object_delta'])
    assert seen[1]['event_type'].item() == 1 and seen[1]['event_delta_steps'].item() == 6
    ref=hands[0].copy()
    ref=np.concatenate([ref,np.repeat(ref[-1:],20,axis=0)])
    before=gt_reference_tau(ref,3)[0]
    # The same changed live history/goal/event have no place in GT's definition.
    np.testing.assert_array_equal(before,gt_reference_tau(ref,3)[0])
    assert np.any(args[1] != hands[:1])
