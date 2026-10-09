import numpy as np
import torch
from consequence_evaluator.hand_execution import current_frame,transform_future,split_motion,compose_motion,HandExecution


def test_common_and_relative_motion_roundtrip():
    rng=np.random.default_rng(9);current=rng.normal(size=(11,3)).astype('float32')
    future=rng.normal(size=(24,11,3)).astype('float32')
    motion=split_motion(current,future)
    recovered=compose_motion(torch.from_numpy(current),torch.from_numpy(motion)).numpy()
    np.testing.assert_allclose(recovered,future,atol=8e-7)
    common=np.zeros((24,33),np.float32);common[:,:3]=[.01,.02,.03]
    moved=compose_motion(torch.from_numpy(current),torch.from_numpy(common)).numpy()
    np.testing.assert_allclose(moved-current[None],np.broadcast_to(common[:,:3,None].transpose(0,2,1),(24,11,3)),atol=2e-7)


def test_fixed_current_object_frame_no_future_object_needed():
    obj=np.repeat(np.eye(4)[None],4,0);obj[:,:3,3]=[1,2,3]
    hand=np.ones((4,11,3))+[1,2,3]
    o,h=current_frame(obj,hand)
    np.testing.assert_array_equal(o[-1],np.eye(4));np.testing.assert_allclose(h,1)
    target=transform_future(obj[-1],np.ones((24,11,3))+[1,2,3])
    np.testing.assert_allclose(target,1)


def test_bridge_batch_and_candidate_information():
    torch.manual_seed(91);model=HandExecution(width=32).eval()
    state=torch.randn(2,1574);action=torch.zeros(2,24,18);action[1,:8,2]=.01
    state[1]=state[0]
    with torch.no_grad():q=model(state,action)
    assert q.shape==(2,24,33) and torch.isfinite(q).all()
    assert not torch.equal(q[0],q[1])
