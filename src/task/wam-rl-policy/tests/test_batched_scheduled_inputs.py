import numpy as np
import pytest
import torch
from wam_rl_policy import batched_scheduled_inputs as module
from wam_rl_policy.goal_scheduler import OrderedGoalScheduler


def test_partial_reset_preserves_other_history_and_batches_once(monkeypatch):
    points=np.random.RandomState(1).randn(32,3)*.01
    class Runtime:
        device=torch.device('cpu')
        geometry=[(None,None,points)]*2
        calls=[]
        def predict(self,batch):
            self.calls.append(len(batch['objects']))
            result=torch.zeros(len(batch['objects']),24,2,11,3)
            result[...,0]=.02
            return {'trajectory':result}
    def samples(objects,hands,geometries,goal_delta):
        assert objects.shape[1:]==(4,7)
        return [dict(objects=o) for o in objects]
    monkeypatch.setattr(module,'v4_online_samples',samples)
    poses=np.repeat(np.eye(4)[None],2,0);goals=poses.copy();goals[:,2,3]=.1
    runtime=Runtime()
    inputs=module.BatchedScheduledInputs(runtime,[OrderedGoalScheduler([poses[i],goals[i]],[0,2],points) for i in range(2)])
    current=np.zeros((2,300));hands=np.zeros((2,11,3))
    for tick in range(7):out=inputs.observation(current,poses,hands,np.full(2,tick/60))
    assert runtime.calls==[2]
    np.testing.assert_allclose(out['tau'][:,0,:,0],.01)
    inputs.reset([0])
    out=inputs.observation(current,poses,hands,np.array([0,7/60]))
    assert runtime.calls==[2,1]
    np.testing.assert_array_equal(out['history_valid'],[0,1])
    assert not out['tau'][0].any()
    lengths=[len(h) for h in inputs.histories]
    with pytest.raises(ValueError,match='consecutive'):inputs.observation(current,poses,hands,np.array([1/60,10/60]))
    assert [len(h) for h in inputs.histories]==lengths
