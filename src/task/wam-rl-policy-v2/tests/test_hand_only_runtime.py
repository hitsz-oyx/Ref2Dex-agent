import torch
import pytest
from wam_rl_policy_v2.wam_v4_runtime import V4Runtime


def test_hand_only_matches_shared_head_and_keeps_mask():
    class Normalizer:
        def denormalize(self,name,x):assert name=='hand';return x*2+1
    class Model:
        normalizer=Normalizer()
        def encode(self,batch):return torch.zeros(2,30,4),None,{'tokens':torch.zeros(2,8,4)}
        def hand_future(self,hand):assert hand.shape==(2,22,4);return torch.full((2,24,22,3),.25)
    runtime=object.__new__(V4Runtime);runtime.model=Model()
    valid=torch.zeros(2,2,11,dtype=torch.bool);valid[:,0]=True
    out=runtime.predict_hand({'start_hand_valid':valid})['trajectory']
    assert out.shape==(2,24,2,11,3)
    assert torch.all(out[:,:,0]==1.5) and not out[:,:,1].any()


def test_hand_only_consolidates_checks_and_rejects_nonfinite_input():
    from wam_rl_policy_v2.deferred_checks import _pending
    class Model:
        def encode(self,batch):
            checks=_pending.get()
            assert checks is not None
            checks.append(torch.tensor(False))
            return torch.zeros(1,22,4),None,{'tokens':torch.zeros(1,0,4)}
        def hand_future(self,x):return torch.zeros(1,24,22,3)
        class normalizer:
            @staticmethod
            def denormalize(name,x):return x
    runtime=object.__new__(V4Runtime);runtime.model=Model();runtime.defer_checks=True
    with pytest.raises(ValueError,match='nonfinite valid observation'):
        runtime.predict_hand({'start_hand_valid':torch.ones(1,2,11,dtype=torch.bool)})
    assert _pending.get() is None


@pytest.mark.parametrize('fail',[False,True])
def test_shared_tf32_is_scoped_and_restored_on_error(fail):
    from wam_rl_policy_v2.wam_v4_runtime import shared_tf32_math
    class Encoder(torch.nn.Module):
        def forward(self,x):
            assert torch.backends.cuda.matmul.allow_tf32
            if fail: raise ValueError('encoder failed')
            return x
    model=torch.nn.Module();model.interaction=Encoder()
    original=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        try:
            with shared_tf32_math(model,True):
                assert not torch.backends.cuda.matmul.allow_tf32
                model.interaction(torch.zeros(1))
                assert not torch.backends.cuda.matmul.allow_tf32
        except ValueError:
            assert fail
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not model.interaction._forward_hooks and not model.interaction._forward_pre_hooks
    finally: torch.backends.cuda.matmul.allow_tf32=original
