import torch
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
