"""Control-contract smoke: causal five ticks, early latch, feasible projection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.rotation_retention_feedback import RotationRetentionFeedback

def main():
    group=torch.arange(4);ctrl=RotationRetentionFeedback(group,torch.zeros(4),torch.full((4,),2),torch.full((4,),4),torch.full((18,),.1))
    base=torch.arange(72,dtype=torch.float32).reshape(4,18)/50;root=torch.zeros(4,13);root[:,2]=.04;q=torch.zeros(4,18);last=None
    for tick in range(11):
        q[:,3:6]=tick*.03;goal=ctrl.target(base,q,q,root,torch.full((4,),.03),torch.full((4,),tick),q[0],q[0])
        assert torch.equal(goal[:,:3],base[:,:3]) and torch.equal(goal[:,6:],base[:,6:]) and torch.equal(goal[:2],base[:2])
        if tick<4:assert torch.equal(goal[2],base[2])
        if tick<6:assert torch.equal(goal[3],base[3])
        if tick>=6:assert (torch.abs(goal[2:,3:6]-q[2:,3:6])<=.1000001).all()
        last=goal
    assert ctrl.rotation_anchor_tick.tolist()==[-1,-1,4,6] and ctrl.event_tick.tolist()==[6]*4
    assert torch.allclose(ctrl.rotation_anchor[2],torch.full((3,),.12)) and torch.allclose(ctrl.rotation_anchor[3],torch.full((3,),.18))
    assert torch.allclose(last[2:,3:6],torch.full((2,3),.2))
    print('PASS: preserve same-state XYZ/fingers, five causal acquisition ticks, fixed early/current anchors, declared native PD projection')

if __name__=='__main__':main()
