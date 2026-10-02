"""Coordinate and differentiable-FK smoke; actual SDK still required."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.task.CmResidual.object_frame_kinematics import UrdfKinematics,object_frame_points,rpy_matrix

def main():
    names=['joint'+str(i) for i in range(1,7)]+['index_proximal_joint','index_intermediate_joint','middle_proximal_joint','middle_intermediate_joint','pinky_proximal_joint','pinky_intermediate_joint','ring_proximal_joint','ring_intermediate_joint','thumb_proximal_yaw_joint','thumb_proximal_pitch_joint','thumb_intermediate_joint','thumb_distal_joint']
    fk=UrdfKinematics(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',names);q=torch.zeros((1,18),dtype=torch.float64,requires_grad=True);root=torch.zeros((1,13),dtype=torch.float64);root[:,6]=1
    pos,rot=fk.forward(q,root);value=pos['index_intermediate'].square().sum();gradient=torch.autograd.grad(value,q)[0];step=1e-6
    for j in [0,3,6,7,14]:
        plus=q.detach().clone();minus=q.detach().clone();plus[0,j]+=step;minus[0,j]-=step;fd=(fk.forward(plus,root)[0]['index_intermediate'].square().sum()-fk.forward(minus,root)[0]['index_intermediate'].square().sum())/(2*step);assert abs(float(fd-gradient[0,j]))<1e-7
    translated=root.clone();translated[:,:3]=torch.tensor([[.1,.2,.3]]);p2,r2=fk.forward(q.detach(),translated);assert torch.allclose(p2['index_intermediate']-pos['index_intermediate'],translated[:,:3]);assert torch.allclose(r2['index_intermediate'],rot['index_intermediate'])
    object_root=root.clone();object_root[:,3:7]=torch.tensor([[0.,0.,2**-.5,2**-.5]]);points=torch.tensor([[[1.,0.,0.]]],dtype=torch.float64);assert torch.allclose(object_frame_points(points,object_root),torch.tensor([[[0.,-1.,0.]]],dtype=torch.float64),atol=1e-12)
    assert np.allclose(rpy_matrix([0.,0.,np.pi/2])@np.array([1.,0.,0.]),[0,1,0],atol=1e-12)
    print('PASS: URDF native joint coverage, FK finite-difference gradients, world translation covariance, object inverse rotation and fixed RPY convention; SDK binding unproved')

if __name__=='__main__':main()
