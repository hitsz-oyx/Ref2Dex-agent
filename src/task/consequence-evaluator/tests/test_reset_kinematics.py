"""Reset link poses and velocities agree with independent URDF FK."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
import torch

TASK=Path(__file__).resolve().parents[1]
ROOT=TASK.parents[2]
sys.path[:0]=[str(ROOT),str(TASK/'src')]
from consequence_evaluator.reset_kinematics import ResetKinematics
from src.task.CmResidual.object_frame_kinematics import UrdfKinematics, quaternion_matrix

XML='<robot name="reset_test"><link name="root"/><link name="slide"/><link name="hinge"/><link name="tip"/><joint name="slide_q" type="prismatic"><parent link="root"/><child link="slide"/><origin xyz=".02 -.03 .04" rpy=".1 .2 -.3"/><axis xyz="1 1 0"/></joint><joint name="hinge_q" type="revolute"><parent link="slide"/><child link="hinge"/><origin xyz=".03 .01 -.02" rpy="-.2 .3 .1"/><axis xyz="0 1 1"/></joint><joint name="fixed_tip" type="fixed"><parent link="hinge"/><child link="tip"/><origin xyz=".04 -.05 .06" rpy=".4 .2 -.1"/></joint></robot>'


@pytest.fixture
def urdf(tmp_path):
    path=tmp_path/'hand.urdf'
    path.write_text(XML)
    return path


def fixture(urdf):
    xml=ET.parse(urdf).getroot()
    names=[j.get('name') for j in xml.findall('joint') if j.get('type')!='fixed']
    bodies=[b.get('name') for b in xml.findall('link')]
    gen=torch.Generator().manual_seed(17)
    q=torch.randn((3,len(names)),generator=gen)*.2
    qdot=torch.randn(q.shape,generator=gen)*.3
    root=torch.zeros((3,13));root[:,6]=1
    root[:,:3]=torch.randn((3,3),generator=gen)*.1
    return names,bodies,q,qdot,root


def test_all_reset_link_poses_match_independent_matrix_fk(urdf):
    names,bodies,q,qdot,root=fixture(urdf)
    # Include a nonidentity actor root; matching only the identity case is weak.
    root[:,3:7]=torch.tensor([.2,-.3,.1,.9]);root[:,3:7]/=root[:,3:7].norm(dim=-1,keepdim=True)
    states=ResetKinematics(urdf,names,bodies,'cpu').states(q,qdot,root)
    positions,rotations=UrdfKinematics(urdf,names).forward(q,root)
    expected_pos=torch.stack([positions[name] for name in bodies],1)
    expected_rot=torch.stack([rotations[name] for name in bodies],1)
    assert torch.allclose(states[:,:,:3],expected_pos,atol=2e-6)
    assert torch.allclose(quaternion_matrix(states[:,:,3:7]),expected_rot,atol=2e-6)


def test_reset_velocities_match_finite_difference_link_poses(urdf):
    names,bodies,q,qdot,root=fixture(urdf)
    fk=ResetKinematics(urdf,names,bodies,'cpu')
    states=fk.states(q,qdot,root)
    dt=.001
    pos_plus,rot_plus=UrdfKinematics(urdf,names).forward(q+qdot*dt,root)
    pos_minus,rot_minus=UrdfKinematics(urdf,names).forward(q-qdot*dt,root)
    vel=torch.stack([(pos_plus[n]-pos_minus[n])/(2*dt) for n in bodies],1)
    dr=torch.stack([(rot_plus[n]-rot_minus[n])/(2*dt) for n in bodies],1)
    r=quaternion_matrix(states[:,:,3:7])
    skew=dr@r.transpose(-1,-2)
    ang=torch.stack((skew[:,:,2,1],skew[:,:,0,2],skew[:,:,1,0]),-1)
    assert torch.allclose(states[:,:,7:10],vel,atol=1e-4)
    assert torch.allclose(states[:,:,10:13],ang,atol=2e-4)


def test_actor_twist_transports_to_every_reset_link(urdf):
    names,bodies,q,qdot,root=fixture(urdf)
    qdot[:]=0
    root[:,7:10]=torch.tensor([.1,.2,.3])
    root[:,10:13]=torch.tensor([.2,-.3,.4])
    states=ResetKinematics(urdf,names,bodies,'cpu').states(q,qdot,root)
    expected=root[:,None,7:10]+torch.cross(root[:,None,10:13].expand(-1,len(bodies),-1),
                                        states[:,:,:3]-root[:,None,:3],dim=-1)
    assert torch.allclose(states[:,:,7:10],expected,atol=1e-6)
    assert torch.allclose(states[:,:,10:13],root[:,None,10:13].expand(-1,len(bodies),-1))


@pytest.mark.parametrize('mismatch',['dof','body'])
def test_unknown_native_topology_fails(mismatch,urdf):
    names,bodies,*_=fixture(urdf)
    if mismatch=='dof':names[-1]='unknown_joint'
    else:bodies.append('unknown_body')
    with pytest.raises(ValueError):ResetKinematics(urdf,names,bodies,'cpu')


def test_measured_com_velocity_includes_angular_transport(urdf):
    names,bodies,q,qdot,root=fixture(urdf)
    fk=ResetKinematics(urdf,names,bodies,'cpu')
    origin=fk.states(q,qdot,root)
    fk.com_offsets=torch.tensor([[.01,.02,.03]]*len(bodies))
    measured=fk.states(q,qdot,root)
    offset=(quaternion_matrix(origin[:,:,3:7])@fk.com_offsets[None,:,:,None]).squeeze(-1)
    assert torch.allclose(measured[:,:,7:10],
           origin[:,:,7:10]+torch.cross(origin[:,:,10:13],offset,dim=-1),atol=1e-6)
    assert torch.equal(measured[:,:,:7],origin[:,:,:7])
