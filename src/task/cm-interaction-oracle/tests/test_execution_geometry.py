"""Decision-time execution inputs and FK frame/identity contracts."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from execution_geometry import execution_inputs, fit_execution, endpoint_flows
from geometric_consequence import NominalSurfaceActions


def test_execution_inputs_do_not_read_future_labels():
    torch.manual_seed(30)
    p=dict(history=torch.randn(8,10,139),native_q=torch.randn(8,32,18))
    h=torch.randn(8,5);g=dict(targets=torch.randn(8,15,18),joint=torch.randn(8,15,18))
    ids=torch.arange(6)
    state,action,norm=execution_inputs(p,h,g,ids)
    p['native_q']+=100
    state2,action2,_=execution_inputs(p,h,g,ids)
    assert torch.equal(state,state2) and torch.equal(action,action2)
    p['history'][6:]+=100
    _,_,norm2=execution_inputs(p,h,g,ids)
    assert torch.equal(norm['mean'],norm2['mean']) and torch.equal(norm['scale'],norm2['scale'])


def test_execution_fit_ignores_heldout_targets_and_state_has_no_contrasts():
    torch.manual_seed(31)
    state=torch.randn(12,8);action=torch.randn(12,15,18)
    target=torch.randn(12,18);arms=torch.arange(12)%15;ids=torch.arange(8)
    a,pred=fit_execution(state,action,target,arms,ids,True)
    target[8:]+=100
    b,again=fit_execution(state,action,target,arms,ids,True)
    assert torch.equal(a['weight'],b['weight']) and torch.equal(pred,again)
    _,blank=fit_execution(state,action,target,arms,ids,False)
    assert (blank-blank[:,:1]).abs().max()==0


def test_endpoint_flow_current_object_frame_and_no_future_pose():
    # Tiny FK smoke on CPU: exact prismatic translation and correspondence.
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf','cpu',seed=32)
    q=torch.zeros(1,1,18);q[:,:,6:]=.2
    root=torch.zeros(1,13);root[:,6]=1
    before=root.clone();before[:,3:7]=torch.tensor([0.,0.,.70710678,.70710678])
    p=dict(hand_root=root,before=before)
    g=dict(points=torch.zeros(1,120,3))
    first,_=endpoint_flows(p,bridge,g,q)
    moved=q.clone();moved[:,:,0]+=.01
    second,_=endpoint_flows(p,bridge,g,moved)
    expected=torch.tensor([0.,-.01,0.]).expand_as(first)
    assert torch.allclose(second-first,expected,atol=1e-6)
    p['trajectory']=torch.randn(1,32,72)*100
    identical,_=endpoint_flows(p,bridge,g,q)
    assert torch.equal(first,identical)
