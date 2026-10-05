"""Raw temporal flow and strict current-state-only oracle contracts."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from oracle_hand_flow import temporal_actions,current_state,exact_observable_groups


def test_chunks_retain_return_path_absent_from_endpoint():
    points=torch.zeros(2,3,120,3);points[:,1,:,0]=.02
    a=temporal_actions(points)
    assert not a['Endpoint'].any() and not a['State'].any()
    assert torch.equal(a['Chunk'][:,:360],-a['Chunk'][:,360:]) and a['Chunk'].abs().max()==1
    points[:,2,:,0]=.01;b=temporal_actions(points)
    assert torch.allclose(b['Endpoint'][:,:360],b['Chunk'][:,:360]+b['Chunk'][:,360:])


def test_current_state_ignores_future_and_actions_and_source_stats_only():
    torch.manual_seed(57)
    p=dict(before=torch.randn(40,72),history=torch.randn(40,10,139),native_q=torch.randn(40,32,18),base_action=torch.randn(40,18))
    g=torch.randn(40,720);ids=torch.arange(34)
    h,norm=current_state(p,g,ids)
    p['native_q']+=100;p['base_action']+=100;p['history'][:,:,36:]+=100
    again,norm2=current_state(p,g,ids)
    assert torch.equal(h,again)
    p['before'][34:]+=100;g[34:]+=100
    _,norm3=current_state(p,g,ids)
    for key in ('physical','current'):assert torch.equal(norm[key]['mean'],norm3[key]['mean'])
    for key in ('history','geometry'):assert torch.equal(norm[key]['projection'],norm3[key]['projection'])


def test_exact_support_does_not_fabricate_future_paired_labels():
    p=dict(before=torch.zeros(3,72),history=torch.zeros(3,10,139),hand_root=torch.zeros(3,13),
           before_fingertip_positions=torch.zeros(3,5,3),before_hand_base_pose=torch.zeros(3,7))
    p['before'][2,4]=1
    pairs,summary=exact_observable_groups(p,[0,1,2])
    assert pairs.tolist()==[[0,1]] and summary['exact_observable_pairs']==1
