"""Four source states and analytic scalar-segment cases, CPU engineering only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.rigid_transport_capacity import VISUAL_IDS,transport_fields,optimal_segments
from scripts.audit_surface_execution import independent_links,pose
from src.task.CmResidual.v118_planner import QUERY_LINKS


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);a=p.parse_args();old=a.execution_source/'qualified';torch.set_num_threads(2)
    with np.load(old/'held_rows.npz') as f:rows={k:f[k][[0,40,100,180]] for k in f.files}
    with np.load(old/'geometry.npz') as f:local=f['object_local']
    with np.load(old/'joint_predictions.npz') as f:predicted=f['action_velocity'][[0,40,100,180]]
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    fields=transport_fields(rows,predicted,local,urdf,'cpu')
    changed={k:v.copy() for k,v in rows.items()};changed['next_obj'][:,:3]+=.1;changed['next_q']+=.1
    altered=transport_fields(changed,predicted,local,urdf,'cpu')
    assert torch.equal(fields['candidates']['causal'],altered['candidates']['causal'])
    assert not torch.equal(fields['target'],altered['target'])
    base=pose(rows['hand_root']);current=independent_links(rows['q'],urdf);nxt=independent_links(predicted,urdf)
    obj=local[None]@pose(rows['current_obj'])[:,:3,:3].transpose(0,2,1)+rows['current_obj'][:,None,:3]
    maximum=0.
    for i,link in enumerate(VISUAL_IDS):
        name=QUERY_LINKS[link];delta=(base@nxt[name])@np.linalg.inv(base@current[name])
        candidate=obj@delta[:,:3,:3].transpose(0,2,1)+delta[:,None,:3,3]-obj
        err=float(np.abs(candidate-fields['candidates']['causal'][:,i+2].numpy()).max());maximum=max(maximum,err);assert err<2e-6
    anchor=torch.zeros(4,64,3,dtype=torch.float64);endpoints=torch.ones(4,1,64,3,dtype=torch.float64)
    fractions=torch.tensor([0,.25,1,1.5],dtype=torch.float64)
    result=optimal_segments(anchor,endpoints,endpoints[:,0]*fractions[:,None,None])
    assert torch.allclose(result['coefficients'][:,0],fractions.clamp(0,1),atol=1e-12,rtol=0)
    print(json.dumps(dict(engineering_smoke='PASS',cpu_reason='4states/analytic solves avoid GPU startup',
                         independent_transport_max_m=maximum,causal_future_label_isolation=True,analytic_segments=True)),flush=True)


if __name__=='__main__':main()
