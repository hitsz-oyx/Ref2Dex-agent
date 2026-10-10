"""Check online projection convergence and latency against a saved300-step smoke."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from probe_reference_tracking import gpu_state,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--iterations',type=int,choices=(60,300),default=60)
    p.add_argument('--positions-only',action='store_true')
    p.add_argument('--cuda-graph',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists() or ROOT/'outputs/consequence-evaluator' not in a.output.parents:
        raise ValueError('fresh bounded output required')
    before=gpu_state(a.gpu)
    if before['used_mib']>512 or before['utilization']>10: raise RuntimeError('GPU not idle')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from consequence_evaluator.tau_projection import project_tau
    torch.set_num_threads(2)
    m=json.loads((a.smoke/'manifest.json').read_text())
    if m['status']!='COMPLETED' or m['mode']!='smoke': raise ValueError('completed engineering smoke required')
    with np.load(a.smoke/'plans.npz',allow_pickle=False) as s:
        raw=s['raw'][0];old=s['hand'][0];ids=s['envs'];old_s=float(s['projection_s'][0])
    with np.load(a.smoke/'trajectory.npz',allow_pickle=False) as s:
        hand=s['hand_keypoints'][0,ids];q=s['dof_position'][0,ids];pose=s['object_pose'][0,ids]
        changed_hand=s['hand_keypoints'][4,ids];changed_q=s['dof_position'][4,ids];changed_pose=s['object_pose'][4,ids]
    world=np.einsum('nij,ntpj->ntpi',pose[:,:3,:3],raw)+pose[:,None,None,:3,3]
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    cache={} if a.cuda_graph else None
    fit=project_tau(hand,world,q,urdf,'cuda:0',iterations=a.iterations,deadline_s=60,
                    positions_only=a.positions_only,graph_cache=cache)
    cold_s=fit['elapsed_s'];dynamic_error=None;steady_s=None;repeat_error=None
    if a.cuda_graph:
        repeat=project_tau(hand,world,q,urdf,'cuda:0',iterations=a.iterations,deadline_s=30,
            positions_only=a.positions_only,graph_cache=cache)
        steady_s=repeat['elapsed_s'];repeat_error=float(np.max(np.abs(repeat['points']-fit['points'])))
        changed_world=np.einsum('nij,ntpj->ntpi',changed_pose[:,:3,:3],raw)+changed_pose[:,None,None,:3,3]
        eager=project_tau(changed_hand,changed_world,changed_q,urdf,'cuda:0',iterations=a.iterations,deadline_s=60)
        changed=project_tau(changed_hand,changed_world,changed_q,urdf,'cuda:0',iterations=a.iterations,deadline_s=30,
            positions_only=a.positions_only,graph_cache=cache)
        dynamic_error=float(np.max(np.sqrt(np.mean((eager['points']-changed['points'])**2,(1,2,3))))*1000)
        steady_s=max(steady_s,changed['elapsed_s'])
    fresh=fit['points'][:,1:]
    rms=np.sqrt(np.mean((fresh-world)**2,(1,2,3)))*1000
    prior=np.sqrt(np.mean((old-world)**2,(1,2,3)))*1000
    delta=np.sqrt(np.mean((fresh-old)**2,(1,2,3)))*1000
    # Engineering convergence/latency gate, fixed before observing outcomes.
    latency_pass=(cold_s<=30 and steady_s<=5 and dynamic_error<=1 and repeat_error<=2e-6) if a.cuda_graph else cold_s<=5
    passed=bool(np.max(rms-prior)<=.5 and np.max(delta)<=1 and latency_pass)
    files=[a.smoke/'manifest.json',a.smoke/'plans.npz',a.smoke/'trajectory.npz',urdf,
           Path(__file__).resolve(),TASK/'src/consequence_evaluator/tau_projection.py',
           TASK/'src/consequence_evaluator/reset_kinematics.py',TASK/'src/consequence_evaluator/fixed_wrist_decoder.py',
           TASK/'src/consequence_evaluator/tau_projection_graph.py']
    a.output.mkdir(parents=True)
    result=dict(status='PASS' if passed else 'FAIL',iterations=a.iterations,positions_only=a.positions_only,
        projection_s=fit['elapsed_s'],cuda_graph=a.cuda_graph,steady_s=steady_s,
        changed_query_point_rmse_max_mm=dynamic_error,repeat_max_error_m=repeat_error,
        original_projection_s=old_s,coordinate_rmse_mm=rms.tolist(),original_rmse_mm=prior.tolist(),
        delta_to300_mm=delta.tolist(),max_degradation_mm=float(np.max(rms-prior)),
        gpu=gpu_state(a.gpu),claim='Engineering convergence/latency only; no behavior evidence')
    write(a.output/'result.json',result)
    write(a.output/'manifest.json',dict(status='COMPLETED',git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256={str(v.resolve()):sha(v) for v in files}))
    np.savez_compressed(a.output/'projection.npz',**fit)
    if a.cuda_graph:np.savez_compressed(a.output/'changed-query.npz',eager_points=eager['points'],
        captured_points=changed['points'],eager_q=eager['q'],captured_q=changed['q'],
        hand=changed_hand,world=changed_world,current_q=changed_q)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
