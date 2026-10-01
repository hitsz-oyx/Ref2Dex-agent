#!/usr/bin/env python3
"""Run the actual native reset method on three label poses, without simulation."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2)
    output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique isolated output required')
    mpath=args.references/'run_manifest.json';m=json.loads(mpath.read_text())
    inputs={**m['input_sha256'],str(mpath.resolve()):sha(mpath)}
    inputs.update({r['generated']:r['generated_sha256'] for r in m['references']})
    reader=ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py'
    paths=[Path(__file__),reader,ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',ROOT/'src/task/CmResidual/v118_planner.py',
        ROOT/'third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py']
    inputs.update({str(p.resolve()):sha(p) for p in paths})
    # The old physics manifest pins every used hand/object asset as well.
    old=json.loads((ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-hold-plateau-curriculum-r1/run_manifest.json').read_text())
    inputs.update({p:h for p,h in old['input_sha256'].items() if '/assets/' in p})
    if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('input drift')
    begin=time.monotonic();q=[];root=[]
    for r in m['references']:
        raw=torch.load(r['generated'],map_location='cpu',weights_only=False)
        q.append(raw[r['peak_frame'],373:391].clone())
        value=torch.zeros(13);value[:7]=raw[r['peak_frame'],198:205];root.append(value)
    q=torch.stack(q);root=torch.stack(root)
    # Extract the trusted local method rather than maintaining a second clamp/
    # coupling implementation. It only writes dummy root/DOF tensors here.
    tree=ast.parse(reader.read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_set_env_state')
    code=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(code)
    namespace={};exec(compile(code,str(reader),'exec'),namespace)
    dummy=SimpleNamespace(_humanoid_root_states=torch.zeros(3,13),_dof_pos=torch.zeros(3,18),_dof_vel=torch.zeros(3,18))
    namespace['_set_env_state'](dummy,torch.arange(3),q.clone(),torch.zeros_like(q))
    actual=dummy._dof_pos
    assets=ROOT/'third_party/DExplore/dexplore/data/assets'
    bridge=DExploreCmv2GeometryBridge(hand_urdf=assets/'inspire_hand_new/inspire_hand_right.urdf',
        object_urdf=assets/'mjcf/airplane.urdf',device='cpu',seed=42)
    geometric={}
    for label,joints in (('raw_reference',q),('actual_reset',actual)):
        scene=bridge.current(joints,root)
        geometric[label]=dict(gap=torch.cdist(scene.hand_points,scene.object_points).amin((1,2)),
            per_link_contact=bridge.geometry.contact_targets(scene.link_poses,scene.object_pose).bool())
    records=[]
    for i,r in enumerate(m['references']):
        delta=actual[i]-q[i]
        records.append(dict(name=r['name'],peak_frame=r['peak_frame'],max_reset_joint_change_rad=float(delta.abs().max()),
            changed_native_joints=(delta.abs()>1e-6).nonzero().flatten().tolist(),
            raw_surface_gap_mm=float(geometric['raw_reference']['gap'][i]*1000),
            native_reset_surface_gap_mm=float(geometric['actual_reset']['gap'][i]*1000),
            raw_geometric_contact_targets=geometric['raw_reference']['per_link_contact'][i].tolist(),
            native_reset_geometric_contact_targets=geometric['actual_reset']['per_link_contact'][i].tolist()))
    if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('post-audit drift')
    output.mkdir(parents=True)
    torch.save(dict(raw_reference_q=q,actual_reset_q=actual,object_root=root,geometric=geometric),output/'poses.pt')
    result=dict(run_status='COMPLETED',status='PASS',no_physics_or_model_training=True,
        cpu_reason='three static label poses; GPU startup dominates; no neural inference',
        input_sha256=inputs,records=records,poses_sha256=sha(output/'poses.pt'),wall_seconds=time.monotonic()-begin,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        boundary='native clamp/coupling label audit and unsigned geometry only; no grip feasibility or cause of PPO failure proven')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],records=records,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
