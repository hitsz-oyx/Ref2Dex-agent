"""Boundary feasibility only: real robot H4 -> v6 -> sparse11 -> actual BC -> PD.

Uses a tiny random v6 with the CPU geometry test seam, not a trained WAM or
PhysX. Does not install a v6 runtime into the official PPO entrypoint.
"""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.online_runtime import online
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.contracts import SCHEMA
from ptv3_shared_wam_v6.events import EVENT_ENCODING
from ptv3_shared_wam_v6.data import collate
from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21
from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM,WAM_TO_NATIVE,expand_action
from wam_rl_policy_v3.inspire_shadow import current_state
from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
from wam_rl_policy_v3.task_runtime import TaskController
from wam_rl_policy_v3.task_spec import TaskSpec
from audit_action_reset_cpu import native_pd_function


def run():
    sys.path.insert(0,str(ROOT/'src/task/ptv3-shared-wam-v5/tests'))
    from test_v5_contract import Geometry,config,stats
    torch.set_num_threads(2);torch.manual_seed(185)
    base=ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1'
    panel=json.loads((base/'panel.json').read_text());row=panel['records'][0]
    with np.load(Path(row['source'])/'trajectory.npz') as z:data={k:z[k][:4].copy() for k in z.files if len(z[k].shape)>0 and len(z[k])>=4}
    receipt=json.loads(Path(row['collection_batch_manifest']).read_text());ids_contact=receipt['observation_contract']['contact_body_ids']
    # Cache payload filenames are declared by the manifest, not object names.
    surface_manifest=json.loads((base/'geometry/surface-manifest.json').read_text())
    surface_row=next(r for r in surface_manifest['objects'] if r['key']=='dexplore/'+row['object']+'@1')
    with np.load(base/'geometry'/surface_row['path']) as z:geo={k:z[k].copy() for k in z.files}
    controller=TaskController(TaskSpec.from_packet(row['spec']),geo['surface'])
    for t in range(4):
        result=controller.observe(data['obj'][t],data['hand'][t,NATIVE_TO_WAM],data['velocity'][t],t/30,
            contact_forces=data['body_contact_forces'][t,ids_contact],force_valid=bool(data['contact_force_valid'][t]),supported=True)
    decision=result['decision'];event=decision.next_event.wam_condition();joints=np.zeros((4,2,21,3),np.float32)
    indices=np.asarray(SPARSE11_IN_21);joints[:,0,indices]=data['hand'][:,NATIVE_TO_WAM]
    valid=np.zeros((2,21),bool);valid[0,indices]=True
    sample=online(geo,data['obj'],joints,np.zeros((4,2,778,3),np.float32),valid,np.zeros((2,778),bool),np.zeros((2,778),int),decision.local_goal_pose,event)
    cfg=dict(config(),schema=SCHEMA,event_conditioning=True,event_encoding=EVENT_ENCODING,event_dropout=.15)
    model=SharedWAMV6(stats(),cfg,Geometry(8)).eval()
    with torch.no_grad():pred=model(collate([sample]))['trajectory'].numpy()
    assert pred.shape==(1,24,2,21,3) and not pred[:,:,1].any()
    canonical_world=pred[0,:,0][:,indices]@data['obj'][-1,:3,:3].T
    tau=canonical_world[:8,WAM_TO_NATIVE].copy()
    objects=np.r_[data['obj'][-1,:3,3],Rotation.from_matrix(data['obj'][-1,:3,:3]).as_quat(),data['velocity'][-1]]
    current=current_state(data['q'][-1:],data['dq'][-1:],data['native_current_obs'][-1:,9:257],objects[None],
        controller.scheduler.spec.object_path[-1,:3,3][None],np.zeros((1,12),np.float32))[0]
    forces=data['body_contact_forces'][-1,ids_contact]
    measured=np.r_[current,data['dof_force'][-1],forces.reshape(-1),float(data['dof_force_valid'][-1]),float(data['contact_force_valid'][-1])].astype(np.float32)
    checkpoint=ROOT/'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/latest.pt'
    packet=torch.load(checkpoint,map_location='cpu',weights_only=False);actor=MeasuredInspirePolicy(with_tau=True).eval();actor.load_state_dict(packet['model'],strict=True)
    with torch.no_grad():raw=actor(torch.tensor(measured[None]),torch.tensor(result['task'][None]),torch.tensor(geo['surface'][None]),tau=torch.tensor(tau[None]),tau_valid=torch.tensor([True])).numpy()
    clipped=np.clip(raw,-1,1);expanded=expand_action(clipped);native,evidence=native_pd_function();ctrl=receipt['native_controller']
    target=native(SimpleNamespace(_pd_action_offset=torch.tensor(ctrl['offset']),_pd_action_scale=torch.tensor(ctrl['scale']),_dof_pos=torch.tensor(data['q'][-1:])),torch.tensor(expanded)).numpy()
    assert np.isfinite(target).all() and target.shape==(1,18)
    return dict(status='PASS',scope='CPU boundary-only, tiny random v6 + geometry seam, real robot H4 and actual frozen BC; no trained-WAM quality claim, no physics, no PPO optimizer',
        official_ppo_v6_connection='UNWIRED',sequence=row['sequence'],wam_schema=SCHEMA,bc_checkpoint=str(checkpoint),bc_step=packet['step'],
        input_history_frames=[0,1,2,3],sparse21_indices=indices.tolist(),native_to_canonical=NATIVE_TO_WAM.tolist(),canonical_to_native=WAM_TO_NATIVE.tolist(),
        event=event,goal_pose=decision.local_goal_pose.tolist(),output_shape=list(pred.shape),student_shape=list(tau.shape),
        invalid_left_zero=True,unknown_joints_invalid=True,dense_skin_valid_count=0,raw_action=raw.tolist(),clipped_action=clipped.tolist(),pd_target=target.tolist(),native_pd_source=evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],scope=report['scope'],output=str(args.output))))
