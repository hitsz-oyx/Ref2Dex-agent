"""Replay full measured source trace under the size-derived two-finger contract."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/ptv3-shared-wam-v6/src')]
from wam_rl_policy_v3.grasp_geometry import GraspGeometry
from wam_rl_policy_v3.stage_evaluation import PalmSurface
from wam_rl_policy_v3.pose_consumer import pose7
from wam_rl_policy_v3.task_feedback import FeedbackObserver
from wam_rl_policy_v3.measured_reward_bridge import reward_features
from wam_rl_policy_v3.measured_execution import online_support
from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
from ptv3_shared_wam_v6.staged_progress import StagedProgress


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];record=next(r for r in catalog if r['sequence']=='s1_cylindersmall_pass_1')
    with np.load(Path(record['source'])/'canonical.npz') as z:points=z['points'].copy()
    with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/record['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
    with np.load(ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz') as z:ref={k:z[k].copy() for k in z.files}
    geometry=GraspGeometry.from_points(points);observer=FeedbackObserver(points);stage=StagedProgress(ref['objects'][0,0,2],geometry=geometry)
    palm=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf');paired=[];mean=[]
    for t in range(len(ref['q'])):
        state={k:v[t] for k,v in ref.items() if len(v)==len(ref['q'])};obj=state['objects'][0];pose=pose7(obj);hand=state['hand'][0,NATIVE_TO_WAM];force=state['contact_forces'][0]
        feedback=observer.update(pose,hand,obj[7:13],t/30,contact_forces=force,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=True)
        f=reward_features(pose,hand,points,SimpleNamespace(local_goal_pose=pose,terminal_goal_pose=pose),obj[7:13],force);f=palm.features(f,pose7(state['palm_pose'][0]),pose,points)
        gap=geometry.paired_tip_gap(hand,pose,points);paired.append(gap);mean.append(f.mean_tip_gap_m)
        stage.update(obj[2],t/30,f,feedback,force,approach_tip_gap_m=gap)
    result=dict(geometry=geometry.manifest(),stage=stage.snapshot(),minimum_mean_tip_gap_m=min(mean),minimum_paired_tip_gap_m=min(paired),scope='same measured force/support/motion observer; demonstrates predicate attainability on original recording, not new policy qualification')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not stage.snapshot()['achieved']['hold']:raise AssertionError('recorded two-finger hold must remain geometrically attainable')

if __name__=='__main__':main()
