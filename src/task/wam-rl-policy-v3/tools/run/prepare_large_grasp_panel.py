"""Freeze v9 reference eligibility from existing recordings, never policy outcomes."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/ptv3-shared-wam-v6/src')]
from wam_rl_policy_v3.stage_evaluation import PalmSurface
from wam_rl_policy_v3.pose_consumer import pose7
from wam_rl_policy_v3.task_feedback import FeedbackObserver
from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
from wam_rl_policy_v3.measured_reward_bridge import reward_features
from wam_rl_policy_v3.teacher_quality import table_support
from ptv3_shared_wam_v6.staged_progress import StagedProgress
from scipy.spatial.transform import Rotation


def load_raw(record):
    source=Path(record['source']);receipt=json.loads((source.parents[1]/'manifest.json').read_text())
    with np.load(source/'trajectory.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/record['sequence']/'admission.npz') as z:table=z['table_pose'].copy();bounds=z['table_bounds'].copy()
    contract=receipt['observation_contract'];ids=contract['contact_body_ids'];palm_id=contract['body_names'].index('hand_base_link')
    obj=np.c_[d['obj'][:,:3,3],Rotation.from_matrix(d['obj'][:,:3,:3]).as_quat(),d['velocity']]
    reference=dict(q=d['q'][:,None],dq=d['dq'][:,None],hand=d['hand'][:,None],objects=obj[:,None],palm_pose=d['body_state'][:,None,palm_id,:7],
                   contact_forces=d['body_contact_forces'][:,None,ids],contact_force_valid=d['contact_force_valid'][:,None],object_contact_force=d['object_contact_force'][:,None],
                   tick=d['frame_tick'][:,None],action=d['applied'])
    return reference,points,table,bounds


def qualify(reference,points,table,bounds,palm,limits):
    obj=reference['objects'][:,0];hand=reference['hand'][:,0,NATIVE_TO_WAM]
    poses=np.tile(np.eye(4),(len(obj),1,1));poses[:,:3,:3]=Rotation.from_quat(obj[:,3:7]).as_matrix();poses[:,:3,3]=obj[:,:3]
    supported,gap,footprint=table_support(poses,points,table,bounds,reference['object_contact_force'][:,0])
    observer=FeedbackObserver(points);stages=StagedProgress(obj[0,2]);goal=poses[0].copy();goal[2,3]+=.08;decision=SimpleNamespace(local_goal_pose=goal,terminal_goal_pose=goal)
    q=reference['q'][:,0];mask=np.asarray(limits['has_limits']);lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper'])
    abort=(footprint&(gap<-.02))|(mask[None]&((q<lo-.1)|(q>hi+.1))).any(1)
    old_palm_min=1e6;new_palm_min=1e6
    for t in range(len(obj)):
        force=reference['contact_forces'][t,0]
        feedback=observer.update(poses[t],hand[t],obj[t,7:13],t/30,contact_forces=force,force_valid=bool(reference['contact_force_valid'][t,0]),supported=bool(supported[t]),workspace_ok=not abort[t])
        f=reward_features(poses[t],hand[t],points,decision,obj[t,7:13],force);old_palm_min=min(old_palm_min,f.palm_gap_m)
        f=palm.features(f,pose7(reference['palm_pose'][t,0]),poses[t],points);new_palm_min=min(new_palm_min,f.palm_gap_m)
        stages.update(obj[t,2],t/30,f,feedback,force,physical_abort=bool(abort[t]))
    s=stages.snapshot();return dict(eligible=s['achieved']['hold'],longest_hold_s=s['longest_hold_s'],stage=s,old_min_wrist_gap_m=old_palm_min,new_min_palm_gap_m=new_palm_min,physical_abort_frames=int(abort.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records']
    baseline=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v8-ablation-20261022-r2/manifest.json').read_text());limits=baseline['native_initial_audit']['physical_joint_limits']
    palm=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    # Every object exposed to v6/v7/v8 consumer development is excluded from unseen.
    exposed={'cylindersmall','airplane','apple','cubelarge','duck','mug'}
    geometry=json.loads((ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry/surface-manifest.json').read_text())
    text=json.dumps(geometry);eligible_objects={r['object'] for r in catalog if 'dexplore/'+r['object'] in text}
    results=[];chosen=[];seen_objects=set()
    for r in sorted(catalog,key=lambda r:r['sequence']):
        if r['engineering']!='PASS' or r['object'] in exposed or r['object'] not in eligible_objects or r['object'] in seen_objects:continue
        ref,points,table,bounds=load_raw(r);score=qualify(ref,points,table,bounds,palm,limits);results.append(dict(sequence=r['sequence'],object=r['object'],split=r['split'],**score))
        if score['eligible']:
            folder=out/r['sequence'];folder.mkdir();np.savez_compressed(folder/'reference.npz',**ref)
            chosen.append(dict(sequence=r['sequence'],object=r['object'],source=r['source'],reference=str(folder/'reference.npz'),score=score));seen_objects.add(r['object'])
        print(json.dumps(dict(sequence=r['sequence'],object=r['object'],eligible=score['eligible'],longest=score['longest_hold_s'])),flush=True)
        if len(chosen)==8:break
    report=dict(status='READY' if len(chosen)==8 else 'INSUFFICIENT',exposed_objects=sorted(exposed),selection='lexical sequence, engineering PASS, cached object geometry, successful original S2c source recording; no evaluated policy outcome used',unseen=chosen,screened=results)
    (out/'panel.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],len(chosen));return int(report['status']!='READY')


if __name__=='__main__':raise SystemExit(main())
