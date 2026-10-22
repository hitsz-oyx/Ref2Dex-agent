"""Check real saved pre-limit states without claiming counterfactual physics."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]
from wam_rl_policy_v3.source_ceiling import SourceCeilingConsumer
from wam_rl_policy_v3.pose_consumer import pose7,grasp_intent_from_retargeter
from wam_rl_policy_v3.grasp_repair import support_plane
from wam_rl_policy_v3.stage_evaluation import PalmSurface
from wam_rl_policy_v3.tau_replay_contract import pd_targets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.baseline.read_text())['rows']
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records']
    palm_surface=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf');records=[]
    for row in rows:
        if row['arm']!='A0' or row['physical_abort']!='physical_joint_limit':continue
        folder=Path(row['run'])/row['id'];audit=json.loads((folder/'native-audit.json').read_text())
        c=next(c for c in catalog if c['sequence']==row['sequence'])
        with np.load(Path(c['source'])/'canonical.npz') as z:points=z['points'].copy()
        with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/row['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
        with np.load(folder/'trajectory.npz') as z:
            t=len(z['action'])-1;q=z['q'][t,0];dq=z['dq'][t,0];palm=pose7(z['palm_pose'][t,0]);obj=pose7(z['objects'][t,0]);force=z['contact_forces'][t,0]
            offset=np.array(audit['controller']['offset']);scale=np.array(audit['controller']['scale']);goal=pose7(z['objects'][0,0]);goal[2,3]+=.08
            table=pose7(z['table_pose'][t,0]);consumer=SourceCeilingConsumer(q,palm,offset,scale,plane=support_plane(z['table_pose'][t,0],bounds),palm_vertices=palm_surface.vertices[::32],object_points=points,joint_limits=audit['physical_joint_limits'],table_geometry=(table,bounds))
            consumer.phase=json.loads((folder/'consumer.json').read_text())[-1]['phase'];consumer.close_pose=palm.copy();consumer.fingers=q[[6,8,10,12,14,15]].copy()
            intent=grasp_intent_from_retargeter(q,palm,obj,goal,np.clip(z['adapter_raw_action'][t],-1,1),offset,scale)
            # Only the fields consumed on these pre-grasp states are required;
            # do not invent stable-grasp/coupling evidence from a single frame.
            valid=bool(z['contact_force_valid'][t,0]);contact=valid and (np.linalg.norm(force,axis=-1)>.1).sum()>=2
            feedback=SimpleNamespace(force_valid=valid,contact_observed=contact,grasp_established=False,grasp_lost=False)
            command,info=consumer.step(hand=z['hand'][t,0],q=q,dq=dq,palm=palm,object_pose=obj,object_velocity=z['objects'][t,0,7:13],forces=force,feedback=feedback,near_fingers=0,fingertip_gap_m=1.,intent=intent,now=t/30)
            admitted,bad=consumer.admission.check(command,q)
            records.append(dict(id=row['id'],tick=t,phase=consumer.phase,admitted=admitted,rejected_joint_indices=bad,
                new_pd_targets=pd_targets(command,q,offset,scale).tolist(),saved_pd_targets=z['pd_targets_actual'][t].tolist(),response=info['source_ceiling']))
    good=len(records)==12 and all(r['admitted'] for r in records)
    report=dict(status='PASS' if good else 'FAIL',states=len(records),admitted=sum(r['admitted'] for r in records),
        unloading=sum(r['response']['unloading'] for r in records),records=records,
        scope='one-step control response on real pre-limit q/dq/force; no physics rollout, no claim that any saved failure is prevented; near/coupling not reconstructed')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}));return int(not good)


if __name__=='__main__':raise SystemExit(main())
