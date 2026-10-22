"""Classify v9/v10 saved failures under the frozen 15mm protocol, without new simulation."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.stage_evaluation import PalmSurface


def death(row):
    if row['final_success']:return 'success'
    if row['physical_abort']:return row['physical_abort']
    if row['final_hold']:return 'held_but_off_goal'
    if row['stage_achieved']['hold']:return 'held_then_lost'
    if row['stage_achieved']['lift']:return 'lift_without_continuous_hold'
    if row['stage_achieved']['contact']:return 'contact_without_lift'
    return 'approach_without_contact'


def classify(manifests):
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];cache={};records=[];ids=set()
    palm=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    for path in manifests:
        for r in json.loads(path.read_text())['runs']:
            if r['arm']!='A1':continue
            if r['id'] in ids:raise ValueError('duplicate trial; do not combine original and carry rows')
            ids.add(r['id']);folder=path.parent/r['id'];infos=json.loads((folder/'consumer.json').read_text())
            with np.load(folder/'trajectory.npz') as z:obj=z['objects'][:,0];hand=z['hand'][:,0];palm_pose=z['palm_pose'][:,0];q=z['q'][:,0];dq=z['dq'][:,0];pd=z['pd_targets_actual'];force=z['contact_forces'][:,0]
            if r['sequence'] not in cache:
                c=next(c for c in catalog if c['sequence']==r['sequence'])
                with np.load(Path(c['source'])/'canonical.npz') as z:points=z['points'].copy()
                hull=ConvexHull(points);width=float(np.ptp(points@hull.equations[:,:3].T,axis=0).min());radius=float(np.linalg.norm(points-points.mean(0),axis=-1).max());cache[r['sequence']]=(points,width,radius)
            points,width,radius=cache[r['sequence']];goal=np.asarray(infos[-1]['postgrasp_object_pose'],dtype=float)
            endpoint=np.eye(4);endpoint[:3,:3]=Rotation.from_quat(obj[-1,3:7]).as_matrix();endpoint[:3,3]=obj[-1,:3]
            pp=np.eye(4);pp[:3,:3]=Rotation.from_quat(palm_pose[-1,3:7]).as_matrix();pp[:3,3]=palm_pose[-1,:3]
            world=points@endpoint[:3,:3].T+endpoint[:3,3];tip_gaps=np.linalg.norm(hand[-1,[2,4,6,8,10],None]-world[None],axis=-1).min(-1)
            relative=np.linalg.inv(endpoint)@goal;error=float(np.linalg.norm(relative[:3,3]));category=death(r)
            if abs(error-r['final_goal_error_m'])>1e-8 or bool(r['final_success'])!=bool(r['final_hold'] and error<=.015):raise ValueError('published outcome disagrees with saved physical endpoint')
            audit=json.loads((folder/'native-audit.json').read_text());limits=audit['physical_joint_limits'];lower=np.asarray(limits['lower']);upper=np.asarray(limits['upper']);mask=np.asarray(limits['has_limits']);bad=np.flatnonzero(mask&((pd[-1]<lower-1e-6)|(pd[-1]>upper+1e-6))).tolist();violations=r['physical_safety_final']['joint_violation_indices']
            records.append(dict(id=r['id'],subset=r['subset'],object=r['object'],sequence=r['sequence'],category=category,controls=r['controls'],phase=infos[-1]['phase'],
                ever_contact=r['stage_achieved']['contact'],ever_lift=r['stage_achieved']['lift'],ever_hold=r['stage_achieved']['hold'],final_hold=r['final_hold'],longest_hold_s=r['longest_hold_s'],
                goal_error_mm=1000*error,goal_error_over_radius=error/radius,goal_error_over_minwidth=error/width,object_radius_m=radius,minimum_extent_m=width,
                object_relative_goal=relative.tolist(),final_object_pose=endpoint.tolist(),goal_pose=goal.tolist(),palm_relative_object=(np.linalg.inv(endpoint)@pp).tolist(),
                wrist_origin_object_gap_mm=1000*float(np.linalg.norm(pp[:3,3]-endpoint[:3,3])),actual_palm_surface_gap_mm=1000*palm.gap(pp,endpoint,points),
                fingertip_gaps_mm=(1000*tip_gaps).tolist(),force_norms_N=np.linalg.norm(force[-1],axis=-1).tolist(),
                final_table_gap_m=r['physical_safety_final']['table_gap_m'],joint_violation_indices=r['physical_safety_final']['joint_violation_indices'],final_q=q[-1].tolist(),final_dq=dq[-1].tolist(),last_PD=pd[-1].tolist(),pd_target_outside_nominal_indices=bad,aborted_joint_targets_within_nominal=all(lower[i]-1e-6<=pd[-1,i]<=upper[i]+1e-6 for i in violations),trajectory=str(folder/'trajectory.npz')))
    groups={}
    for subset in sorted({r['subset'] for r in records}):
        rows=[r for r in records if r['subset']==subset];errors=np.array([r['goal_error_mm'] for r in rows]);held=[r for r in rows if r['final_hold']];full=[r for r in rows if r['controls']==600]
        aborted=[r for r in rows if r['category']=='physical_joint_limit']
        groups[subset]=dict(joint_abort_n=len(aborted),abort_pd_all18_legal=sum(not r['pd_target_outside_nominal_indices'] for r in aborted),abort_violation_joint_targets_legal=sum(r['aborted_joint_targets_within_nominal'] for r in aborted),n=len(rows),primary_deaths=dict(collections.Counter(r['category'] for r in rows)),primary_sum=len(rows),goal_error_gt15mm=sum(r['goal_error_mm']>15 for r in rows),held_n=len(held),held_off_goal=sum(r['goal_error_mm']>15 for r in held),
            terminal_error_quantiles_mm=np.quantile(errors,[0,.5,.9,1]).tolist(),full_horizon_n=len(full),full_horizon_error_quantiles_mm=np.quantile([r['goal_error_mm'] for r in full],[0,.5,.9,1]).tolist() if full else None)
    return dict(groups=groups,records=records,scope='Offline read-only; no changed scoring. Goal error overlaps held/off-goal and approach failures; do not add it as a disjoint death. All terminal errors include early aborts; fixed-horizon survivors separately reported.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=classify(a.manifest);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':main()
