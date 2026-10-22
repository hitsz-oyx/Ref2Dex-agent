"""Replay each formerly unnamed v10 unseen A1 abort, no simulation or scoring change."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.pose_consumer import pose7, reduce_to_wrist_plan
from wam_rl_policy_v3.grasp_repair import geometric_grasp_candidate,support_plane

def audit(run):
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];records=[];denominator=collections.Counter()
    for gpu in (4,5,6):
        manifest=run/f'g{gpu}-resume2/manifest.json'
        for r in json.loads(manifest.read_text())['runs']:
            if r['subset']!='unseen' or r['arm']!='A1':continue
            denominator[r['object']]+=1
            if not r['physical_abort']:continue
            folder=manifest.parent/r['id'];infos=json.loads((folder/'consumer.json').read_text());stages=json.loads((folder/'stages.json').read_text())
            with np.load(folder/'trajectory.npz') as z:
                q=z['q'][-1,0];dq=z['dq'][-1,0];palm=pose7(z['palm_pose'][-1,0]);obj=pose7(z['objects'][-1,0]);hand=z['hand'][-1,0];future=z['forecast_hand'][-2];prior_hand=z['hand'][-2,0];prior_palm=pose7(z['palm_pose'][-2,0]);prior_obj=pose7(z['objects'][-2,0]);table=z['table_pose'][-1,0];forces=z['contact_forces'][-1,0];object_z=z['objects'][:,0,2];pd=z['pd_targets_actual'][-1];valid=bool(z['tau_valid'][-2])
            source=Path(next(c['source'] for c in catalog if c['sequence']==r['sequence']))
            with np.load(source/'canonical.npz') as z:points=z['points'].copy()
            with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/r['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
            world=points@obj[:3,:3].T+obj[:3,3];gaps=np.linalg.norm(hand[[2,4,6,8,10],None]-world[None],axis=-1).min(-1)
            limits=json.loads((folder/'native-audit.json').read_text())['physical_joint_limits'];mask=np.array(limits['has_limits'],bool);lo=np.array(limits['lower']);hi=np.array(limits['upper']);margins=np.minimum(q-lo,hi-q);margin_pd=np.minimum(pd-lo,hi-pd);coast=q+dq/30
            width=float(np.ptp(points@ConvexHull(points).equations[:,:3].T,axis=0).min());diam=float(np.linalg.norm(points[:,None]-points[None],axis=-1).max());extents=np.sort(np.ptp((points-points.mean(0))@np.linalg.svd(points-points.mean(0),full_matrices=False)[2].T,axis=0))
            plan=reduce_to_wrist_plan(prior_hand,prior_palm,future,prior_obj,points) if valid else None
            hint=prior_palm if plan is None else prior_obj@plan.local_poses[-1]
            candidate,choice=geometric_grasp_candidate(palm,hand,obj,points,*support_plane(table,bounds),hint)
            info=infos[-1];reach=info.get('reachability') or {};phase=info['phase'];category={'approach':'approach_object_table_penetration','close':'close_object_table_penetration','lift':'lift_object_table_reentry','hold':'hold_object_table_reentry'}[phase]
            if r['physical_abort']!='physical_table_penetration':raise ValueError('unexpected original abort; register a named causal category')
            records.append(dict(id=r['id'],object=r['object'],sequence=r['sequence'],seed=r['seed'],category=category,abort_state_tick=r['controls'],command_tick=r['controls']-1,phase_last_command=phase,stage_abort=stages[-1],
              tip_gaps_mm=(gaps*1000).tolist(),second_tip_gap_mm=float(np.sort(gaps)[1]*1000),wrist_geometric_candidate_gap_mm=float(np.linalg.norm(candidate[:3,3]-palm[:3,3])*1000),wrist_candidate_angle_rad=float(Rotation.from_matrix(candidate[:3,:3]@palm[:3,:3].T).magnitude()),wrist_last_command_gap_mm=float(np.linalg.norm(np.array(info['wrist_target'])[:3,3]-palm[:3,3])*1000),candidate_choice=choice,
              joint_margin_rad={str(i):float(margins[i]) for i in range(6,18) if mask[i]},minimum_finger_margin_rad=float(margins[6:][mask[6:]].min()),coasting_minimum_finger_margin_rad=float(np.minimum(coast-lo,hi-coast)[6:][mask[6:]].min()),minimum_finger_pd_margin_rad=float(margin_pd[6:][mask[6:]].min()),
              physical_abort=r['physical_abort'],table_gap_mm=r['physical_safety_final']['table_gap_m']*1000,force_norms_N=np.linalg.norm(forces,axis=-1).tolist(),last_measured_contact=info['measured_contact'],recovery_active='recovery_candidate' in reach,hand_support_gap_mm=None if reach.get('actual_support_gap_m') is None else reach['actual_support_gap_m']*1000,
              object_min_width_mm=width*1000,object_diameter_mm=diam*1000,pca_extents_mm=(extents*1000).tolist(),aspect_ratio=float(extents[-1]/max(extents[0],1e-12)),object_z_last_step_mm=float(np.diff(object_z)[-1]*1000),ever_contact=r['stage_achieved']['contact'],ever_lift=r['stage_achieved']['lift'],ever_hold=r['stage_achieved']['hold'],trajectory=str(folder/'trajectory.npz')))
    if len(records)!=17 or sum(denominator.values())!=40:raise ValueError('expected 17 aborts in 40 original unseen A1')
    byobject={o:dict(aborts=sum(r['object']==o for r in records),total=n) for o,n in sorted(denominator.items())}
    return dict(n=17,denominator=40,phase_distribution=dict(collections.Counter(r['phase_last_command'] for r in records)),category_distribution=dict(collections.Counter(r['category'] for r in records)),by_object=byobject,records=sorted(records,key=lambda r:r['id']),scope='Near cause is measured object CAD/table penetration, not hand support gap. Phase is preceding command n-1; geometry is abort state n. Candidate is recomputed canonical geometry at n, not claimed to have been executed. Observational shape association is not causal attribution.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=audit(a.run.resolve());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
if __name__=='__main__':main()
