"""Separate palm-origin predicate mismatch from measured object endpoint error."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/ptv3-shared-wam-v6/src')]
from wam_rl_policy_v3.grasp_geometry import GraspGeometry
from wam_rl_policy_v3.stage_evaluation import PalmSurface
from wam_rl_policy_v3.pose_consumer import pose7


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records']
    r=next(r for r in catalog if r['sequence']=='s1_cylindersmall_pass_1')
    with np.load(Path(r['source'])/'canonical.npz') as z:points=z['points'].copy()
    geometry=GraspGeometry.from_points(points)
    palm=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    with np.load(ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz') as z:ref={k:z[k].copy() for k in z.files}
    origin=[];surface=[]
    for obj,p in zip(ref['objects'][:,0],ref['palm_pose'][:,0]):
        pose=pose7(obj);pp=pose7(p);world=points@pose[:3,:3].T+pose[:3,3]
        origin.append(float(np.linalg.norm(world-pp[:3,3],axis=-1).min()));surface.append(palm.gap(pp,pose,points))
    rows=[]
    for mpath in sorted(a.run.glob('g*-resume2/manifest.json')):
        m=json.loads(mpath.read_text())
        for row in m['runs']:
            record=next(r for r in catalog if r['sequence']==row['sequence'])
            with np.load(Path(record['source'])/'canonical.npz') as z:g=GraspGeometry.from_points(z['points'])
            rows.append(dict(id=row['id'],geometry=g.manifest(),final_hold=row['final_hold'],error_m=row['final_goal_error_m'],within_geometry=g.arrived(row['final_goal_error_m']),within_old15mm=row['goal_within15mm'],abort=row['physical_abort']))
    result=dict(reference_origin_min_m=min(origin),reference_surface_min_m=min(surface),reference_origin_ever_within_old100mm=min(origin)<=.1,reference_surface_ever_within_geometry=min(surface)<=geometry.palm_gap_m,geometry=geometry.manifest(),
        diagnosis='Origin threshold is incompatible with this recorded grasp; actual palm geometry fixes measurement. Object endpoint error is independently measured and cannot be excused by this origin mismatch.',n=len(rows),rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
