"""Quantify spatial-waypoint/realized-h24 endpoint mismatch on exact panels."""
import argparse
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_goal_chain import MANIFEST,load,dump,rigid_goal_delta,OrderedGoalScheduler
from ptv3_shared_wam_v6.data import V6Windows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    data=V6Windows(MANIFEST,'train');rows=[r for r in load(out/'panels.json')['rows'] if r['evaluate']];values=defaultdict(list);modes=defaultdict(Counter)
    for r in rows:
        seq,_=data._sequence(r['seq_id']);meta=data.meta['sequences'][r['seq_id']];slot=r['slot'];tick=r['tick']
        key='s%d:%s'%(meta['source_id'],r['object']) if meta['source_id']<3 else r['object']
        with np.load(data.geometry_root/data.geometry[key]['path']) as f:query=f['query']
        cur=seq['object_poses'][tick,slot].astype(float);oracle=seq['object_poses'][tick+24,slot].astype(float);planner=np.asarray(r['planner']['pose'])
        od=rigid_goal_delta(cur,oracle,query);pd=np.asarray(r['planner']['delta'])
        if np.max(np.abs(pd-rigid_goal_delta(cur,planner,query)))>1e-6:raise AssertionError('scheduler delta/frame mismatch')
        relative=np.linalg.inv(cur)@oracle
        translation=np.linalg.norm(relative[:3,3]);rotation=Rotation.from_matrix(relative[:3,:3]).magnitude();point=np.linalg.norm(od,axis=-1).mean()
        good=np.flatnonzero(seq['object_pose_valid'][:,slot]);target_frame=np.interp(r['planner']['target_reference_index'],np.arange(len(good)),good)
        gap=np.array([np.linalg.norm(pd-od,axis=-1).mean()*1000,np.linalg.norm(planner[:3,3]-oracle[:3,3])*1000,
            Rotation.from_matrix(oracle[:3,:3].T@planner[:3,:3]).magnitude(),target_frame-tick,
            translation*1000,rotation,point*1000,float(translation>.05 or rotation>.35 or point>.05)])
        for source in ('all',str(r['source_id'])):
            k=(r['panel'],source);values[k].append(gap);modes[k][r['planner']['mode']]+=1
    report={};names=['planner_oracle_point_gap_mm','planner_oracle_translation_gap_mm','planner_oracle_rotation_gap_rad',
        'planner_target_reference_frame_minus_t','oracle_translation_demand_mm','oracle_rotation_demand_rad','oracle_point_demand_mm','oracle_outside_scheduler_bound']
    for k,vs in values.items():
        v=np.array(vs);report['/'.join(k)]=dict(windows=len(v),modes=dict(modes[k]),metrics={name:dict(mean=float(v[:,i].mean()),
            p5_p25_p50_p75_p95=np.percentile(v[:,i],[5,25,50,75,95]).tolist()) for i,name in enumerate(names)})
    dump(out/'goal-contract-gaps.json',dict(status='COMPLETED',results=report,
        independently_verified='Different input semantics;planner uses unmodified deployment path module with oracle instruction reference and causal observed state;not full TaskScheduler',
        inference='Sensitivity to this mismatch does not prove scheduler-goal retraining would improve task success.'))
    for k,v in report.items():
        if k.endswith('/all'):print(k,v,flush=True)


if __name__=='__main__':main()
