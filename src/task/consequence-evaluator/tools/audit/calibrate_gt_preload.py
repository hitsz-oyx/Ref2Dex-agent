"""Train-launch-only PD preload statistics, no learned model or simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2]; ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.retargeter import commanded_targets, ACTIVE_FINGERS
from consequence_evaluator.contracts import is_within


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not is_within(a.output,ROOT/'outputs/consequence-evaluator'):
        p.error('fresh task-owned statistics output required')
    names=['act-temporal-open_loop24-20261009-r1','act-temporal-receding8-20261009-r1','act-temporal-overlap8-20261009-r1']
    residuals=[];records=[];global_targets=[];wrist_errors=[];wrist_comp=[]
    for name in names:
        path=ROOT/'outputs/consequence-evaluator'/name/'act.pkl'
        with path.open('rb') as stream: packet=pickle.load(stream)
        assert packet['role_names'][0]=='reactive_teacher'
        q=packet['dof_position'][:,0];target=commanded_targets(q[:-1],packet['actions'][:,0])
        height=packet['object_pose'][1:,0,2,3]-packet['object_pose'][0,0,2,3]
        supported=packet['table_footprint'][1:,0]&(np.abs(packet['support_gap'][1:,0])<=.02)
        held=(packet['surface_gap'][1:,0]<=.01)&(height>=.03)&~supported
        delta=target-q[1:];residuals.append(delta[held][:,ACTIVE_FINGERS]);global_targets.append(target[held][:,ACTIVE_FINGERS])
        next_velocity=(q[np.minimum(np.arange(1,543)+1,542)]-q[1:])*30
        next_velocity[-1]=(q[-1]-q[-2])*30
        wrist_errors.append(delta[:,:6]);wrist_comp.append(delta[:,:6]-.1*next_velocity[:,:6])
        records.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),held_samples=int(held.sum())))
    values=np.concatenate(residuals);targets=np.concatenate(global_targets)
    report=dict(schema='ref2dex.gt-preload-statistics.v1',statistics_scope='frozen_ref7_train_launches_only',
        sources=records,active_fingers=ACTIVE_FINGERS.tolist(),fit_role='reactive_teacher',
        finger_preload_median_rad=np.median(values,0).tolist(),
        finger_preload_mean_rad=values.mean(0).tolist(),
        finger_preload_p10_p90_rad=np.quantile(values,[.1,.9],axis=0).tolist(),
        global_finger_pd_target_median_rad=np.median(targets,0).tolist(),
        wrist_damping_feedforward_s=.1,
        wrist_target_rmse_before=np.sqrt(np.mean(np.concatenate(wrist_errors)**2,0)).tolist(),
        wrist_target_rmse_after_forward_velocity=np.sqrt(np.mean(np.concatenate(wrist_comp)**2,0)).tolist(),
        diagnostic_only=True,held_selection='source measured lifted near-object unsupported samples',
        input_limitations='no test packet statistics; preload estimates do not prove contact force or deployment robustness')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
