"""Fit a two-coefficient one-step PD inverse from train-only approach frames."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.retargeter import commanded_targets,ACTIVE_FINGERS
from consequence_evaluator.contracts import is_within


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not is_within(args.output,ROOT/'outputs/consequence-evaluator'):
        parser.error('fresh task-owned output required')
    independent=np.concatenate((np.arange(6),ACTIVE_FINGERS))
    names=['act-temporal-open_loop24-20261009-r1','act-temporal-receding8-20261009-r1','act-temporal-overlap8-20261009-r1']
    xs=[];ys=[];records=[]
    for name in names:
        path=ROOT/'outputs/consequence-evaluator'/name/'act.pkl'
        with path.open('rb') as stream:p=pickle.load(stream)
        assert p['role_names'][0]=='reactive_teacher'
        q=p['dof_position'][:,0];v=p['dof_velocity'][:-1,0]
        u=commanded_targets(q[:-1],p['actions'][:,0])
        xs.append(np.stack((q[1:]-q[:-1],v),-1)[:40,independent])
        ys.append((u-q[:-1])[:40,independent])
        records.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ticks=[0,39]))
    x=np.concatenate(xs).astype('float64');y=np.concatenate(ys).astype('float64')
    coefficient=np.stack([np.linalg.lstsq(x[:,j],y[:,j],rcond=None)[0] for j in range(12)])
    error=np.sum(x*coefficient[None],-1)-y
    assert np.isfinite(coefficient).all() and np.all((coefficient[:,0]>0)&(coefficient[:,0]<10))
    report=dict(schema='ref2dex.pd-step-inverse.v1',engineering_only=True,
        statistics_scope='frozen_ref7_train_launches_only',sources=records,independent_dofs=independent.tolist(),
        coefficients=coefficient.tolist(),fit_coordinate_units='translation m, rotation/fingers rad, joint velocity per second',
        equation='u=q_current+a*(q_desired_next-q_current)+b*dq_current',
        fit_stage='first40 source approach frames; no lifted/contact-stage calibration',
        train_target_rmse=np.sqrt(np.mean(error**2,0)).tolist(),
        model_limitations='local decoupled PD approximation; coupling/inertia/contact errors remain; no future dq or action labels at execution',
        computation_device='CPU: small NumPy least-squares/statistics, no neural training')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
