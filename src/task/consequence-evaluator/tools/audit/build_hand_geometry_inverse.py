"""Oracle 11-point inverse with reset calibration; no future q/action labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT),str(TASK/'src')]
from consequence_evaluator.contracts import HAND_LINKS, is_within
from consequence_evaluator.object_relative_servo import recover_wrist
from consequence_evaluator.retargeter import ACTIVE_FINGERS, FINGER_SCALE, wrist_rotation
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics

COUPLINGS = ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists() or not is_within(args.output,ROOT/'outputs/consequence-evaluator'):
        parser.error('fresh task-owned output required')
    occupied=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('GPU occupied: '+occupied)
    torch.set_num_threads(2);device='cuda:%s'%args.gpu;start=time.monotonic()
    with args.source.open('rb') as stream:packet=pickle.load(stream)
    # These are the ONLY inverse inputs extracted from the source packet.
    hand=packet['hand_keypoints'][:,0].copy()
    reset_q=packet['dof_position'][0,0].copy()
    assert hand.shape==(543,11,3) and reset_q.shape==(18,)
    rest=(wrist_rotation(reset_q).T@(hand[0]-reset_q[:3]).T).T
    wrist=[recover_wrist(hand[0],rest,reset_q)]
    for points in hand[1:]:wrist.append(recover_wrist(points,rest,wrist[-1]))
    wrist=np.stack(wrist)
    fixed=torch.as_tensor(np.tile(wrist,(2,1)),device=device)
    target=torch.as_tensor(np.tile(hand,(2,1,1)),device=device)
    active=torch.as_tensor(ACTIVE_FINGERS,device=device)
    scale=torch.as_tensor(FINGER_SCALE,device=device)
    initial=np.stack((np.broadcast_to(reset_q[ACTIVE_FINGERS],(543,6)),
                      np.broadcast_to(.5*FINGER_SCALE,(543,6)))).reshape(-1,6).copy()
    variable=torch.as_tensor(initial,device=device).clone()
    variable.clamp_(min=0);variable.copy_(torch.minimum(variable,scale));variable.requires_grad_(True)
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    fk=TorchInspireKinematics(urdf,device);ids=[QUERY_LINKS.index(name) for name in HAND_LINKS]

    def coupled(value):
        q=fixed.clone();q[:,active]=value
        for dst,src,ratio in COUPLINGS:q[:,dst]=q[:,src]*ratio
        return q

    def points(value):return fk.forward(coupled(value)[:,None])[:,0,ids,:3,3]

    with torch.no_grad():best_error=(points(variable)-target).square().sum((1,2));best=variable.detach().clone()
    optimizer=torch.optim.Adam([variable],lr=.025)
    for iteration in range(300):
        if time.monotonic()-start>110:raise RuntimeError('bounded inverse exceeded110s')
        error=(points(variable)-target).square().sum((1,2))
        with torch.no_grad():
            improved=error<best_error;best[improved]=variable[improved];best_error=torch.minimum(error,best_error)
        optimizer.zero_grad();error.mean().backward();optimizer.step()
        with torch.no_grad():variable.clamp_(min=0);variable.copy_(torch.minimum(variable,scale))
    with torch.no_grad():
        errors=best_error.reshape(2,543);choice=errors.argmin(0)
        select=choice*543+torch.arange(543,device=device)
        result=coupled(best)[select];fitted=points(best)[select]
        delta=fitted-torch.as_tensor(hand,device=device)
    report=dict(schema='ref2dex.hand-geometry-inverse.v1',engineering_only=True,training_allowed=False,
        source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        hand_sha256=hashlib.sha256(hand.tobytes()).hexdigest(),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        inverse_inputs='future11point geometry, reset q18, static kinematics; no later measured q/dq or source controls',
        root_calibration='known reset q and fixed proximal-link origins',
        finger_inverse='coupled six controls, two fixed reset/midrange starts, bounded local geometric fit',
        elapsed_s=time.monotonic()-start,gpu=args.gpu,iterations=300,
        coordinate_rmse_mm=float(delta.square().mean().sqrt())*1000,
        per_point_distance_rmse_mm=(delta.square().sum(-1).mean(0).sqrt()*1000).cpu().tolist(),
        limits='privileged future geometry oracle, offline local optimization; loaded actual joints need not satisfy target coupling')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'inverse.npz',q=result.cpu().numpy(),fitted_points=fitted.cpu().numpy(),
        target_points=hand,reset_q=reset_q,chosen_start=choice.cpu().numpy())
    report['inverse_sha256']=hashlib.sha256((args.output/'inverse.npz').read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
