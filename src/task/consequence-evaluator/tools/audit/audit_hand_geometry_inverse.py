"""Bounded GPU audit of 11-point observability and native finger coupling."""
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
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import HAND_LINKS, is_within
from consequence_evaluator.retargeter import ACTIVE_FINGERS, FINGER_SCALE
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics

COUPLINGS = ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not is_within(args.output, ROOT/'outputs/consequence-evaluator'):
        parser.error('fresh task-owned output required')
    occupied = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
    start = time.monotonic()
    torch.set_num_threads(2)
    device = 'cuda:%s' % args.gpu
    with args.source.open('rb') as stream:
        packet = pickle.load(stream)
    assert packet['role_names'][0] == 'reactive_teacher'
    q = torch.as_tensor(packet['dof_position'][:,0], device=device)
    truth = torch.as_tensor(packet['hand_keypoints'][:,0], device=device)
    fk = TorchInspireKinematics(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf', device)
    ids = [QUERY_LINKS.index(name) for name in HAND_LINKS]

    def points(value):
        return fk.forward(value[:,None])[:,0,ids,:3,3]

    with torch.no_grad():
        fk_error = float((points(q)-truth).abs().max())
    if fk_error > 1e-3:
        raise ValueError('FK does not reproduce measured source geometry')
    ticks = [0,48,60,100,140,176,200,300,500]
    sample = q[ticks].detach().requires_grad_(True)
    flattened = points(sample).flatten(1)
    jac = torch.stack([torch.autograd.grad(flattened[:,j].sum(),sample,
        retain_graph=True)[0] for j in range(33)],1)
    control_map = torch.zeros((18,12),device=device)
    control_map[:6,:6] = torch.eye(6,device=device)
    for j,index in enumerate(ACTIVE_FINGERS):
        control_map[index,j+6] = 1
    for dst,src,ratio in COUPLINGS:
        control_map[dst] = ratio*control_map[src]
    # Translation is in metres; rotation/joints are in radians. Report this
    # scaling explicitly: rank is local structural observability, not force.
    full_s = torch.linalg.svdvals(jac).detach().cpu().numpy()
    coupled_s = torch.linalg.svdvals(jac@control_map).detach().cpu().numpy()
    residual = np.stack([(q[:,dst]-ratio*q[:,src]).cpu().numpy()
                         for dst,src,ratio in COUPLINGS],-1)
    active = torch.tensor(ACTIVE_FINGERS,device=device)
    variable = q[:,active].detach().clone().requires_grad_(True)
    scale = torch.as_tensor(FINGER_SCALE,device=device)

    def coupled(value):
        out = q.detach().clone()
        out[:,active] = value
        for dst,src,ratio in COUPLINGS:
            out[:,dst] = ratio*out[:,src]
        return out

    # Truth initialization and exact measured wrist make this a favorable
    # coupled-fit lower bound. It is explicitly NOT a deployable inverse.
    with torch.no_grad():
        before = points(coupled(variable))-truth
    optimizer = torch.optim.Adam([variable],lr=.025)
    best = variable.detach().clone()
    best_error = before.square().sum((1,2))
    for iteration in range(240):
        if time.monotonic()-start > 110:
            raise RuntimeError('bounded geometry audit exceeded 110s')
        error = (points(coupled(variable))-truth).square().sum((1,2))
        with torch.no_grad():
            improved = error < best_error
            best[improved] = variable[improved]
            best_error = torch.minimum(best_error,error)
        optimizer.zero_grad(); error.mean().backward(); optimizer.step()
        with torch.no_grad():
            variable.clamp_(min=0); variable.copy_(torch.minimum(variable,scale))
    with torch.no_grad():
        fitted_q = coupled(best)
        fitted_points = points(fitted_q)
        after = fitted_points-truth
    delta = (best-q[:,active]).detach().cpu().numpy()
    report = dict(schema='ref2dex.hand-geometry-inverse-audit.v1',engineering_only=True,
        source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        elapsed_s=time.monotonic()-start,gpu=args.gpu,source_fk_max_abs_m=fk_error,
        jacobian_ticks=ticks,jacobian_coordinate_units='wrist translation m, wrist angles and finger joints rad',
        full18_rank_at_1e_6=(full_s>1e-6).sum(-1).tolist(),
        coupled12_rank_at_1e_6=(coupled_s>1e-6).sum(-1).tolist(),
        full18_singular_values=full_s.tolist(),coupled12_singular_values=coupled_s.tolist(),
        coupling_residual_rmse_rad=np.sqrt(np.mean(residual**2,0)).tolist(),
        coupling_residual_max_abs_rad=np.abs(residual).max(0).tolist(),
        coupled_before_coordinate_rmse_mm=float(before.square().mean().sqrt())*1000,
        coupled_best_coordinate_rmse_mm=float(after.square().mean().sqrt())*1000,
        coupled_best_per_point_rmse_mm=after.square().sum(-1).mean(0).sqrt().cpu().tolist(),
        best_active_q_difference_rmse_rad=np.sqrt(np.mean(delta**2,0)).tolist(),
        fit_limitations='truth initialization and measured wrist; bounded local fit, no global minimum or deployment claim; joint pose observability is not command/force observability')
    report['coupled_best_per_point_rmse_mm'] = [v*1000 for v in report['coupled_best_per_point_rmse_mm']]
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'fit.npz',q=fitted_q.cpu().numpy(),points=fitted_points.cpu().numpy(),
        per_frame_point_rmse_mm=after.square().mean((1,2)).sqrt().cpu().numpy()*1000,
        true_q=q.cpu().numpy(),coupling_residual_rad=residual)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
