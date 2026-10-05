#!/usr/bin/env python3
"""Read-only ref14_2 P2/P3 audit of frozen actual flows and tiny FK feasibility.

CPU statistics; optional FK <=4 states is geometry-only engineering smoke.
No simulation, learning, proposed candidate search, noise sweep, or new claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src')]
from oracle_y_utility import CANDIDATES
from src.task.CmResidual.dexplore_cm_geometry import (native_joint_limits,
    dexplore_action_to_native_targets, pose_xyzw_to_matrix)
from src.task.CmResidual.v118_planner import TorchInspireKinematics, QUERY_LINKS, NATIVE_TO_URDF


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def diversity(flow):
    """RMS per corresponding point/chunk distance, preserving metric units."""
    flow = np.asarray(flow, dtype=float)
    if flow.ndim < 3 or flow.shape[1] != len(CANDIDATES) or not np.isfinite(flow).all():
        raise ValueError('finite [states,seven_candidates,...] flow required')
    flat = flow.reshape(len(flow), len(CANDIDATES), -1)
    distance = np.sqrt(np.mean((flat[:,:,None]-flat[:,None,:])**2, axis=-1))
    singular = np.linalg.svd(flat-flat[:,:1], compute_uv=False)
    ranks = (singular > np.maximum(singular[:,:1]*1e-6, 1e-10)).sum(-1)
    energy = singular**2
    weights = energy/np.maximum(energy.sum(-1,keepdims=True),1e-30)
    effective = np.exp(-(weights*np.log(np.maximum(weights,1e-30))).sum(-1))
    effective[energy.sum(-1)==0] = 0
    return dict(states=len(flow), pair_rms_mm_mean=(distance.mean(0)*1000).tolist(),
                pair_rms_mm_min=(distance.min(0)*1000).tolist(),
                baseline_contrast_ranks=ranks.tolist(),
                entropy_effective_contrast_rank=effective.tolist(),
                singular_values_mean=singular.mean(0).tolist(),
                rank_tolerance='max(1e-6*largest_singular_value,1e-10 metres)',
                baseline_rms_mm_mean=(distance[:,0].mean(0)*1000).tolist())


def hand_local(tips, pose):
    matrix = pose_xyzw_to_matrix(pose.reshape(-1,7)).reshape(*pose.shape[:-1],4,4)
    return torch.einsum('btnj,btjk->btnk', tips-matrix[:,:,:3,3][:,:,None], matrix[:,:,:3,:3])


def mimic_matrix():
    """Native PD independent drivers and exact follower slope (no claim on dynamics)."""
    drivers = (0,1,2,3,4,5,6,8,10,12,14,15)
    matrix = torch.zeros(18,len(drivers),dtype=torch.float64)
    for col, driver in enumerate(drivers):
        matrix[driver,col] = 1
    for follower, driver, slope in ((7,6,1.05),(9,8,1.05),(11,10,1.05),
                                    (13,12,1.05),(16,15,.6),(17,15,.8)):
        matrix[follower,drivers.index(driver)] = slope
    return drivers, matrix


def coupled_step_scale(q, step, lower, upper):
    """Largest uniform scale in [0,1] preserving all native follower limits.

    Reports failure if initial q is already outside limits; does not clip each
    follower separately because that would violate mimic coupling.
    """
    q, step = np.asarray(q), np.asarray(step)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if not all(np.isfinite(v).all() for v in (q,step)) or any(np.isnan(v).any() for v in (lower,upper)) or np.any(lower > upper):
        raise ValueError('finite state/step and non-NaN ordered limits required')
    if np.any(q < lower-1e-7) or np.any(q > upper+1e-7):
        return None
    active = np.abs(step) > 1e-15
    allowed = np.where(step > 0, upper-q, q-lower)
    return float(np.clip(np.min(allowed[active]/np.abs(step[active])) if active.any() else 1., 0., 1.))


def geometry_joint_limits(urdf):
    """Local correction: continuous URDF joints have no position bound.

    Do not change the shared target-mapping helper used by active collectors.
    """
    lower, upper = native_joint_limits(urdf,'cpu')
    movable = [joint for joint in ET.parse(urdf).getroot().findall('joint')
               if joint.get('type','fixed') != 'fixed']
    continuous = [native for native, index in enumerate(NATIVE_TO_URDF)
                  if movable[index].get('type') == 'continuous']
    lower[continuous], upper[continuous] = -torch.inf, torch.inf
    return lower, upper, continuous


def tiny_geometry(q, urdf):
    """Local 15D fingertip flow Jacobian in common FK axes, 12 drivers.

    Columns use .001m wrist and .01rad angular increments for conditioning;
    DLS targets are synthetic .001m wrist-z shifts only. No policy adjustment.
    """
    q = q[:4].double()
    fk = TorchInspireKinematics(urdf, 'cpu')
    tip_ids = [QUERY_LINKS.index(n+'_tip') for n in ('index','middle','pinky','ring','thumb')]
    drivers, coupling = mimic_matrix()
    units = torch.tensor([.001]*3+[.01]*9,dtype=torch.float64)
    direction = coupling*units
    eps = 1e-3
    def points(value):
        return fk.forward(value)[:,:,tip_ids,:3,3].flatten(2)
    plus = points(q[:,None]+eps*direction.T[None])
    minus = points(q[:,None]-eps*direction.T[None])
    jac = ((plus-minus)/(2*eps)).transpose(1,2).numpy()
    lower, upper, continuous = geometry_joint_limits(urdf)
    reports = []
    for state, j in enumerate(jac):
        singular = np.linalg.svd(j, compute_uv=False)
        rank = int((singular > max(singular[0]*1e-6,1e-10)).sum())
        desired = np.tile([0.,0.,.001],5)
        # Dimensionless driver coordinates; .01 unit-scaled ridge fixed engineering default.
        coeff = np.linalg.solve(j.T@j+1e-8*np.eye(len(drivers)),j.T@desired)
        step = direction.numpy()@coeff
        scale = coupled_step_scale(q[state].numpy(),step,lower.numpy(),upper.numpy())
        achieved = j@coeff*(scale if scale is not None else 0.)
        realized = None
        if scale is not None:
            moved = q[state:state+1]+torch.tensor(step*scale)[None]
            actual = (points(moved[:,None])-points(q[state:state+1,None]))[0,0].numpy()
            realized = float(np.sqrt(np.mean((actual-desired)**2))*1000)
        reports.append(dict(state=state, rank=rank, singular_values=singular.tolist(),
            wrist_singular_values=np.linalg.svd(j[:,:6],compute_uv=False).tolist(),
            finger_singular_values=np.linalg.svd(j[:,6:],compute_uv=False).tolist(),
            condition_nonzero=float(singular[0]/singular[rank-1]) if rank else None,
            native_initial_in_limits=scale is not None, limit_step_scale=scale,
            linear_residual_rms_mm=float(np.sqrt(np.mean((achieved-desired)**2))*1000),
            nonlinear_fk_residual_rms_mm=realized))
    return dict(states=reports,drivers=list(drivers),coupling=coupling.tolist(),
                continuous_native_indices=continuous,position_limit_contract='continuous joints unbounded',
                column_units=units.tolist(),finite_difference_epsilon=eps,damping_squared=1e-8,
                scope='geometry only; no contact/dynamics/PD tracking guarantees')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tiny-fk', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be unique')
    torch.set_num_threads(2)
    archive = torch.load(args.archive,map_location='cpu',weights_only=False)
    if archive['schema'] != 'ref2dex.paired_actual_flow.v1':
        raise ValueError('unknown flow schema')
    # Hash every frozen source before using the archive; never silently accept drift.
    hashes = {str(args.archive.resolve()):sha(args.archive)}
    for path, expected in archive['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('frozen input drift: '+path)
        hashes[path] = expected
    batches, first_q = [], None
    for batch in archive['batches']:
        result_path = Path(batch['batch_result'])
        result = json.loads(result_path.read_text())
        rows = batch['rows'].long()
        flow = batch['flow'].numpy().reshape(len(rows),7,2,120,3)*.02
        surface = diversity(flow)
        surface['per_finger'] = {name:diversity(flow[:,:,:,24*i:24*(i+1)])
                                for i,name in enumerate(('index','middle','pinky','ring','thumb'))}
        panels = [Path(p) for p in result['input_sha256'] if p.endswith('panel.pt')]
        world, local, wrist = [], [], []
        reference = None
        for candidate, path in enumerate(panels):
            p = torch.load(path,map_location='cpu',weights_only=False)
            if p['candidate'] != candidate:
                raise ValueError('candidate identity/order mismatch')
            initial = p['before_fingertip_positions'][rows]
            if reference is None:
                reference = {key:p[key][rows].clone() for key in ('before','history','before_fingertip_positions','before_hand_base_pose')}
                if first_q is None:
                    first_q = p['history'][rows,-1,:18]
            elif any(not torch.equal(p[key][rows],value) for key,value in reference.items()):
                raise ValueError('panels do not share identical initial state')
            tips = torch.stack((initial,p['fingertip_positions'][rows,3],p['fingertip_positions'][rows,7]),1)
            poses = torch.stack((p['before_hand_base_pose'][rows],p['hand_base_pose'][rows,3],p['hand_base_pose'][rows,7]),1)
            loc = hand_local(tips,poses)
            world.append((tips[:,1:]-tips[:,:-1]).numpy())
            local.append((loc[:,1:]-loc[:,:-1]).numpy())
            wrist.append((poses[:,1:,:3]-poses[:,:-1,:3]).numpy())
        batches.append(dict(source=str(result_path),surface_current_object_frame=surface,
            fingertip_world=diversity(np.stack(world,1)),
            fingertip_moving_hand_frame=diversity(np.stack(local,1)),
            wrist_translation_world=diversity(np.stack(wrist,1))))
    report = dict(schema='ref2dex.candidate_flow_geometry_audit.v1',kind='ENGINEERING_OFFLINE_AUDIT',
        candidates=list(CANDIDATES),batches=batches,input_sha256=hashes,
        code_sha256={str(path.resolve()):sha(path) for path in (
            Path(__file__), ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
            ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',
            ROOT/'src/task/CmResidual/v118_planner.py')},
        device='CPU: pure array statistics plus at most4-state finite-difference FK engineering smoke',
        limitations=['Descriptive original common-anchor panels; not later rolling same-state opportunities.',
            'RMS component distances and algebraic rank do not establish useful action diversity or new candidate count.',
            'Moving-hand frame removes rigid wrist motion; world wrist metric covers translation only.',
            'Geometry feasibility ignores contact, dynamics and execution tracking; no planner claim.'])
    if args.tiny_fk:
        urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
        report['asset_sha256'] = {str(urdf):sha(urdf)}
        report['geometry'] = tiny_geometry(first_q,urdf)
    # Atomic exclusive output; no overwrite even if another audit completed meanwhile.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(kind=report['kind'],batches=len(batches),output=str(args.output))))

if __name__ == '__main__':
    main()
