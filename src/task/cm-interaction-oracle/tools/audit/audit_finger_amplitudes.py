#!/usr/bin/env python3
"""Auditable commanded/PD/realized finger magnitudes, with explicit units."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from intervention import INDEPENDENT_FINGER_NAMES, INDEPENDENT_FINGER_INDICES
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits, dexplore_action_to_native_targets, NATIVE_TO_URDF, world_to_object_frame, pose_xyzw_to_matrix
from probe_duration_response import residual_fit, sha


def local_tip_motion(p):
    before=world_to_object_frame(p['before_fingertip_positions'],pose_xyzw_to_matrix(p['before_hand_base_pose']),vector=False)
    after=world_to_object_frame(p['fingertip_positions'].flatten(0,1),pose_xyzw_to_matrix(p['hand_base_pose'].flatten(0,1)),vector=False)
    return after.reshape_as(p['fingertip_positions'])-before[:,None]


def finger_audit(p, design=None, labels=None):
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    lo,hi=native_joint_limits(asset,'cpu'); ranges=hi-lo
    joints=[j for j in ET.parse(asset).getroot().findall('joint') if j.get('type')!='fixed']
    metadata=[dict(native_index=i,name=joints[NATIVE_TO_URDF[i]].get('name'),range_rad=float(ranges[i])) for i in range(6,18)]
    driver=list(INDEPENDENT_FINGER_INDICES); arms=p['arm']; amps=p.get('amplitude',torch.ones(len(arms)))
    levels=p.get('amplitude_levels',[1.]); rows=[]
    delta_action=p['actions']-p['base_actions']; pd=p['pd_targets']-p['pd_base_targets']
    active=torch.arange(32)[None]<p['duration'][:,None]
    q_delta=tip_delta=None
    if 'native_q' in p:
        q_delta=p['native_q'][:,7]-p['history'][:,-1,:18]
        tip_delta=local_tip_motion(p)[:,7]
    for amp in levels:
        for arm,name in enumerate(p['arm_names']):
            nominal=p['delta'][arm]*amp
            if not nominal[6:].count_nonzero() and arm!=0: continue
            keep=(arms==arm)&(amps==amp); valid=active[keep]
            baseline=torch.zeros(1,18)
            target=(dexplore_action_to_native_targets(nominal[None],baseline,lo,hi)-dexplore_action_to_native_targets(baseline,baseline,lo,hi))[0]
            row=dict(alpha=amp,arm=arm,name=name,trials=int(keep.sum()),
                nominal_action=nominal[6:].tolist(),nominal_pd_rad=target[6:].tolist(),
                nominal_driver_range_fraction=(target[driver]/ranges[driver]).tolist())
            if keep.any():
                delivered=delta_action[keep][valid]; targets=pd[keep][valid]
                intended=nominal[6:].abs()>0
                clipped=(delivered[:,6:]-nominal[6:]).abs()>1e-6
                row.update(pd_mean_rad=targets[:,6:].mean(0).tolist(),pd_mean_deg=torch.rad2deg(targets[:,6:]).mean(0).tolist(),
                    pd_mean_range_fraction=(targets[:,6:]/ranges[6:]).mean(0).tolist(),
                    pd_min_rad=targets[:,6:].amin(0).tolist(),pd_max_rad=targets[:,6:].amax(0).tolist(),
                    delivered_action_mean=delivered[:,6:].mean(0).tolist(),
                    clipping_fraction=float(clipped[:,intended].float().mean()) if intended.any() else 0.)
                if q_delta is not None:
                    row.update(raw_q_delta8_mean_rad=q_delta[keep,6:].mean(0).tolist(),
                        raw_local_tip_travel8_mean_mm=(tip_delta[keep].norm(dim=-1).mean(0)*1000).tolist())
            rows.append(row)
    adjusted=[]
    if q_delta is not None and design is not None:
        assert len(levels)==1
        n_cells=len(p['arm_names'])-1
        values=torch.cat((q_delta[:,driver],tip_delta.flatten(1)), -1).numpy().astype(float)
        beta=residual_fit(design,labels,values,n_cells)[0]
        for arm in range(1,n_cells+1):
            vector=beta[arm-1,6:].reshape(5,3)*1000
            adjusted.append(dict(arm=arm,name=p['arm_names'][arm],
                driver_q_delta8_arm_minus_zero_rad=beta[arm-1,:6].tolist(),
                local_tip_delta8_arm_minus_zero_vector_mm=vector.tolist(),
                local_tip_delta8_arm_minus_zero_vector_norm_mm=np.linalg.norm(vector,axis=-1).tolist()))
    return dict(native_joints=metadata,native_array_indices=list(range(6,18)),driver_names=list(INDEPENDENT_FINGER_NAMES),driver_indices=driver,
        tip_names=p.get('fingertip_names'),cells=rows,adjusted_motion=adjusted,
        actual_post_q_and_true_tip_available=q_delta is not None,
        limits='PD is a command difference at same current q, not realized angle. Raw before/after motion includes baseline evolution. Adjusted vector norm is norm of a contrast, not a contrast of mean travel, and is descriptive. Equal driver range fractions do not imply equal mimic fractions or millimetres.')


def markdown(audit):
    joints=audit['native_joints']; drivers=audit['driver_indices']
    lines=['## Per-finger perturbation amplitude audit','',
        'Units: native action dimensionless; PD targets/actual q radians; tip displacement mm.',
        'PD target changes are commanded, not actual executed joint angles. Driver table order: index, middle, pinky, ring, thumb-yaw, thumb-pitch. JSON PD/action arrays use all12 native joints6..17, including followers.', '',
        '| Driver | Native index | Physical range (rad) | Native mimic followers |','| --- | --- | --- | --- |']
    follower={6:'7: ×1.05',8:'9: ×1.05',10:'11: ×1.05',12:'13: ×1.05',14:'none',15:'16: ×0.6;17: ×0.8'}
    for name,index in zip(audit['driver_names'],drivers):
        lines.append(f'| {name} | {index} | {joints[index-6]["range_rad"]:.4f} | {follower[index]} |')
    lines+=['','### Delivered PD target magnitude per arm','',
        'Signed means over ALL active steps; min/max per joint and normalized range fractions are preserved in `finger_amplitudes.json`. Composite arms exclude thumb-yaw. Clip % counts active commanded finger coordinates.', '',
        '| alpha | Arm | n | index rad | middle rad | pinky rad | ring rad | yaw rad | pitch rad | clip % |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in audit['cells']:
        if not row['trials']: continue
        values=row['pd_mean_rad']
        cells=' | '.join(f'{values[i-6]:+.5f}' for i in drivers)
        lines.append(f'| {row["alpha"]:g} | {row["name"]} | {row["trials"]} | {cells} | {row["clipping_fraction"]*100:.2f} |')
    lines+=['','For composite ± at a20% driver range dose, intermediate followers6→7 etc receive ±0.336rad (10.70% of their3.14rad range); thumb-pitch followers receive ±0.066/0.088rad (2.10/2.80% of their ranges). At5% driver dose these are divided by4. Yaw has no target coupling.','']
    if audit['adjusted_motion']:
        lines+=['### Measured native driver response at step8','',
            'Current-state-adjusted arm-minus-zero contrasts of actual q(step8)−q(before), radians. Not PD targets.', '',
            '| Arm | index | middle | pinky | ring | yaw | pitch |','| --- | --- | --- | --- | --- | --- | --- |']
        for row in audit['adjusted_motion']:
            cells=' | '.join(f'{v:+.5f}' for v in row['driver_q_delta8_arm_minus_zero_rad'])
            lines.append(f'| {row["name"]} | {cells} |')
        lines+=['','### Measured five-finger tip response at step8','',
            'True tip positions in the measured hand-base frame. Values are norms of adjusted displacement vector contrasts (mm), not differences of average travel. Raw within-arm travel means are in JSON and include ordinary baseline evolution.', '',
            '| Arm | index mm | middle mm | pinky mm | ring mm | thumb mm |', '| --- | --- | --- | --- | --- | --- |']
        for row in audit['adjusted_motion']:
            cells=' | '.join(f'{v:.2f}' for v in row['local_tip_delta8_arm_minus_zero_vector_norm_mm'])
            lines.append(f'| {row["name"]} | {cells} |')
    else:
        lines+=['Actual post-step q and true tip motion are unavailable in the legacy packet; old contact-body positions are not substituted for tips.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path)
    args=parser.parse_args(); args.run_dir.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2)
    p=torch.load(args.dataset,weights_only=False); d=torch.load(args.diagnostic,weights_only=False) if args.diagnostic else {}
    a=finger_audit(p,d.get('design'),d.get('labels'))
    (args.run_dir/'finger_amplitudes.json').write_text(json.dumps(a,indent=2)+'\n')
    (args.run_dir/'finger_amplitudes.md').write_text(markdown(a))
    (args.run_dir/'manifest.json').write_text(json.dumps(dict(dataset_sha256=sha(args.dataset),script_sha256=sha(__file__),
        diagnostic_sha256=sha(args.diagnostic) if args.diagnostic else None,
        native_contract_sha256=sha(ROOT/'src/task/CmResidual/dexplore_cm_geometry.py'),
        joint_map_source_sha256=sha(ROOT/'src/task/CmResidual/v118_planner.py'),
        asset_sha256=sha(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'),
        device='CPU file/statistical audit; no models'),indent=2)+'\n')


if __name__=='__main__': main()
