"""Independent matrix transport, native dispatch, and measured contact audit."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.retargeter import commanded_targets, wrist_rotation, ACTIVE_FINGERS
from consequence_evaluator.object_relative_servo import local_points
from consequence_evaluator.contracts import HAND_LINKS, is_within


def pose(q):
    out = np.broadcast_to(np.eye(4), (*q.shape[:-1], 4, 4)).copy()
    out[..., :3, 3] = q[..., :3]
    out[..., :3, :3] = wrist_rotation(q)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--surface-gpu', type=int)
    args = parser.parse_args()
    if args.output.exists() or not is_within(args.output, ROOT / 'outputs/consequence-evaluator'):
        parser.error('fresh task-owned output required')
    with args.packet.open('rb') as stream:
        p = pickle.load(stream)
    assert p['schema'] == 'ref2dex.object-relative-gt-servo.v1'
    assert p['engineering_only'] and not p['training_allowed']
    source_path = Path(p['gt_source'])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == p['gt_source_sha256']
    with source_path.open('rb') as stream:
        source = pickle.load(stream)
    assert len(p['actions']) == 542 and not p['done'][:-1].any()
    np.testing.assert_array_equal(p['actions'], p['requested_controls'])
    np.testing.assert_array_equal(p['live_object_query_poses'], p['object_pose'][:-1])
    targets = commanded_targets(source['dof_position'][:-1, 0], source['actions'][:, 0])
    desired = p['desired_pd_targets']
    np.testing.assert_array_equal(desired[:, 0], targets)
    anchor = int(p['object_anchor'] == 'future')
    inverse = np.linalg.inv(source['object_pose'][anchor:anchor+542, 0].astype('float64'))
    report = dict(status='PASS', source_sha256=p['gt_source_sha256'],
        packet_sha256=hashlib.sha256(args.packet.read_bytes()).hexdigest(),
        actual_requested_max_abs=0., anchor=p['object_anchor'], gate=p['gate'])
    errors = []
    layout = p.get('servo_layout', 'command_vs_measured')
    query_indices = np.arange(542)//24*24 if layout == 'chunk_alignment' else np.arange(542)
    preload=None
    if layout in ('finger_preload','finger_preload_late','geometry_pd'):
        path=Path(p['preload_statistics_path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==p['preload_statistics_sha256']
        preload=json.loads(path.read_text())
    inverse_q=None
    if layout in ('geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late'):
        inverse_path=Path(p['geometry_inverse_path'])/'inverse.npz'
        assert hashlib.sha256(inverse_path.read_bytes()).hexdigest()==p['geometry_inverse_sha256']
        with np.load(inverse_path) as data:
            inverse_q=data['q'].copy()
            np.testing.assert_array_equal(data['target_points'],source['hand_keypoints'][:,0])
    for role, target in ((2, targets), (3, source['dof_position'][1:, 0] if layout == 'command_vs_measured' else targets)):
        expect = p['object_pose'][query_indices, role].astype('float64') @ inverse[query_indices] @ pose(target)
        if layout in ('wrist_geometry_late','geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late'):
            assert p['geometry_switch_tick']==120
            values=targets.copy()
            values[120:,:6]=source['dof_position'][121:,0,:6]
            if role==3 or layout in ('geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late'):
                qnext=source['dof_position'][1:,0,:6]
                qfollowing=source['dof_position'][np.minimum(np.arange(1,543)+1,542),0,:6]
                velocity=(qfollowing-qnext)*30
                velocity[-1]=(qnext[-1]-qnext[-2])*30
                values[120:,:6]+=.1*velocity[120:]
            if layout=='geometry_load_delta_late' and role==3:
                current=inverse_q[120].copy()
                current[:6]=source['dof_position'][120,0,:6]+.1*(source['dof_position'][121,0,:6]-source['dof_position'][120,0,:6])*30
                previous=commanded_targets(p['dof_position'][119,3],p['actions'][119,3])
                offset=previous-current
                np.testing.assert_allclose(p['command_anchor_offset'],offset,atol=1e-5)
                values[120:,:6]+=offset[:6]
            expect=pose(values)
            if layout in ('geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late'):
                finger_q=source['dof_position'][:,0] if role==2 and layout=='geometry_inverse_late' else inverse_q
                fingers=targets[:,ACTIVE_FINGERS].copy()
                fingers[120:]=finger_q[121:][:,ACTIVE_FINGERS]
                if layout=='geometry_load_delta_late' and role==3:
                    fingers[120:]+=p['command_anchor_offset'][ACTIVE_FINGERS]
                np.testing.assert_allclose(desired[:,role-1][:,ACTIVE_FINGERS],fingers,atol=1e-6)
            if layout=='geometry_inverse_relative_late' and role==3:
                nominal=expect[120:].copy()
                corrected=p['object_pose'][120:-1,role].astype('float64')@inverse[120:]@nominal
                delta=corrected[:,:3,3]-nominal[:,:3,3]
                delta*=np.minimum(1.,.02/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[:,None]
                rv=Rotation.from_matrix(corrected[:,:3,:3]@np.swapaxes(nominal[:,:3,:3],1,2)).as_rotvec()
                rv*=np.minimum(1.,.15/np.maximum(np.linalg.norm(rv,axis=-1),1e-12))[:,None]
                expect[120:,:3,:3]=Rotation.from_rotvec(rv).as_matrix()@nominal[:,:3,:3]
                expect[120:,:3,3]=nominal[:,:3,3]+delta
        if layout in ('finger_preload','finger_preload_late','geometry_pd'):
            values=targets.copy()
            fingers=source['dof_position'][1:,0][:,ACTIVE_FINGERS].copy()
            if role==3:fingers+=np.asarray(preload['finger_preload_median_rad'],dtype='float32')
            if layout=='finger_preload_late':
                assert p['geometry_switch_tick']==120
                fingers[:120]=targets[:120,ACTIVE_FINGERS]
            if layout=='geometry_pd':
                qnext=source['dof_position'][1:,0,:6]
                qfollowing=source['dof_position'][np.minimum(np.arange(1,543)+1,542),0,:6]
                velocity=(qfollowing-qnext)*30
                velocity[-1]=(source['dof_position'][-1,0,:6]-source['dof_position'][-2,0,:6])*30
                values[:,:6]=qnext+.1*velocity
                if role==2:fingers=targets[:,ACTIVE_FINGERS]
            expect=pose(values)
            np.testing.assert_allclose(desired[:,role-1][:,ACTIVE_FINGERS],fingers,atol=1e-6)
        if layout == 'transport_ablation' and role == 2:
            expect = pose(target)
            expect[:, :3, 3] += p['object_pose'][:-1, role, :3, 3]-source['object_pose'][anchor:anchor+542, 0, :3, 3]
        elif layout in ('transport_ablation', 'chunk_alignment') and role == 3:
            nominal = pose(target)
            delta = expect[:, :3, 3]-nominal[:, :3, 3]
            delta *= np.minimum(1., .02/np.maximum(np.linalg.norm(delta, axis=-1), 1e-12))[:, None]
            rotation_delta = expect[:, :3, :3] @ np.swapaxes(nominal[:, :3, :3], 1, 2)
            rv = Rotation.from_matrix(rotation_delta).as_rotvec()
            rv *= np.minimum(1., .15/np.maximum(np.linalg.norm(rv, axis=-1), 1e-12))[:, None]
            expect[:, :3, :3] = Rotation.from_rotvec(rv).as_matrix() @ nominal[:, :3, :3]
            expect[:, :3, 3] = nominal[:, :3, 3]+delta
        error = float(np.max(np.abs(expect-pose(desired[:, role-1]))))
        if error > 3e-5:
            raise ValueError('SE(3) matrix transport mismatch: %s' % error)
        if layout not in ('finger_preload','finger_preload_late','geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late','geometry_pd'):
            np.testing.assert_array_equal(desired[:, role-1, ACTIVE_FINGERS], target[:, ACTIVE_FINGERS])
        errors.append(error)
    if layout=='geometry_inverse_repeat':
        np.testing.assert_array_equal(desired[:,1],desired[:,2])
    if layout == 'chunk_alignment':
        np.testing.assert_array_equal(p['alignment_query_ticks'], np.arange(0, 542, 24))
        np.testing.assert_array_equal(p['alignment_query_poses'], p['object_pose'][np.arange(0, 542, 24)])
        for tick in range(542):
            for arm in (0, 1):
                np.testing.assert_array_equal(desired[tick, arm+1], p['alignment_query_chunks'][tick//24][arm][tick % 24])
    report['transport_matrix_max_abs'] = errors
    rebuilt = commanded_targets(p['dof_position'][:-1, 1:], p['actions'][:, 1:])
    error = np.max(np.abs(rebuilt-desired), -1)
    np.testing.assert_allclose(error, p['commanded_target_max_abs_error'], atol=2e-6)
    report['native_target_max_abs_error'] = error.max(0).tolist()
    source_local = local_points(source['hand_keypoints'][:, 0], source['object_pose'][:, 0])
    summaries, traces = {}, {}
    for role, name in enumerate(p['role_names']):
        world = p['hand_keypoints'][:, role]-source['hand_keypoints'][:, 0]
        local = local_points(p['hand_keypoints'][:, role], p['object_pose'][:, role])-source_local
        # Per-frame coordinate RMSE, distinct from point-distance error.
        world_error = np.sqrt(np.mean(world**2, axis=(1, 2)))*1000
        local_error = np.sqrt(np.mean(local**2, axis=(1, 2)))*1000
        obj_height = p['object_pose'][:, role, 2, 3]-p['object_pose'][0, role, 2, 3]
        summary = dict(outcome=p['role_outcomes'][name],
            world_coordinate_rmse_mm=float(np.sqrt(np.mean(world[1:]**2))*1000),
            local_coordinate_rmse_mm=float(np.sqrt(np.mean(local[1:]**2))*1000),
            max_object_speed_mps=float(np.linalg.norm(p['object_velocity'][:, role, :3], axis=-1).max()),
            max_object_angular_speed_radps=float(np.linalg.norm(p['object_velocity'][:, role, 3:], axis=-1).max()),
            max_object_lift_m=float(obj_height.max()), peak_lift_tick=int(np.argmax(obj_height)))
        if role:
            clip = p['clipped_coordinate_counts'][:, role-1]
            summary.update(clipped_coordinates=int(clip.sum()),
                first_clip_tick=int(np.flatnonzero(clip)[0]) if clip.any() else None,
                clips_first120=int(clip[:120].sum()))
        summaries[name] = summary
        traces[name] = dict(world_error_mm=world_error, local_error_mm=local_error, object_lift_m=obj_height)
    report['roles'] = summaries
    args.output.mkdir(parents=True)
    if args.surface_gpu is not None:
        occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.surface_gpu),
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError('surface audit GPU occupied: '+occupied)
        import torch
        from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics
        from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
        torch.set_num_threads(2)
        device = 'cuda:%s' % args.surface_gpu
        assets = ROOT / 'third_party/DExplore/dexplore/data/assets'
        geom = _surface_geometry_class()(hand_urdf=assets/'inspire_hand_new/inspire_hand_right.urdf',
            object_urdf=assets/'mjcf/airplane.urdf', query_links=QUERY_LINKS,
            object_count=1024, hand_count=1538, seed=42, device=device)
        fk = TorchInspireKinematics(assets/'inspire_hand_new/inspire_hand_right.urdf', device)
        source_q = torch.as_tensor(source['dof_position'][140:201, 0, None], device=device)
        with torch.no_grad():
            link = fk.forward(source_q)[:, 0]
            key_indices = [QUERY_LINKS.index(name) for name in HAND_LINKS]
            fk_error = float(np.max(np.abs(link[:, key_indices, :3, 3].cpu().numpy()-source['hand_keypoints'][140:201, 0])))
            report['source_fk_keypoint_max_abs_m'] = fk_error
            if fk_error > 1e-3:
                raise ValueError('source FK too inaccurate for historical per-finger surface audit')
            names = p['native_body_names']; fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            query_ids = [names.index(name) for name in QUERY_LINKS]
            from consequence_evaluator.physical_geometry import poses
            groups = [('GT source', link, source['object_pose'][140:201, 0])]
            for role, name in enumerate(p['role_names']):
                states = torch.as_tensor(p['native_rigid_body_states'][140:201, role][:, query_ids], device=device)
                groups.append((name, poses(states), p['object_pose'][140:201, role]))
            gap_summary = {}
            for name, link, opose in groups:
                hand, _ = geom.hand(link)
                obj, _ = geom.object(torch.as_tensor(opose, device=device))
                gaps = []
                for finger in fingers:
                    mask = np.array([QUERY_LINKS[index].startswith(finger+'_') for index in geom.hand_link])
                    value = torch.cdist(hand[:, mask], obj).amin(dim=(1, 2)).cpu().numpy()*1000
                    gaps.append(value)
                gaps = np.stack(gaps, -1)
                traces.setdefault(name, {})['finger_surface_gap_mm_140_200'] = gaps
                gap_summary[name] = dict(fingers=fingers,
                    gap_mean_mm=gaps.mean(0).tolist(), gap_max_mm=gaps.max(0).tolist())
            report['sampled_per_finger_visual_surface_gaps'] = gap_summary
            report['surface_limitations'] = 'unsigned sampled visual-surface proximity, not exact collision distance or pair-labeled contact'
    np.savez_compressed(args.output / 'traces.npz', **{name+'__'+key: value for name, data in traces.items() for key, value in data.items()})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for name in p['role_names']:
        for ax, key in zip(axes, ('world_error_mm', 'local_error_mm', 'object_lift_m')):
            ax.plot(traces[name][key], label=name, linewidth=1)
    for ax, label in zip(axes, ('World hand error (mm)', 'Object-local hand error (mm)', 'Object lift (m)')):
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel('Native control tick (30Hz)'); fig.tight_layout()
    fig.savefig(args.output / 'servo_comparison.png', dpi=140)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
