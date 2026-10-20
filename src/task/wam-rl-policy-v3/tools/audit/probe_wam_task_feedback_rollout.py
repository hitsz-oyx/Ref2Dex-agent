"""Frozen WAM24/8 rollout with TaskController feedback from generated geometry.

This is a motion-only scheduler diagnostic, without physical execution or
measured contact forces. The fixed-goal rollout is read-only comparison data.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def infer(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', REF2DEX_DEEP_OPT='all')
    sys.path.insert(0, str(TASK/'src'))
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.goal_scheduler import rigid_goal_delta
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime, v4_online_samples, collate_v4
    from wam_rl_policy_v3.mano_preview import forecast_world, forecast_object_poses

    started = time.monotonic()
    out = args.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:
        raise ValueError('fresh task-local output required')
    if __import__('shutil').disk_usage(ROOT).free < 20*1024**3:
        raise ValueError('20GiB disk floor')
    meta = json.loads((args.baseline/'manifest.json').read_text())
    if meta['checkpoint_step'] != 80000 or meta['replan_stride'] != 8:
        raise ValueError('frozen step80000 stride8 baseline required')
    corpus = TeacherRolloutCorpus(Path(meta['source_manifest']), 'train')
    index = next(i for i, row in enumerate(corpus.records) if row['sequence'] == meta['sequence'])
    episode = corpus.episode(index)
    spec = TaskSpec.from_packet(episode['task_spec'])
    controller = TaskController(spec, episode['object_points'])
    with np.load(args.baseline/'preview.npz') as z:
        actual_hand = z['teacher_hand'].copy(); actual_pose = z['object_poses'].copy()
        fixed_hand = z['target_hand'].copy(); fixed_pose = z['display_object_poses'].copy()
        fixed_valid = z['prediction_valid'].copy(); fixed_goal = z['goal_poses'].copy()
        old_first = z['wam_delta_anchor'][0].copy()
    n = len(actual_hand)
    if n > 1200: raise ValueError('bounded recording required')
    out.mkdir(parents=True)
    torch.set_num_threads(2); torch.manual_seed(246)
    runtime = V4Runtime(Path(meta['wam']), [dict(object_code='dexplore/'+meta['object'], scale=1.)],
        args.geometry_cache.resolve(), dense_manifest=args.geometry_cache/'surface-manifest.json',
        allow_geometry_replacement=True, device='cuda:0')
    hand = np.full_like(actual_hand, np.nan); pose = np.full_like(actual_pose, np.nan)
    hand[:4] = actual_hand[:4]; pose[:4] = actual_pose[:4]
    valid = np.zeros(n, bool); goals = np.full_like(pose, np.nan)
    traces = []; windows = []; input_hand = []; input_pose = []; deltas = []; scenes = []
    stop = None; first_diff = None

    def observe(t):
        if t:
            velocity = np.r_[(pose[t, :3, 3]-pose[t-1, :3, 3])*30,
                Rotation.from_matrix(pose[t, :3, :3]@pose[t-1, :3, :3].T).as_rotvec()*30]
        else:
            velocity = np.zeros(6)
        result = controller.observe(pose[t], hand[t], velocity, t/30,
            force_valid=False, supported=None)
        d = result['decision']; f = result['feedback']
        goals[t] = d.local_goal_pose
        evidence = controller.observer.last_evidence
        traces.append(dict(tick=t, phase=d.phase.name, progress=d.path_progress,
            source_cursor=d.source_cursor_frame, terminated=d.terminated,
            failure_reason=d.failure_reason, transitions=d.transitions,
            grasp_established=f.grasp_established, grasp_lost=f.grasp_lost,
            force_valid=f.force_valid, feedback_source=f.source,
            fingertip_gap_m=evidence['fingertip_gap_m'], near_fingers=evidence['near_fingers']))
        return result

    for t in range(4): current = observe(t)
    for anchor in range(3, n-1, 8):
        if time.monotonic()-started > 180: raise TimeoutError('three-minute inference limit')
        decision = current['decision']
        if decision.terminated:
            stop = dict(tick=anchor, reason=decision.failure_reason or 'task_complete'); break
        hp = hand[anchor-3:anchor+1]; op = pose[anchor-3:anchor+1]
        objects = np.c_[op[:, :3, 3], Rotation.from_matrix(op[:, :3, :3]).as_quat()][None]
        goal = rigid_goal_delta(op[-1], decision.local_goal_pose, runtime.geometry[0][2])
        sample = v4_online_samples(objects, hp[None], runtime.geometry, goal_delta=goal[None])[0]
        if not runtime.workspace_valid([sample])[0]:
            stop = dict(tick=anchor, reason='predicted_state_outside_WAM_workspace'); break
        batch = {k: v.to(runtime.device) for k, v in collate_v4([sample]).items()}
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            prediction = runtime.predict(batch)
            hd = prediction['trajectory'][0, :, 0].float().cpu().numpy()
            sd = prediction['scene'][0, 0].float().cpu().numpy()
        if not np.isfinite(hd).all() or not np.isfinite(sd).all():
            stop = dict(tick=anchor, reason='nonfinite_prediction'); break
        if first_diff is None:
            first_diff = float(np.abs(hd-old_first).max()*1000)
        world = forecast_world(hand[anchor], hd, pose[anchor])
        objects, residual = forecast_object_poses(runtime.geometry[0][2], sd, pose[anchor])
        input_hand.append(hp.copy()); input_pose.append(op.copy()); deltas.append(hd); scenes.append(sd)
        consumed = []
        for j, frame in enumerate(range(anchor+1, min(anchor+9, n))):
            hand[frame] = world[j+1]; pose[frame] = objects[j]; valid[frame] = True
            consumed.append(frame); current = observe(frame)
            if current['decision'].terminated:
                stop = dict(tick=frame, reason=current['decision'].failure_reason or 'task_complete'); break
        windows.append(dict(anchor=anchor, first=consumed[0], last=consumed[-1]))
        if stop: break
    if not valid.any(): raise ValueError('no generated continuation')
    common = valid & fixed_valid
    hand_error = np.linalg.norm(hand-actual_hand, axis=-1).mean(-1)*1000
    fixed_error = np.linalg.norm(fixed_hand-actual_hand, axis=-1).mean(-1)*1000
    object_error = np.linalg.norm(pose[:, :3, 3]-actual_pose[:, :3, 3], axis=-1)*1000
    fixed_object_error = np.linalg.norm(fixed_pose[:, :3, 3]-actual_pose[:, :3, 3], axis=-1)*1000
    goal_gap = np.linalg.norm(goals[:, :3, 3]-fixed_goal[:, :3, 3], axis=-1)*1000
    last = int(np.flatnonzero(valid)[-1]); peak = torch.cuda.max_memory_allocated()/2**20
    np.savez_compressed(out/'rollout.npz', actual_hand=actual_hand, actual_pose=actual_pose,
        fixed_hand=fixed_hand, fixed_pose=fixed_pose, fixed_valid=fixed_valid, fixed_goal=fixed_goal,
        hand=hand, pose=pose, valid=valid, goal=goals, common=common,
        hand_error_mm=hand_error, fixed_hand_error_mm=fixed_error, object_error_mm=object_error,
        fixed_object_error_mm=fixed_object_error, goal_translation_difference_mm=goal_gap,
        input_hand_history=np.asarray(input_hand), input_object_history=np.asarray(input_pose),
        delta=np.asarray(deltas), scene_delta=np.asarray(scenes), flow_query=runtime.geometry[0][2],
        controller_points=episode['object_points'])
    result = dict(status='COMPLETED', experiment_id='P-20261020-wam-mano-preview',
        run_id=out.name, work_version='wam-task-feedback-rollout-v1', task='wam-rl-policy-v3',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        python=sys.executable, torch=torch.__version__, gpu=args.gpu, seed=246,
        baseline=str(args.baseline.resolve()), checkpoint=meta['wam'], checkpoint_step=runtime.step,
        geometry_cache=str(args.geometry_cache.resolve()), sequence=meta['sequence'],
        forecast_horizon=24, replan_stride=8, recording_frames=n, consumed_frames=int(valid.sum()),
        common_frames=int(common.sum()), last_tick=last, stop=stop, windows=windows,
        feedback='current TaskController, every generated tick, motion-only; force_valid=False, supported=None',
        history='only initial H4 is recorded; subsequent histories and velocities use generated states',
        first_forecast_vs_fixed_max_difference_mm=first_diff,
        fixed_common_hand_mean_mm=float(fixed_error[common].mean()),
        adaptive_common_hand_mean_mm=float(hand_error[common].mean()),
        fixed_common_object_mean_mm=float(fixed_object_error[common].mean()),
        adaptive_common_object_mean_mm=float(object_error[common].mean()),
        adaptive_last_hand_mm=float(hand_error[last]), goal_difference_mean_mm=float(goal_gap[common].mean()),
        grasp_established_ticks=sum(t['grasp_established'] for t in traces),
        phase_counts={name: sum(t['phase']==name for t in traces) for name in ('ACQUIRE','TRACK','RELEASE')},
        task_complete=bool(current['decision'].task_complete), elapsed_s=time.monotonic()-started,
        peak_cuda_mib=peak, traces=traces,
        limitations=['single Inspire initialization, not WAM TRAIN', 'no physics, robot commands or contact forces',
            'scheduler sees generated geometry, not measured execution',
            'trajectory error is reproduction distance, not standalone physical validity',
            'v4 does not consume planned contact/release event tokens',
            'scheduler termination stops prediction; missing tail must not be scored or fabricated'])
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('traces','windows')}), flush=True)


def visualize(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    out = args.output; result = json.loads((out/'result.json').read_text())
    with np.load(out/'rollout.npz') as z: d = {k: z[k].copy() for k in z.files}
    t = np.arange(result['recording_frames'])/30; v = d['common']
    fig, axes = plt.subplots(3,1,figsize=(10,9),constrained_layout=True)
    axes[0].plot(t[v],d['fixed_hand_error_mm'][v],label='Fixed recorded-controller goals')
    axes[0].plot(t[v],d['hand_error_mm'][v],label='Generated-state TaskController goals')
    axes[0].set_ylabel('Hand error (mm)'); axes[0].legend()
    axes[1].plot(t[v],d['fixed_object_error_mm'][v],label='Fixed goals')
    axes[1].plot(t[v],d['object_error_mm'][v],label='Adaptive goals')
    axes[1].set_ylabel('Object translation error (mm)'); axes[1].legend()
    trace=result['traces']; ticks=np.array([r['tick'] for r in trace])
    gap=np.array([np.nan if r['fingertip_gap_m'] is None else r['fingertip_gap_m']*1000 for r in trace])
    axes[2].plot(ticks/30,gap,label='Generated nearest fingertip-object gap')
    axes[2].set_ylabel('Gap (mm)'); axes[2].set_xlabel('Recording time (s)'); axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('WAM24 / consume8 / predicted history / motion-only TaskController')
    fig.savefig(out/'scheduler-comparison.png',dpi=150);plt.close(fig)
    bones=np.array([(0,i) for i in range(1,6)]+[(i,i+5) for i in range(1,6)])
    fig=plt.figure(figsize=(12,4),dpi=100);axes=[fig.add_subplot(1,3,k+1,projection='3d') for k in range(3)]
    series=[(d['actual_hand'],d['actual_pose'],'Recorded teacher'),
        (d['fixed_hand'],d['fixed_pose'],'Fixed goals / predicted H4'),
        (d['hand'],d['pose'],'TaskController / predicted H4')]
    handles=[]
    for ax,(_,_,name) in zip(axes,series):
        ax.set_title(name);ax.set_box_aspect((1,1,1));ax.view_init(25,-65)
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
        lines=[ax.plot([],[],[],color='#087f8c',lw=2)[0] for _ in bones]
        cloud=ax.scatter([],[],[],s=6,c='#e99326');handles.append((lines,cloud))
    title=fig.suptitle('');fig.tight_layout(rect=(0,0,1,.85))
    last=result['last_tick'];center=None;radius=None
    writer=FFMpegWriter(fps=30,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p'])
    path=out/'scheduler-comparison.mp4'
    if path.exists():raise ValueError('refuse video overwrite')
    with writer.saving(fig,str(path),100):
        for frame in range(result['recording_frames']):
            a=min(frame,last)
            fixed=(d['actual_hand'][frame],d['actual_pose'][frame]) if frame<4 else (d['fixed_hand'][frame],d['fixed_pose'][frame])
            # A labeled frozen terminal state retains the full source clock.
            states=[(d['actual_hand'][frame],d['actual_pose'][frame]),
                fixed,(d['hand'][a],d['pose'][a])]
            pts=np.concatenate([np.r_[h,p[None,:3,3]] for h,p in states])
            target=(pts.min(0)+pts.max(0))/2;size=max(.12,float((pts.max(0)-pts.min(0)).max()/2)*1.2)
            center=target if center is None else .15*target+.85*center
            radius=size if radius is None else max(size,.15*size+.85*radius)
            for ax,(h,p),(lines,cloud) in zip(axes,states,handles):
                for line,(first,end) in zip(lines,bones):
                    q=h[[first,end]];line.set_data_3d(q[:,0],q[:,1],q[:,2])
                q=d['flow_query']@p[:3,:3].T+p[:3,3];cloud._offsets3d=(q[:,0],q[:,1],q[:,2])
                ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            row=trace[min(frame,len(trace)-1)]
            status=('STOP: '+str(result['stop']['reason'])+' / frozen display') if frame>last else row['phase']
            title.set_text('t=%.2fs | WAM24/8 | %s\nMotion-only generated feedback / raw points / no physical execution'%(frame/30,status))
            writer.grab_frame()
    plt.close(fig)
    (out/'visualization.json').write_text(json.dumps(dict(status='COMPLETED',video=str(path),
        plot='scheduler-comparison.png',frames=result['recording_frames'],hz=30,
        tail='explicitly labeled frozen state after scheduler termination'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('infer','visualize'),required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--geometry-cache',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=7)
    args=parser.parse_args()
    if args.stage=='infer':infer(args)
    else:visualize(args)


if __name__=='__main__':main()
