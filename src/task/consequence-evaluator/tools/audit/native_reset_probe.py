"""Bounded real-PhysX regression probe for the first reset/physics transition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.contracts import is_within


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reset-mode', choices=['native', 'batched'], default='native')
    parser.add_argument('--motion-root', type=Path, help='explicit reference-only diagnostic input override')
    parser.add_argument('--geometry-steps', type=int, default=0,
                        help='At most128 frozen-policy steps to check measured hand/object geometry')
    a = parser.parse_args()
    if not 0 <= a.geometry_steps <= 128:
        parser.error('geometry engineering check is bounded to128 steps')
    output = a.output.resolve()
    if not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists():
        parser.error('fresh task-owned output required')
    trained = json.loads((a.run_dir/'run_manifest.json').read_text())
    config = json.loads((a.run_dir/'config.json').read_text())
    motions = a.motion_root.resolve() if a.motion_root else Path(config['motion_root'])
    checkpoint = Path(trained['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != trained['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                 '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('free disk below20GiB reserve')
    sources = [Path(__file__), a.run_dir/'config.json', a.run_dir/'run_manifest.json',
               Path(config['cfg_env']), checkpoint,
               *sorted(motions.glob('*/interaction_hand_inspire.pt')),
               *sorted((TASK/'src/consequence_evaluator').glob('*.py'))]
    if a.geometry_steps:
        sources += [ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',
                    ROOT/'src/task/CmResidual/v118_planner.py',
                    ROOT/'third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py']
        assets=ROOT/'third_party/DExplore/dexplore/data/assets'
        sources += [p for p in assets.rglob('*') if p.is_file()]
    frozen = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output.mkdir(parents=True)
    scratch = ROOT/'tmp/consequence-reset-probe'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch/'torch-extensions'),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    native_root = ROOT/'third_party/DExplore/dexplore'
    sys.path[:0] = [str(TASK/'src'), str(native_root), str(ROOT),
                   str(ROOT/'src/task/CmResidual/tools')]
    manifest = dict(status='INITIALIZING', run_id=output.name, pid=os.getpid(),
                    physical_gpu=a.gpu, seed=17, reset_mode=a.reset_mode,
                    seconds_budget=180, output_budget_bytes=2**20,
                    sources=frozen, motion_root=str(motions),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    started = time.monotonic()
    def save():
        manifest['elapsed_s'] = time.monotonic()-started
        (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    def deadline(signum, frame):
        raise TimeoutError('180s reset probe deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    try:
        import dexplore_ddp_rank_bootstrap
        import evaluate as native
        from env.tasks.base_dexplore_task import DexploreTask
        from utils.reference_action import inspire_reference_action
        from consequence_evaluator.reset_kinematics import task_kinematics
        import torch
        if a.geometry_steps:
            from consequence_evaluator.physical_geometry import PhysicalGeometry
        if a.reset_mode == 'batched':
            from consequence_evaluator.native_reset import install_reset_patch
            install_reset_patch()
        base = native.EvalPlayer
        class ResetProbePlayer(base):
            def run(self):
                task = self.env.task
                task._state_init = DexploreTask.StateInit.Start
                task._hybrid_init_prob = 1.
                task._adaptive_kappa_enabled = False
                task._enable_early_termination = False
                obs=self.env_reset()
                if self.get_batch_size(obs['obs'],1)!=task.num_envs:
                    raise AssertionError('native player batch initialization failed')
                before = task._target_states.clone()
                initial_q = task._dof_pos.clone()
                initial_body = task._rigid_body_pos.clone()
                initial_progress = task.progress_buf.clone()
                action = inspire_reference_action(task, 1)
                obs,_,_,_=self.env_step(self.env, action)
                after = task._target_states.clone()
                displacement = (after[:,:3]-before[:,:3]).norm(dim=-1)
                fk = task_kinematics(task)
                computed = fk.states(task._dof_pos, task._dof_vel, task._humanoid_root_states)
                bodies = task._rigid_body_state.view(task.num_envs,-1,13)[:,:task.num_bodies].clone()
                fk_position_error = (computed[:,:,:3]-bodies[:,:,:3]).norm(dim=-1).max().item()
                fk_rotation_error = torch.minimum(
                    (computed[:,:,3:7]-bodies[:,:,3:7]).norm(dim=-1),
                    (computed[:,:,3:7]+bodies[:,:,3:7]).norm(dim=-1)).max().item()
                report = dict(before=before.cpu().tolist(), after=after.cpu().tolist(),
                              displacement_m=displacement.cpu().tolist(),
                              initial_progress=initial_progress.cpu().tolist(),
                              post_progress=task.progress_buf.cpu().tolist(),
                              initial_q=initial_q.cpu().tolist(),
                              initial_body=initial_body.cpu().tolist(),
                              post_body=task._rigid_body_pos.cpu().tolist(),
                              fk_position_error_m=fk_position_error,
                              fk_quaternion_error=fk_rotation_error,
                              fk_velocity_error_m_s=(computed[:,:,7:10]-bodies[:,:,7:10]).norm(dim=-1).max().item(),
                              fk_angular_velocity_error_rad_s=(computed[:,:,10:13]-bodies[:,:,10:13]).norm(dim=-1).max().item(),
                              pass_reset_persistence=bool((displacement < .1).all()),
                              limitation='first native step only; no grasp or policy qualification')
                (output/'reset_check.json').write_text(json.dumps(report, indent=2)+'\n')
                print('RESET_PROBE '+json.dumps({k:report[k] for k in
                      ('displacement_m','pass_reset_persistence')}), flush=True)
                if not report['pass_reset_persistence']:
                    raise AssertionError('reset object teleported on first PhysX step')
                if fk_position_error > .0005 or fk_rotation_error > .0005:
                    raise AssertionError('URDF reset FK disagrees with measured native body poses')
                if (report['fk_velocity_error_m_s'] > .001
                        or report['fk_angular_velocity_error_rad_s'] > .005):
                    raise AssertionError('reset FK velocity disagrees with measured PhysX state')
                if a.geometry_steps:
                    geometry=PhysicalGeometry(task,assets)
                    gaps, proxies, heights = [], [], []
                    for _ in range(a.geometry_steps):
                        if not isinstance(obs,dict):
                            obs={'obs':obs}
                        points,gap=geometry.measure(task)
                        if (points.shape!=(task.num_envs,11,3) or not torch.isfinite(points).all()
                                or not torch.isfinite(gap).all() or bool((gap<0).any())):
                            raise AssertionError('invalid measured hand/object geometry')
                        force=(task._contact_forces[:,task._contact_body_ids].norm(dim=-1)>.1).any(-1)
                        force &= task._tar_contact_forces.norm(dim=-1)>.1
                        gaps.append(gap.cpu());proxies.append(force.cpu())
                        heights.append((task._target_states[:,2]-before[:,2]).cpu())
                        obs,_,done,_=self.env_step(self.env,self.get_action(obs,True))
                        if bool(done.any()):
                            raise AssertionError('geometry smoke crossed first episode boundary')
                    gap=torch.stack(gaps);proxy=torch.stack(proxies);height=torch.stack(heights)
                    report['geometry_check']=dict(steps=a.geometry_steps,frames=gap.numel(),
                        gap_min_m=float(gap.min()),gap_median_m=float(gap.median()),
                        force_proxy_frames=int(proxy.sum()),
                        force_and_near_frames=int((proxy&(gap<=.01)).sum()),
                        force_far_frames=int((proxy&(gap>.03)).sum()),
                        elevated_force_frames=int((proxy&(height>=.03)).sum()),
                        elevated_force_near_frames=int((proxy&(height>=.03)&(gap<=.01)).sum()),
                        limitation='bounded frozen-policy geometry smoke; sampled unsigned proximity, no pairwise contact GT')
                    (output/'reset_check.json').write_text(json.dumps(report,indent=2)+'\n')
                if a.reset_mode == 'batched':
                    subset_checks=[]
                    for offset in (0,1,0):
                        ids=torch.arange(offset,task.num_envs,2,device=task._dof_pos.device)
                        untouched=torch.ones(task.num_envs,dtype=torch.bool,device=ids.device)
                        untouched[ids]=False
                        roots_before=task._root_states.clone()
                        body_before=task._rigid_body_state.clone()
                        progress_before=task.progress_buf.clone()
                        self.env_reset(ids)
                        expected=fk.states(task._dof_pos[ids],task._dof_vel[ids],
                                           task._humanoid_root_states[ids])
                        cached=task._rigid_body_state.view(task.num_envs,-1,13)
                        roots=task._root_states.view(task.num_envs,-1,13)
                        if (not torch.equal(roots[untouched],roots_before.view_as(roots)[untouched])
                                or not torch.equal(cached[untouched],body_before.view_as(cached)[untouched])
                                or not torch.equal(task.progress_buf[untouched],progress_before[untouched])
                                or not torch.allclose(cached[ids,:task.num_bodies],expected,atol=1e-6)):
                            raise AssertionError('subset reset changed unrelated state or has stale FK')
                        target=task._target_states[ids].clone()
                        self.env_step(self.env,inspire_reference_action(task,1))
                        moved=(task._target_states[ids,:3]-target[:,:3]).norm(dim=-1)
                        subset_checks.append(dict(ids=ids.cpu().tolist(),displacement_m=moved.cpu().tolist()))
                        if not bool((moved < .1).all()):
                            raise AssertionError('repeated subset reset lost target state')
                    report['subset_checks']=subset_checks
                    (output/'reset_check.json').write_text(json.dumps(report,indent=2)+'\n')
        native.EvalPlayer = ResetProbePlayer
        sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', config['cfg_env'],
                    '--cfg_train', str(native_root/'data/cfg/train/rlg/inspire.yaml'),
                    '--motion_file', str(motions), '--checkpoint', str(checkpoint),
                    '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                    '--pipeline', 'gpu', '--graphics_device_id', '0', '--num_envs', '8',
                    '--seed', '17', '--output_path', str(output/'native-runtime')]
        manifest.update(status='RUNNING', command=sys.argv[1:])
        save()
        os.chdir(ROOT/'third_party/DExplore')
        native.main()
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in frozen.items()):
            raise RuntimeError('reset probe input/source drift')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 2**20:
            raise RuntimeError('reset probe exceeded1MiB output budget')
        manifest.update(status='COMPLETED')
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == '__main__':
    main()
