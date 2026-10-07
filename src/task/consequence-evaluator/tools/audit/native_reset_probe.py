"""Bounded real-PhysX regression probe for the first reset/physics transition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reset-mode', choices=['native', 'batched'], default='native')
    a = parser.parse_args()
    output = a.output.resolve()
    if not output.is_relative_to(ROOT/'outputs/consequence-evaluator') or output.exists():
        parser.error('fresh task-owned output required')
    trained = json.loads((a.run_dir/'run_manifest.json').read_text())
    config = json.loads((a.run_dir/'config.json').read_text())
    checkpoint = Path(trained['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != trained['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                 '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
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
        import torch
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
                self.env_reset()
                before = task._target_states.clone()
                initial_q = task._dof_pos.clone()
                initial_body = task._rigid_body_pos.clone()
                initial_progress = task.progress_buf.clone()
                action = inspire_reference_action(task, 1)
                self.env_step(self.env, action)
                after = task._target_states.clone()
                displacement = (after[:,:3]-before[:,:3]).norm(dim=-1)
                report = dict(before=before.cpu().tolist(), after=after.cpu().tolist(),
                              displacement_m=displacement.cpu().tolist(),
                              initial_progress=initial_progress.cpu().tolist(),
                              post_progress=task.progress_buf.cpu().tolist(),
                              initial_q=initial_q.cpu().tolist(),
                              initial_body=initial_body.cpu().tolist(),
                              post_body=task._rigid_body_pos.cpu().tolist(),
                              pass_reset_persistence=bool((displacement < .1).all()),
                              limitation='first native step only; no grasp or policy qualification')
                (output/'reset_check.json').write_text(json.dumps(report, indent=2)+'\n')
                print('RESET_PROBE '+json.dumps({k:report[k] for k in
                      ('displacement_m','pass_reset_persistence')}), flush=True)
                if not report['pass_reset_persistence']:
                    raise AssertionError('reset object teleported on first PhysX step')
        native.EvalPlayer = ResetProbePlayer
        sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', config['cfg_env'],
                    '--cfg_train', str(native_root/'data/cfg/train/rlg/inspire.yaml'),
                    '--motion_file', config['motion_root'], '--checkpoint', str(checkpoint),
                    '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                    '--pipeline', 'gpu', '--graphics_device_id', '0', '--num_envs', '8',
                    '--seed', '17', '--output_path', str(output/'native-runtime')]
        manifest.update(status='RUNNING', command=sys.argv[1:])
        save()
        os.chdir(ROOT/'third_party/DExplore')
        native.main()
        manifest.update(status='COMPLETED')
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == '__main__':
    main()
