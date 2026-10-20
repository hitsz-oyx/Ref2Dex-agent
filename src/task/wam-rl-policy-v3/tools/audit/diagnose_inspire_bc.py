"""Compare teacher-state imitation and native BC drift on a fixed TRAIN lift panel.

Panel selection uses only admitted teacher coupling coverage. Physical lift is
reported separately from TaskController completion; neither is a DEV success rate.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT', TASK.parents[2])).resolve()
PANEL = ('s1_toothpaste_lift', 's1_stamp_lift', 's1_waterbottle_lift', 's1_apple_lift')


def write(path, packet):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(packet, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def longest_true(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def height_metrics(height):
    import numpy as np
    lifted = np.asarray(height) - height[0]
    return dict(max_lift_m=float(lifted.max()), final_lift_m=float(lifted[-1]),
                longest_lift5cm_frames=longest_true(lifted >= .05))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--geometry-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--controls', type=int, default=300)
    p.add_argument('--replay-teacher', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if (out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents
            or not 100 <= a.controls <= 1200 or shutil.disk_usage(ROOT).free < 20*2**30):
        raise ValueError('fresh bounded task output, controls and disk reserve required')
    out.mkdir(parents=True)
    scratch = ROOT/'tmp'/('inspire-diag-' + str(os.getpid()))
    scratch.mkdir(mode=0o700, parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(TASK/'src'))
    import numpy as np
    import torch
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.inspire_shadow import ACTIVE

    torch.set_num_threads(2)
    packet = torch.load(a.checkpoint.resolve(), map_location='cpu', weights_only=False)
    if packet.get('schema') != 'ref2dex.inspire-measured-current-bc.v1':
        raise ValueError('Current421 checkpoint required')
    model = MeasuredInspirePolicy(with_tau=False).cuda().eval()
    model.load_state_dict(packet['model'], strict=True)
    corpus = TeacherRolloutCorpus(a.manifest.resolve(), 'train')
    indices = {r['sequence']: i for i, r in enumerate(corpus.records)}
    report = dict(status='RUNNING', task='wam-rl-policy-v3',
                  experiment_id='P-20261019-inspire-task-contract', run_id=out.name,
                  git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  python=sys.executable, torch=torch.__version__, gpu=a.gpu,
                  checkpoint=str(a.checkpoint.resolve()), checkpoint_step=int(packet['step']),
                  source_manifest=str(a.manifest.resolve()), panel=list(PANEL),
                  seed=246, controls=a.controls, optimizer_steps=0, cases={}, processes=[])
    write(out/'manifest.json', report)
    episodes = {}
    for name in PANEL:
        episode = corpus.episode(indices[name]); row = corpus.records[indices[name]]
        episodes[name] = episode
        with np.load(Path(row['source'])/'trajectory.npz') as z:
            raw = {k: z[k].copy() for k in ('q', 'hand', 'obj', 'action')}
        episode['raw'] = raw
        with torch.inference_mode():
            prediction = model(torch.as_tensor(episode['measured'], device='cuda'),
                               torch.as_tensor(episode['task'], device='cuda'),
                               torch.as_tensor(episode['object_points'][None], device='cuda'),
                               shape_index=torch.zeros(len(episode['action']), dtype=torch.long,
                                                       device='cuda')).cpu().numpy()
        error = np.abs(prediction - episode['action'])
        metrics = {}
        for count in (4, 10, 30, 100, a.controls, len(error)):
            keep = episode['active'][:count]
            if keep.any():
                metrics[str(count)] = dict(rows=int(keep.sum()),
                    command_mae=float(error[:count][keep].mean()),
                    root_mae=float(error[:count][keep, :6].mean()),
                    finger_mae=float(error[:count][keep, 6:].mean()))
        report['cases'][name] = dict(admitted_commands=len(error), teacher_forced=metrics,
            teacher_height_first_controls=height_metrics(raw['obj'][:a.controls+1, 2, 3]),
            teacher_height_admitted=height_metrics(raw['obj'][:len(error)+1, 2, 3]))
        np.savez_compressed(out/(name+'-teacher-forced.npz'), prediction=prediction,
                            target=episode['action'], active=episode['active'], ticks=episode['ticks'])
    write(out/'manifest.json', report)
    # Release inference VRAM before starting the native host and its policy peer.
    del model
    torch.cuda.empty_cache()
    native_python = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'
    py = '/home2/wyy/miniconda3/envs/graspenv/bin/python'
    env = dict(os.environ, REF2DEX_RUN_ROOT=str(ROOT), REF2DEX_CODE_COMMIT=report['git_commit'])
    children = []
    logs = []

    def start(name, args):
        log = (out/(name+'.log')).open('x'); logs.append(log)
        proc = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc)
        report['processes'].append(dict(name=name, pid=proc.pid, argv=proc.args))
        write(out/'manifest.json', report)
        return proc

    def finish(proc, seconds):
        code = proc.wait(timeout=seconds)
        if code:
            raise RuntimeError('child exit code '+str(code)+': '+str(proc.args[2]))

    try:
        if a.replay_teacher:
            for name in PANEL:
                proc = start('replay-'+name, [py, '-u', TASK/'tools/audit/replay_inspire_commands.py',
                    '--cache', a.geometry_cache.resolve(), '--audit-manifest', a.manifest.resolve(),
                    '--sequence', name, '--output', out/('replay-'+name), '--gpu', a.gpu,
                    '--match-collection-batch', '--controls', 200])
                finish(proc, 650)
                report['cases'][name]['canonical_replay'] = json.loads(
                    (out/('replay-'+name)/'report.json').read_text())
                # Per-frame replay errors remain in their own bounded artifact.
                report['cases'][name]['canonical_replay'].pop('frames', None)
                write(out/'manifest.json', report)
        panel_file = out/'panel.json'
        write(panel_file, dict(sequences=list(PANEL)))
        sock = scratch/'s'
        policy = start('policy', [py, '-u', TASK/'tools/run/inspire_measured_bc_eval.py',
            '--manifest', a.manifest.resolve(), '--checkpoint', a.checkpoint.resolve(),
            '--output', out/'closed-loop', '--socket', sock, '--gpu', a.gpu,
            '--controls', a.controls, '--trace-controls', a.controls,
            '--seed', 246, '--seconds', 900, '--state-transport', 'packed'])
        deadline = time.monotonic()+120
        while not sock.exists():
            if policy.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('BC policy did not open socket')
            time.sleep(.2)
        host = start('native', [native_python, '-u', TASK/'tools/run/inspire_physics_host.py',
            '--cache', a.geometry_cache.resolve(), '--output', out/'closed-loop/native',
            '--socket', sock, '--server-pid', policy.pid, '--gpu', a.gpu,
            '--envs', len(PANEL), '--fixed-motions-json', panel_file, '--seed', 246,
            '--seconds', 900, '--state-transport', 'packed', '--extended-state'])
        finish(policy, 920); finish(host, 30)
        with np.load(out/'closed-loop/first100-decisions.npz') as z:
            trace = {k: z[k].copy() for k in z.files}
        for i, name in enumerate(PANEL):
            episode = episodes[name]; raw = episode['raw']
            ticks = trace['tick'][:, i].astype(int)
            # Exclude the final unexecuted bootstrap decision and out-of-prefix ticks.
            mask = trace['applied'] & (ticks < len(episode['action']))
            teacher_action = raw['action'][ticks[mask]][:, ACTIVE]
            action_error = np.abs(trace['command'][mask, i] - teacher_action)
            q_error = np.abs(trace['q'][mask, i] - raw['q'][ticks[mask]])
            object_error = np.linalg.norm(trace['objects'][mask, i, :3] - raw['obj'][ticks[mask], :3, 3], axis=-1)
            hand_error = np.linalg.norm(trace['hand'][mask, i] - raw['hand'][ticks[mask]], axis=-1).mean(-1)
            report['cases'][name]['closed_loop'] = dict(
                initial_q_max=float(q_error[0].max()), initial_hand_mean_m=float(hand_error[0]),
                initial_object_position_m=float(object_error[0]),
                height=height_metrics(trace['objects'][:, i, 2]),
                prefixes={str(n): dict(rows=min(n, len(action_error)),
                    time_aligned_command_mae=float(action_error[:n].mean()),
                    q_mae=float(q_error[:n].mean()), object_error_m=float(object_error[:n].mean()),
                    hand_error_m=float(hand_error[:n].mean())) for n in (4, 10, 30, 100, a.controls)})
        report.update(status='COMPLETED',
            native_manifest=json.loads((out/'closed-loop/native/manifest.json').read_text()),
            task_result=json.loads((out/'closed-loop/manifest.json').read_text()),
            scope='four TRAIN lift motions selected by teacher coverage; no general success claim')
    except BaseException as e:
        report.update(status='FAILED', error=repr(e)); raise
    finally:
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for log in logs: log.close()
        write(out/'manifest.json', report)


if __name__ == '__main__':
    main()
