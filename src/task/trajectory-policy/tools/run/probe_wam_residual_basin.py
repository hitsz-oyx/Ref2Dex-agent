"""Four frozen finger interventions on the exact WAM+R base; no PPO updates."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'tools/run'), str(TASK / 'src'),
               str(ROOT / 'src/task/consequence-evaluator/src'),
               '/home2/wyy/isaac-gym/isaacgym/python']
import train_wam_residual_ppo as runner

EXP = 'P-20261014-wam-residual-basin'
NAMES = ('base', 'thumb_pitch_minus', 'thumb_pitch_plus', 'finger_close')
START, STOP = 40, 160


def biases():
    value = np.zeros((4, 12), np.float32)
    value[1, 11] = -4.
    value[2, 11] = 4.
    value[3, 6:] = 2.
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    scratch = ROOT / 'tmp' / out.name
    address = scratch / 'w.sock'
    if (out.exists() or scratch.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents
            or not out.name.startswith('wam-ref2-execution-') or len(os.fsencode(str(address))) >= 108
            or Path(sys.prefix).resolve() != runner.RUNTIME.resolve()):
        raise ValueError('fresh owned output and pinned native runtime required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU occupied before frozen intervention')
    # Parent supplies only the measured reset prefix and provenance. The
    # learned residual mean is bypassed: candidate0 is the exact zero residual.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    from isaacgym import gymtorch
    import torch
    from trajectory_policy import residual_ppo
    del gymtorch
    phases = runner.curriculum_phases('contact', 4)
    diagnostic_controls = ((len(runner.diagnostic_phases(phases)) + 3) // 4) * runner.DIAGNOSTIC

    class FixedActor(residual_ppo.ResidualActor):
        def __init__(self):
            super().__init__()
            self.control_calls = 0

        def distribution(self, obs):
            tick = self.control_calls - diagnostic_controls
            self.control_calls += 1
            delta = torch.as_tensor(biases(), device=obs.device)
            if not START <= tick < STOP:
                delta = torch.zeros_like(delta)
            # No stochastic sample is executed, and this distribution is never
            # passed to PPO. Its location records the exact frozen intervention.
            return torch.distributions.Normal(delta, self.log_std.exp().expand_as(delta))

    residual_ppo.ResidualActor = FixedActor
    runner.DEADLINE_S = 900
    scratch.mkdir(mode=0o700)
    out.mkdir()
    controls = diagnostic_controls + runner.EVAL
    args = SimpleNamespace(gpu=a.gpu, envs=4, updates=0, resume=a.parent.resolve(),
                           curriculum='contact', allow_curriculum_change=False, eval_every=0,
                           experiment_id=EXP, rpcs=controls, required_contact_phases=(0,), required_contact_envs=(0,))
    log = (out / 'server.log').open('w')
    server = subprocess.Popen(
        [runner.GRASPENV, str(TASK / 'tools/run/wam_native_server.py'), '--gpu', str(a.gpu),
         '--controls', str(controls), '--experiment-id', EXP, '--head-dir', str(runner.HEAD),
         '--socket', str(address), '--output', str(out / 'server'), '--chunk-dir', str(runner.CHUNK),
         '--effect-checkpoint', str(runner.EFFECT), '--control-checkpoint', str(runner.CONTROL), '--open-queries'],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(240):
            if server.poll() is not None:
                raise RuntimeError('inference server exited before READY')
            ready = out / 'server/status.json'
            if ready.exists() and json.loads(ready.read_text())['status'] == 'READY':
                break
            time.sleep(.5)
        else:
            raise TimeoutError('server not ready')
        runner.run_native(args, out, address, server)
        server.wait(timeout=60)
        if server.returncode:
            raise RuntimeError('server exit failure')
        with np.load(out / 'eval.npz') as packet:
            expected = np.zeros_like(packet['delta'])
            expected[START:STOP] = biases()[None]
            np.testing.assert_array_equal(packet['delta'], expected)
        report = json.loads((out / 'result.json').read_text())
        report.update(frozen_probe=True, learned_residual_bypassed=True, updates_executed=0,
                      candidate_names=list(NAMES), fixed_delta=biases().tolist(), window=[START, STOP])
        for row, name in zip(report['outcomes'], NAMES):
            row['candidate'] = name
        assert report['actor_parameter_change'] == 0.
        runner.write(out / 'result.json', report)
        manifest = json.loads((out / 'manifest.json').read_text())
        manifest.update(frozen_probe=True, learned_residual_bypassed=True, candidate_names=list(NAMES),
                        window=[START, STOP], fixed_delta=biases().tolist(),
                        result_sha256=runner.sha(out / 'result.json'))
        runner.write(out / 'manifest.json', manifest)
        print(json.dumps(report), flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        log.close()


if __name__ == '__main__':
    main()
