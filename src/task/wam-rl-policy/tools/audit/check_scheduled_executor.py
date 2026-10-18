"""One recorded-state GPU forward through scheduler, v4 and BC actor.

No simulator action is executed and no policy is trained. Input NPZ contains
seven observed 60Hz rows: object[7,7], hand[7,11,3], time[7], observation[7,300],
and goal[7,3]. Identity JSON identifies exactly one cached UniDex object.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from wam_rl_policy.goal_scheduler import OrderedGoalScheduler
from wam_rl_policy.scheduled_executor import BCWarmStartActor, ScheduledV4ExecutorInputs
from wam_rl_policy.wam_v4_runtime import V4Runtime, object_poses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'identity', 'cache', 'wam', 'bc', 'dense_manifest', 'output'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('real model forward requires CUDA')
    torch.set_num_threads(2)
    start = time.monotonic()
    with np.load(args.input, allow_pickle=False) as packet:
        data = {k:packet[k] for k in packet.files}
    identity = json.loads(args.identity.read_text())
    if identity.get('split') != 'train':
        raise ValueError('engineering smoke must use TRAIN input')
    device = torch.device('cuda', args.gpu)
    runtime = V4Runtime(args.wam, [identity], args.cache,
                        dense_manifest=args.dense_manifest, device=str(device))
    poses = object_poses(data['object'])
    target = poses[0].copy()
    target[:3, 3] = data['goal'][0]
    scheduler = OrderedGoalScheduler([poses[0], target], [0., 2.], runtime.geometry[0][2],
                                     interaction_points=runtime.geometry[0][0])
    # Observe startup measurements at their actual cadence. The reference
    # comes from the task goal, never from future recorded object motion.
    for tick in range(6):
        scheduler.update(poses[tick], float(data['time'][tick]), hand_points=data['hand'][tick])
    current = data['observation'][-1]
    p2 = np.concatenate([current[a:b] for a,b in ((0,44),(66,131),(161,191))])
    bridge = ScheduledV4ExecutorInputs(runtime, scheduler)
    torch.cuda.synchronize(device)
    begin = time.monotonic()
    result = bridge.observation(data['object'], data['hand'], data['time'], poses[-1], p2)
    torch.cuda.synchronize(device)
    forward_seconds = time.monotonic()-begin
    bc = torch.load(args.bc, map_location='cpu', weights_only=False)
    actor = BCWarmStartActor(bc).to(device).eval()
    with torch.inference_mode():
        observation = torch.as_tensor(result['observation'][None], device=device)
        mean = actor(observation)
        command = actor.act_inference(observation)
    if command.shape != (1,24) or not torch.isfinite(command).all():
        raise ValueError('nonfinite or malformed executor action')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'prediction.npz', observation=result['observation'],
        tau=result['tau'], interaction=result['interaction'], interaction_valid=result['interaction_valid'],
        mean=mean.cpu().numpy(), command=command.cpu().numpy())
    report = dict(status='PASS', scope='one archived TRAIN state; no physical action or success claim',
        wam_step=runtime.step, wam=str(args.wam), bc=str(args.bc), identity=identity,
        python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device), forward_seconds=forward_seconds,
        elapsed_seconds=time.monotonic()-start, peak_mib=torch.cuda.max_memory_allocated(device)/2**20,
        tau_shape=list(result['tau'].shape), interaction_shape=list(result['interaction'].shape),
        action_shape=list(command.shape), clipped_channels=int((mean.abs()>1).sum()),
        scheduler_mode=result['scheduler']['mode'], reference='initial pose to task goal; fixed initial rotation')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
