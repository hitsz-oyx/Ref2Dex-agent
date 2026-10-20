"""Measure full-DEV Current421 imitation in physical wrist PD-target units."""
import argparse
import json
import os
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/run')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True, help='uniform/startup_half checkpoint parent')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=7)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('fresh audit output required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import numpy as np
    import torch
    from train_measured_scheduled_bc import load
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.bc_sampling import startup_rows

    torch.set_num_threads(2)
    data = load(a.manifest.resolve(), 'dev', workers=4)
    tensors = {k: torch.as_tensor(data[k], device='cuda') for k in ('measured', 'task', 'points', 'object_id')}
    early = np.zeros(len(data['action']), bool)
    early[startup_rows(data['ticks'], 30)] = True
    report = dict(source_manifest=str(a.manifest.resolve()), python=sys.executable,
                  gpu=a.gpu, scope='full DEV teacher-state imitation; no robot success claim',
                  physical_contract='translation PD delta=action_xyz metres; rotation Euler joint deltas=pi*action',
                  rows=len(early), startup_rows=int(early.sum()), checkpoints={})
    for mode in ('uniform', 'startup_half'):
        for endpoint in ('best', 'latest'):
            path = a.run/mode/(endpoint+'.pt')
            packet = torch.load(path.resolve(), map_location='cpu', weights_only=False)
            model = MeasuredInspirePolicy(with_tau=False).cuda().eval()
            model.load_state_dict(packet['model'], strict=True)
            with torch.inference_mode():
                prediction = model(tensors['measured'], tensors['task'], tensors['points'],
                                   shape_index=tensors['object_id']).cpu().numpy()
            error = prediction-data['action']
            groups = {}
            for name, mask in [('all', np.ones(len(early), bool)), ('startup', early), ('rest', ~early)]:
                e = error[mask]
                position = np.linalg.norm(e[:, :3], axis=-1)
                groups[name] = dict(rows=int(mask.sum()), command_mae=float(np.abs(e).mean()),
                    wrist_translation_pd_l2_mean_m=float(position.mean()),
                    wrist_translation_pd_l2_p95_m=float(np.quantile(position, .95)),
                    rotation_joint_delta_l2_mean_rad=float((np.linalg.norm(e[:, 3:6], axis=-1)*np.pi).mean()),
                    finger_command_mae=float(np.abs(e[:, 6:]).mean()))
            report['checkpoints'][mode+'/'+endpoint] = dict(checkpoint=str(path.resolve()),
                step=int(packet['step']), groups=groups)
            del model
            torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
