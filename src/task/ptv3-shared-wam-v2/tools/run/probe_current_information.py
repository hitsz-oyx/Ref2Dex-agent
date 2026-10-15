"""Matched current-reference/current-geometry information Probe for q_PD R."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from current_information import ARMS, InformationRetargeter, pack
from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED, freeze_chunk, consume_target
from ptv3_shared_wam.checkpoint import atomic_json, git_commit
from trajectory_policy.pd_retargeter import pd_from_action
from trajectory_policy.retargeter_data import ACTIVE

ROOT = Path(__file__).resolve().parents[5]


@torch.no_grad()
def evaluate(model, data, indices, arm, controller, records):
    model.eval()
    predictions = torch.cat([model(data['state'][ids], data['tau'][ids], data['extra'][ids], arm)
                             for ids in indices.split(128)])
    state, labels = data['state'][indices], data['pd'][indices]
    absolute = freeze_chunk(predictions, state[:, :18], GEOMETRY_LIFTED)[:, 0]
    _, applied, clipped = consume_target(absolute, state[:, :18], GEOMETRY_LIFTED, controller)
    actual = pd_from_action(applied, state[:, :18], state.new_tensor(controller['offset']),
                            state.new_tensor(controller['scale']))
    delta = (actual-labels)[:, list(ACTIVE)].cpu().numpy()
    xyz = np.linalg.norm(delta[:, :3], axis=-1)*1000
    tick = data['tick'][indices].cpu().numpy()
    seq = data['sequence'][indices].cpu().numpy()
    result = dict(xyz_mean_mm=float(xyz.mean()), xyz_p95_mm=float(np.percentile(xyz, 95)),
                  rotation_p95_rad=float(np.percentile(abs(delta[:, 3:6]).max(-1), 95)),
                  finger_p95_rad=float(np.percentile(abs(delta[:, 6:]).max(-1), 95)),
                  clipping=float(clipped.float().mean()),
                  normalized_all8_L1=float(((predictions-data['target'][indices])/model.base.action_scale).abs().mean()))
    for label, mask in [('startup', tick < 16), ('later', tick >= 16)]:
        result[label] = dict(count=int(mask.sum()), mean_mm=float(xyz[mask].mean()) if mask.any() else None,
                             p95_mm=float(np.percentile(xyz[mask], 95)) if mask.any() else None)
    result['objects'] = {}
    objects = np.asarray([records[int(s)]['object'] for s in seq])
    for obj in sorted(set(objects)):
        e = xyz[objects == obj]
        result['objects'][obj] = dict(count=len(e), mean_mm=float(e.mean()), p95_mm=float(np.percentile(e, 95)))
    model.train()
    return result, predictions.cpu().numpy()


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--card', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--updates', type=int, default=3000)
    a = p.parse_args()
    out = a.output.resolve()
    if ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents or (out/'manifest.json').exists():
        raise ValueError('fresh Task output with retained initial_R.pt required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    import yaml
    card = yaml.safe_load(a.card.read_text().split('---', 2)[1])
    if card.get('task') != 'ptv3-shared-wam-v2' or card.get('seeds') != [160, 161, 162]:
        raise ValueError('assigned Task Probe identity required')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    scratch = ROOT/'tmp'/out.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('bounded900s current-information Probe')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    initial = torch.load(out/'initial_R.pt', map_location='cpu', weights_only=False)
    config = initial['config']
    source = Path(initial['parent_provenance']['source'])
    workspace = dict(schema=config['encoder_workspace_admission'], origin_m=config['voxel_origin_m'],
                     voxel_m=config['voxel_m'])
    datasets = {split: V2RLRobotWindows(source, split, 32, GEOMETRY_LIFTED, workspace)
                for split in ['train', 'val']}
    manifest = dict(status='RUNNING', experiment_id=card['experiment_id'], run_id=out.name,
                    git_commit=git_commit(), physical_gpu=a.gpu, parent_step=initial['parent_step'],
                    parent_checkpoint=initial['parent_checkpoint'], source=str(source),
                    updates=a.updates, batch=128, learning_rate=3e-4, seeds=[160, 161, 162],
                    arms=ARMS, tau='same measured GT future24 for every arm; oracle R-only diagnostic',
                    added_input_time='t only; current references permitted; no t+1/t+16 reference',
                    scope='offline PD prediction only; main WAM training untouched',
                    limits=dict(seconds=900, artifacts_bytes=150_000_000, min_free_bytes=20*1024**3))
    atomic_json(out/'manifest.json', manifest)
    try:
        packed = {split: pack(d) for split, d in datasets.items()}
        extra_mean = packed['train']['extra'].mean(0)
        extra_scale = packed['train']['extra'].std(0).clip(1e-3)
        np.savez_compressed(out/'extra_train_stats.npz', mean=extra_mean, std=extra_scale)
        train_ids = datasets['train'].sample_indices(np.random.default_rng(162), 512)
        # Retain the exact existing scientific dev panel (dev membership is unchanged).
        dev_ids = np.load(Path(initial['parent_checkpoint']).parent/'validation.npy')
        np.savez_compressed(out/'panels.npz', TRAIN_indices=train_ids, dev_indices=dev_ids)
        atomic_json(out/'encoder_workspace_admission.json',
                    {s:d.workspace_admission for s,d in datasets.items()})
        data = {s:{k:torch.as_tensor(v, device='cuda') for k,v in values.items()}
                for s,values in packed.items()}
        del packed
        ids = dict(train=torch.as_tensor(train_ids, device='cuda'), val=torch.as_tensor(dev_ids, device='cuda'))
        torch.manual_seed(160)
        template = InformationRetargeter(initial['model'], extra_mean, extra_scale)
        initial_predictions = None
        results = {}
        for arm in ARMS:
            model = copy.deepcopy(template).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            rng = np.random.default_rng(161)
            curve, selected, best_key = [], None, None
            arm_started = time.monotonic()
            for step in range(a.updates+1):
                if step:
                    ix = torch.as_tensor(datasets['train'].sample_indices(rng, 128), device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(data['train']['state'][ix], data['train']['tau'][ix], data['train']['extra'][ix], arm)
                    loss = ((pred-data['train']['target'][ix])/model.base.action_scale).abs().mean()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise FloatingPointError('nonfinite matched information update')
                    optimizer.step()
                if step % 500 == 0 or step == a.updates:
                    row = dict(step=step, panels={}, elapsed_s=time.monotonic()-arm_started)
                    predictions = {}
                    for split in data:
                        row['panels'][split], predictions[split] = evaluate(model, data[split], ids[split],
                                                                          arm, datasets[split].controller, datasets[split].records)
                    if step == 0:
                        if initial_predictions is None:
                            initial_predictions = predictions['val']
                        else:
                            np.testing.assert_allclose(predictions['val'], initial_predictions, atol=1e-6, rtol=0)
                    curve.append(row)
                    metrics = row['panels']['val']
                    key = (metrics['xyz_p95_mm'], metrics['xyz_mean_mm'])
                    if best_key is None or key < best_key:
                        best_key, selected = key, copy.deepcopy(row)
                        checkpoint = out/('%s-step%d.pt' % (arm, step))
                        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                        arm=arm, step=step, schema='ref2dex.R-current-information.v1'), checkpoint)
                        np.savez_compressed(out/(arm+'-selected-predictions.npz'), **predictions)
                    atomic_json(out/(arm+'-curve.json'), curve)
                    gpu = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                         '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                    atomic_json(out/'progress.json', dict(arm=arm, step=step, elapsed_s=time.monotonic()-started,
                                                         gpu=gpu, loss=float(loss) if step else None))
                    print(json.dumps(dict(arm=arm, step=step, dev=metrics, elapsed_s=time.monotonic()-started)), flush=True)
                    consumers = set(map(int, subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()))
                    if consumers-{os.getpid()}:
                        raise RuntimeError('foreign GPU process')
                    if sum(f.stat().st_size for f in out.iterdir() if f.is_file()) > 150_000_000:
                        raise RuntimeError('150MB artifact budget exceeded')
                    import shutil
                    if shutil.disk_usage(ROOT).free < 20*1024**3:
                        raise RuntimeError('less than20GiB free disk')
            results[arm] = dict(initial=curve[0], selected=selected, final=curve[-1])
            atomic_json(out/'partial_result.json', results)
            del model, optimizer
        baseline = results['baseline']['selected']['panels']['val']
        signals = {}
        for arm in list(ARMS)[1:]:
            m = results[arm]['selected']['panels']['val']
            signals[arm] = dict(mean_ratio=m['xyz_mean_mm']/baseline['xyz_mean_mm'],
                                p95_ratio=m['xyz_p95_mm']/baseline['xyz_p95_mm'])
            signals[arm]['promising'] = (signals[arm]['mean_ratio'] <= .85 and
                                         signals[arm]['p95_ratio'] <= .85 and m['clipping'] < .01)
        conclusion = 'PROMISING' if any(v['promising'] for v in signals.values()) else 'UNCLEAR'
        atomic_json(out/'result.json', dict(status='COMPLETED', conclusion=conclusion, arms=results,
                    signals=signals, elapsed_s=time.monotonic()-started,
                    scope='current-information oracle R-only Probe; no grasp/generalization claim'))
        manifest.update(status='COMPLETED', conclusion=conclusion)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        atomic_json(out/'manifest.json', manifest)
        signal.alarm(0)


if __name__ == '__main__':
    run()
