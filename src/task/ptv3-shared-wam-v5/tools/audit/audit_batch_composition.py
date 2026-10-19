"""Read-only checkpoint inference with identical targets and different batch neighbours.

Serialization instrumentation/fixed-depth control exists only in this process.
No optimizer, training mutation, checkpoint rewrite, or production patch.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.data import collate
from ptv3_shared_wam_v5.storage import read_json


def summary(values):
    x = values.detach().float().flatten().cpu().numpy()
    return dict(count=len(x), mean=float(x.mean()), p50=float(np.quantile(x, .5)),
                p95=float(np.quantile(x, .95)), max=float(x.max()))


def differences(reference, actual, sample):
    mask = torch.as_tensor(sample['start_hand_valid'])[None].expand(24, -1, -1)
    delta = (actual['trajectory']-reference['trajectory']).norm(dim=-1)*1000
    confidence = (actual['interaction'][..., 0]-reference['interaction'][..., 0]).abs()
    parts = (actual['interaction'][..., 1:].argmax(-1) != reference['interaction'][..., 1:].argmax(-1))
    gt = torch.as_tensor(sample['trajectory_target'])
    valid = torch.as_tensor(sample['trajectory_valid']) & mask
    r = (reference['trajectory']-gt).norm(dim=-1)*1000
    e = (actual['trajectory']-gt).norm(dim=-1)*1000
    return dict(tau_delta_mm=summary(delta[mask]), tau_h1_4_delta_mm=summary(delta[:4][mask[:4]]),
        c_abs_delta=summary(confidence), part_flip_rate=float(parts.float().mean()),
        reference_epe_mm=float(r[valid].mean()), actual_epe_mm=float(e[valid].mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-memory-gib', type=float)
    p.add_argument('--git-commit', required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_device(a.device)
    if a.max_memory_gib is not None:
        free, total = torch.cuda.mem_get_info()
        budget = int(a.max_memory_gib * 2**30)
        if budget <= 0 or free - budget < 8 * 2**30:
            raise ValueError('audit must leave at least8GiB beyond other GPU allocations')
        torch.cuda.set_per_process_memory_fraction(budget / total)
    config = read_json(a.inputs / 'config.json')
    stats = read_json(a.inputs / 'stats.json')
    meta = read_json(a.inputs / 'export.json')
    torch.backends.cuda.matmul.allow_tf32 = config.get('tf32', True)
    saved = torch.load(a.inputs / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert saved['config'] == config and saved['world_size'] == 4
    model = SharedWAMV5(stats, config).to(a.device)
    model.load_state_dict(saved['model'], strict=True)
    step = saved['step']
    del saved
    model.eval()
    assert not any(getattr(m, 'shuffle_orders', False) for m in model.modules())
    with np.load(a.inputs / 'panel.npz', allow_pickle=False) as z:
        keys = [k for k in z.files if not k.startswith('sentinel__')]
        data = {k: z[k] for k in keys}
        sentinel = {k: z['sentinel__' + k] for k in keys}
    panel = [{k: v[i] for k, v in data.items()} for i in range(meta['panel_size'])]
    count = meta['target_count']
    targets = panel[:count]
    from ptv3.structure import Point
    original = Point.serialization
    trace, mode, stage = [], {'fixed': False}, [0]
    def serialization(point, order='z', depth=None, shuffle_orders=False, prefix=''):
        level = stage[0]
        stage[0] += 1
        if mode['fixed']:
            depth = max(1, fixed_initial_depth - level)
            assert int(point.grid_coord.max()) < 2**depth
        original(point, order=order, depth=depth, shuffle_orders=shuffle_orders, prefix=prefix)
        # Record first target's patch memberships using voxel IDs, not batch offsets.
        ids = torch.where(point.batch == 0)[0]
        orders = {}
        for i, name in enumerate(order):
            sorted_ids = point.serialized_order[i]
            selected = sorted_ids[point.batch[sorted_ids] == 0]
            membership = torch.empty(len(ids), device=ids.device, dtype=torch.long)
            membership[selected-ids[0]] = torch.arange(len(ids), device=ids.device)//config['patch_size']
            orders[name] = membership.cpu().tolist()
        trace.append(dict(level=level, depth=point.serialized_depth,
            grid_max=int(point.grid_coord.max()), first_sample_voxels=len(ids), patches=orders))
    Point.serialization = serialization
    fixed_initial_depth = max(meta['panel_depths'] + [meta['sentinel_depth']]) + 2
    assert fixed_initial_depth <= 16
    cases = []
    def infer(name, samples, indices):
        stage[0] = 0
        begin = len(trace)
        batch = {k: v.to(a.device) for k, v in collate(samples).items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['amp']):
            pred = model(batch)
        result = [{k: pred[k][i].float().cpu() for k in ('trajectory','interaction')} for i in indices]
        del pred, batch
        torch.cuda.synchronize()
        info = dict(name=name, batch=len(samples), fixed_depth=mode['fixed'], trace=trace[begin:])
        cases.append(info)
        print(json.dumps(dict(case=name, batch=len(samples), depths=[t['depth'] for t in info['trace']])), flush=True)
        return result
    started = time.monotonic()
    outputs = {}
    try:
        reference = [infer('single-%d'%i, [s], [0])[0] for i,s in enumerate(targets)]
        outputs['repeat_single'] = [differences(reference[0], infer('repeat-single', [targets[0]], [0])[0], targets[0])]
        for label, n in [('batch8', 8), ('batch128', 128)]:
            actual = []
            for first in range(0, count, n):
                take = min(n, count-first)
                samples = targets[first:first+take] + panel[count:count+n-take]
                actual.extend(infer(label+'-%d'%first, samples, list(range(take))))
            outputs[label] = [differences(r,x,s) for r,x,s in zip(reference,actual,targets)]
        actual = infer('batch8-real-sentinel', [targets[0]]+[sentinel]*7, [0])[0]
        outputs['real_sentinel'] = [differences(reference[0], actual, targets[0])]
        shifted = {k: np.array(v, copy=True) for k,v in targets[0].items()}
        # Move ONLY neighbour's observed hand; target geometry/labels stay byte-identical.
        for key in ('start_hand','dense_hand'):
            valid = shifted['start_hand_valid'] if key == 'start_hand' else shifted['dense_hand_valid']
            values = shifted[key]
            values[...,0][valid] += 3.0
        actual = infer('batch8-synthetic-distant-hand', [targets[0]]+[shifted]*7, [0])[0]
        outputs['synthetic_depth_stress'] = [differences(reference[0], actual, targets[0])]
        mode['fixed'] = True
        fixed_reference = infer('fixed-depth-single', [targets[0]], [0])[0]
        actual = infer('fixed-depth-synthetic-neighbour', [targets[0]]+[shifted]*7, [0])[0]
        outputs['fixed_depth_control'] = [differences(fixed_reference, actual, targets[0])]
    finally:
        Point.serialization = original
    report = dict(status='COMPLETED', git_commit=a.git_commit, training_commit='6334e1ea',
        checkpoint_step=step, python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), eval_mode=True, seed=meta['seed'],
        max_memory_gib=a.max_memory_gib,
        target_sources=meta['target_sources'], target_ids=meta['target_ids'],
        target_depths=meta['target_depths'], fixed_initial_depth=fixed_initial_depth,
        comparisons=outputs, cases=cases, elapsed_s=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        limitation='Depth-stress panel, not population accuracy. Synthetic-neighbour and fixed-depth runs are diagnostic controls only; frozen training and its stats were not changed.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','comparisons')}),flush=True)


if __name__ == '__main__':
    main()
