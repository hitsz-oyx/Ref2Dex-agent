"""Read-only, fixed-TRAIN-batch gradient comparison; no optimizer updates."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v6.geometry_consistency import geometry_components, BONES
from ptv3_shared_wam_v5.objectives import forecast_loss as base_loss


def norm(gradients):
    return torch.stack([g.float().square().sum() for g in gradients if g is not None]).sum().sqrt()


def dot(a, b):
    return sum((x.float() * y.float()).sum() for x, y in zip(a, b)
               if x is not None and y is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'stats', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    stats = json.loads(args.stats.read_text())
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = config.get('tf32', True)
    dataset = V6Windows(args.manifest, 'train')
    indices = []
    samples = []
    for index, _ in dataset.draw(64, 170, 0, 0):
        sample = dataset[index]
        if ((sample['contact_target'] > .25) & sample['interaction_valid']).any():
            indices.append(index)
            samples.append(sample)
        if len(samples) == 4:
            break
    if len(samples) != 4:
        raise ValueError('four real contact TRAIN examples required')
    batch = {k: v.to(device) for k, v in collate(samples).items()}
    report = dict(scope='four fixed TRAIN samples, train-mode BF16, no updates; not a matched-training causal test',
                  indices=indices, config=str(args.config), python=sys.version,
                  torch=torch.__version__, checkpoints=[])
    for path in args.checkpoints:
        torch.manual_seed(170)
        model = SharedWAMV6(stats, config).to(device)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        step = saved['step']
        del saved
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.get('amp', True)):
            prediction = model(batch)
        base, terms = base_loss(prediction, batch, config, model.normalizer)
        components, diagnostics = geometry_components(prediction, batch, config, torch.float64)
        means = {k: n / d.clamp_min(1) for k, (n, d) in components.items()}
        geometry = config['geometry_weight'] * (means['geometry_contact'] +
                   config['geometry_clearance_weight'] * means['geometry_clearance'])
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        base_grad = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
        geometry_grad = torch.autograd.grad(geometry, parameters, retain_graph=True, allow_unused=True)
        bn, gn = norm(base_grad), norm(geometry_grad)
        cross = dot(base_grad, geometry_grad)
        scene_grad = torch.autograd.grad(config['scene_weight'] * terms['scene'], parameters,
                                         retain_graph=True, allow_unused=True)
        sn = norm(scene_grad)
        sg = dot(scene_grad, geometry_grad)
        sb = dot(scene_grad, base_grad)
        total_norm = (bn.square() + gn.square() + 2 * cross).clamp_min(0).sqrt()
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        largest = sorted([(n, float(g.norm())) for n, g in zip(names, geometry_grad) if g is not None],
                         key=lambda item: item[1], reverse=True)[:8]
        with torch.no_grad():
            joints = batch['start_hand'][:, None] + prediction['trajectory'].float()
            first = joints[..., [a for a, _ in BONES], :]
            last = joints[..., [b for _, b in BONES], :]
            valid = batch['trajectory_valid'] & batch['start_hand_valid'][:, None]
            bone_valid = valid[..., [a for a, _ in BONES]] & valid[..., [b for _, b in BONES]]
            lengths = (last - first).norm(dim=-1)[bone_valid]
        row = dict(checkpoint=str(path), step=step,
                   losses=dict(base=float(base), hand=float(terms['hand']), scene=float(terms['scene']),
                               geometry_weighted=float(geometry), **{k: float(v) for k, v in means.items()}),
                   gradients=dict(base_norm=float(bn), geometry_weighted_norm=float(gn),
                                  geometry_to_base=float(gn / bn.clamp_min(1e-20)),
                                  base_geometry_cosine=float(cross / (bn * gn).clamp_min(1e-20)),
                                  scene_norm=float(sn), scene_geometry_cosine=float(sg / (sn * gn).clamp_min(1e-20)),
                                  scene_dot_total=float(sb + sg), total_norm=float(total_norm),
                                  clip_factor=float((config['clip_grad'] / total_norm.clamp_min(1e-20)).clamp(max=1)),
                                  largest_geometry_parameters=largest),
                   predicted_bone_length_mm=dict(min=float(lengths.min() * 1000),
                                                 median=float(lengths.median() * 1000)),
                   diagnostics=diagnostics, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        report['checkpoints'].append(row)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
        del base_grad, geometry_grad, scene_grad, prediction, components, means, terms, base, geometry, model, parameters
        gc.collect()
        torch.cuda.empty_cache()
        if time.monotonic() - started > 300:
            raise TimeoutError('bounded audit exceeded five minutes')
    report['elapsed_s'] = time.monotonic() - started
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
