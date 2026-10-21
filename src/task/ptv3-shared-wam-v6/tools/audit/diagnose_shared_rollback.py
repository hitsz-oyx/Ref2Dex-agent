"""Causal parameter-group rollback on disposable real-checkpoint copies."""
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
from ptv3_shared_wam_v5.objectives import forecast_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'stats', 'source', 'degraded', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    stats = json.loads(args.stats.read_text())
    torch.backends.cuda.matmul.allow_tf32 = config.get('tf32', True)
    dataset = V6Windows(args.manifest, 'train')
    samples, indices = [], []
    for index, _ in dataset.draw(64, 170, 0, 0):
        sample = dataset[index]
        if ((sample['contact_target']>.25)&sample['interaction_valid']).any():
            samples.append(sample)
            indices.append(index)
        if len(samples)==4:
            break
    batch = {k: v.cuda() for k, v in collate(samples).items()}
    saved_source = torch.load(args.source, map_location='cpu', weights_only=False)
    source = saved_source['model']
    del saved_source
    saved_degraded = torch.load(args.degraded, map_location='cpu', weights_only=False)
    degraded = saved_degraded['model']
    del saved_degraded
    variants = dict(
        degraded=lambda n:False,
        source=lambda n:True,
        restore_shared_ffn=lambda n:n.startswith('interaction.layers.') and ('.linear1.' in n or '.linear2.' in n),
        restore_shared_attention=lambda n:n.startswith('interaction.layers.') and '.self_attn.' in n,
        restore_shared_norms=lambda n:n.startswith('interaction.layers.') and ('.norm1.' in n or '.norm2.' in n),
        restore_scene_context=lambda n:n.startswith('scene_context.'),
        restore_shared_all=lambda n:n.startswith('interaction.layers.'),
        restore_shared_linear2=lambda n:n.startswith('interaction.layers.') and '.linear2.' in n)
    report = dict(scope='four actual fixed TRAIN examples; group rollback diagnostic, no updates or checkpoint writes',
                  indices=indices, source=str(args.source), degraded=str(args.degraded), results=[])
    started = time.monotonic()
    for variant, predicate in variants.items():
        torch.manual_seed(170)
        model = SharedWAMV6(stats, config).cuda().train()
        model.load_state_dict({n:source[n] if predicate(n) else v for n,v in degraded.items()}, strict=True)
        captured = {}
        def context_hook(module, inputs, output):
            captured['shared_rms'] = float(inputs[0].detach().float().square().mean().sqrt())
            captured['context_rms'] = float(output.detach().float().square().mean().sqrt())
        hook = model.scene_context.register_forward_hook(context_hook)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(batch)
        loss, terms = forecast_loss(prediction, batch, config, model.normalizer)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        grads = torch.autograd.grad(terms['scene'], parameters, allow_unused=True)
        norm = float(torch.stack([g.float().square().sum() for g in grads if g is not None]).sum().sqrt())
        hook.remove()
        row = dict(variant=variant, **captured, scene_loss=float(terms['scene']),
                   hand_loss=float(terms['hand']), original_total_loss=float(loss), scene_gradient_norm=norm,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        report['results'].append(row)
        report['elapsed_s'] = time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
        del parameters, grads, prediction, loss, terms, model
        gc.collect()
        torch.cuda.empty_cache()
        if report['elapsed_s']>300:
            raise TimeoutError('five minute rollback audit budget')


if __name__=='__main__':
    main()
