"""Bounded gradient-direction probe; reads a checkpoint and performs no update."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.code_root / 'src/task/ptv3-shared-wam-v4/src'))
    from ptv3_shared_wam_v4 import paths
    from ptv3_shared_wam_v4.data import NineWindows, collate
    from ptv3_shared_wam_v4.model import SharedWAMV4
    from ptv3_shared_wam_v4.objectives import forecast_loss
    from ptv3_shared_wam.loading import to_device
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    started = time.monotonic()
    packet = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config, stats = packet['config'], packet['stats']
    model = SharedWAMV4(stats, config).cuda().eval()
    model.load_state_dict(packet['model'], strict=True)
    del packet['model']
    del packet
    recipe = stats['recipe']
    dataset = NineWindows(recipe['main3_root'], recipe['native_root'], 'train', recipe)
    # Inspect the entire final shared transformer layer, not just one weight.
    parameters = list(model.interaction.layers[-1].parameters())
    rows = []
    for seed in (181, 182, 183):
        panel = dataset.panel(2, seed)
        indices = [i for source in sorted(panel, key=int) for i in panel[source]]
        batch = to_device(collate([dataset[i] for i in indices])['model_inputs'], torch.device('cuda:0'))
        with torch.autocast('cuda', dtype=torch.float16, enabled=bool(config.get('amp', False))):
            prediction = model(batch)
            total, terms = forecast_loss(prediction, batch, config, model.normalizer, distributed=False)
        losses = {'hand': config['hand_weight'] * terms['hand'],
                  'scene': config['scene_weight'] * terms['scene'],
                  'C': config.get('interaction_weight', 1.) * terms['interaction']}
        gradients = {}
        for name, loss in losses.items():
            values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            gradients[name] = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().float().flatten()
                                          for p, g in zip(parameters, values)])
        gradients['hand+scene'] = gradients['hand'] + gradients['scene']
        norms = {k: float(v.norm()) for k, v in gradients.items()}
        if any(not torch.isfinite(v).all() for v in gradients.values()):
            raise ValueError('nonfinite component gradient')
        cosine = {a + ':' + b: float(torch.dot(gradients[a], gradients[b]) /
                                    (gradients[a].norm() * gradients[b].norm()).clamp_min(1e-20))
                  for a, b in (('hand', 'scene'), ('hand', 'C'), ('scene', 'C'), ('hand+scene', 'C'))}
        rows.append(dict(seed=seed, sample_indices=indices,
                         weighted_losses={k: float(v.detach()) for k, v in losses.items()},
                         gradient_norms=norms, cosine=cosine))
        del prediction, total, terms, losses, gradients, batch, values
        print(json.dumps(rows[-1]), flush=True)
    result = dict(work_version='wam-v4-component-gradient.v1', checkpoint=str(args.checkpoint),
                  code_root=str(args.code_root), dependency_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=str(args.code_root), text=True).strip(),
                  scope='eval-mode final shared transformer layer; three source-balanced TRAIN batches; no optimizer',
                  parameters=sum(p.numel() for p in parameters), rows=rows,
                  python=sys.version, torch=torch.__version__, elapsed_s=time.monotonic() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
