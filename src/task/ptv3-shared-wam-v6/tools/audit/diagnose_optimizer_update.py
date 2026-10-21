"""One disposable same-batch step per optimizer policy; never writes weights."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v6.objectives import forecast_loss
from ptv3_shared_wam_v5.training_contract import restore_optimizer_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'stats', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
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
        if ((sample['contact_target'] > .25) & sample['interaction_valid']).any():
            samples.append(sample)
            indices.append(index)
        if len(samples) == 4:
            break
    batch = {k: v.cuda() for k, v in collate(samples).items()}
    report = dict(scope='four fixed TRAIN samples; one disposable step/policy, not a matched DDP continuation',
                  indices=indices, results=[], python=sys.version, torch=torch.__version__)
    start = time.monotonic()
    for path in args.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for policy in ('restored-adam', 'fresh-adam', 'sgd'):
            torch.manual_seed(170)
            model = SharedWAMV6(stats, config).cuda().train()
            model.load_state_dict(saved['model'], strict=True)
            groups = model.parameter_groups(config['geometry_lr'], config['shared_lr'])
            if policy == 'sgd':
                optimizer = torch.optim.SGD(groups, lr=config['shared_lr'], weight_decay=config['weight_decay'])
            else:
                optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'], fused=True)
                if policy == 'restored-adam':
                    restore_optimizer_state(optimizer, saved['optimizer'])
            torch.cuda.reset_peak_memory_stats()
            rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            def forward():
                torch.set_rng_state(rng[0])
                torch.cuda.set_rng_state(rng[1])
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.get('amp', True)):
                    prediction = model(batch)
                    loss, terms = forecast_loss(prediction, batch, config, model.normalizer,
                                                step=saved['step']+1)
                values = {k: float(v) for k, v in terms.items() if k != 'components'}
                return loss, values
            before, before_terms = forward()
            before_value = float(before)
            before.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
            originals = {n: p.detach().clone() for n, p in model.named_parameters()}
            optimizer.step()
            dot, delta_sq, weight_sq = 0., 0., 0.
            groups_dot = {}
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    delta = parameter-originals[name]
                    delta_sq += float(delta.square().sum())
                    weight_sq += float(originals[name].square().sum())
                    if parameter.grad is not None:
                        value = float((parameter.grad*delta).sum())
                        dot += value
                        key = '.'.join(name.split('.')[:2])
                        groups_dot[key] = groups_dot.get(key, 0.)+value
                after, after_terms = forward()
            row = dict(checkpoint=str(path), step=saved['step'], policy=policy,
                       before_loss=before_value, after_loss=float(after), before_terms=before_terms,
                       after_terms=after_terms, preclip_norm=float(norm), clipped_grad_dot_update=dot,
                       delta_norm=math.sqrt(delta_sq), relative_delta=math.sqrt(delta_sq/weight_sq),
                       most_ascending_groups=sorted(groups_dot.items(), key=lambda x:x[1], reverse=True)[:6],
                       most_descending_groups=sorted(groups_dot.items(), key=lambda x:x[1])[:6],
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            report['results'].append(row)
            report['elapsed_s'] = time.monotonic()-start
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps(row), flush=True)
            del model, optimizer, originals, before, after, parameter, delta, groups
            gc.collect()
            torch.cuda.empty_cache()
            if report['elapsed_s'] > 300:
                raise TimeoutError('five minute audit budget')
        del saved


if __name__ == '__main__':
    main()
