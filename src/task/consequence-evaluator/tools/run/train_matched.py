"""Fit the same evaluator with a null future slot and a real rigid future slot."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))

import numpy as np
import torch

from consequence_evaluator.data import Windows, sha
from consequence_evaluator.model import Evaluator, matched_loss
from consequence_evaluator.contracts import is_within, ARMS, future_mode, MIN_PREFERENCE_PAIRS


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def normalized(data, ids, device, statistics):
    inputs, labels = data.batch(ids, device)
    inputs = {key: (value - statistics[key][0]) / statistics[key][1]
              for key, value in inputs.items()}
    return inputs, labels


def require_supervision(data):
    if data.manifest.get('pair_coverage_required') is not True:
        raise ValueError('fit requires a production pair-coverage manifest; engineering audit data are not trainable')
    groups = {split: set() for split in ('train', 'val', 'test')}
    for split in groups:
        for pair_id in data.pair_ids[split]:
            left, right = data.arrays['pairs'][pair_id]
            groups[split].add(tuple(sorted((str(data.arrays['episode'][left]),
                                             str(data.arrays['episode'][right])))))
    counts = {split: len(values) for split, values in groups.items()}
    if any(counts[split] < minimum for split, minimum in MIN_PREFERENCE_PAIRS.items()):
        raise ValueError('minimum local preference coverage required: '
                         + json.dumps({'observed': counts, 'minimum': MIN_PREFERENCE_PAIRS}, sort_keys=True))
    expert_ids = np.flatnonzero((data.arrays['split'] == 'train') & data.arrays['progress_mask'].any(axis=1))
    if not len(expert_ids):
        raise ValueError('reliable train expert progress anchors required for the joint objective')
    return expert_ids


@torch.no_grad()
def evaluate(model, data, pair_ids, device, statistics, use_future, batch):
    model.eval()
    correct = ties = count = 0
    for start in range(0, len(pair_ids), batch):
        pairs = data.arrays['pairs'][pair_ids[start:start + batch]]
        left, _ = normalized(data, pairs[:, 0], device, statistics)
        right, _ = normalized(data, pairs[:, 1], device, statistics)
        delta = model(**left, use_future=use_future)['score'] - model(**right, use_future=use_future)['score']
        correct += int((delta > 0).sum())
        ties += int((delta == 0).sum())
        count += len(delta)
    model.train()
    return dict(pairs=count, strict_accuracy=correct / count if count else None, ties=ties)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--width', type=int, default=128)
    a = p.parse_args()
    if not 1 <= a.updates <= 1000 or not 1 <= a.seconds <= 1800 or not 1 <= a.batch <= 128:
        p.error('bounded probe: <=1000 updates, <=1800 seconds, <=128 pairs/batch')
    if a.width < 4 or a.width % 4:
        p.error('width must be a positive multiple of four')
    owned = (ROOT / 'outputs/consequence-evaluator').resolve()
    output = a.output.resolve()
    if not is_within(output, owned):
        p.error('output must be under outputs/consequence-evaluator')
    if output.exists():
        raise FileExistsError('choose a fresh output directory')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                                       '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU is occupied; no model is loaded: ' + occupied.replace('\n', ', '))
    # Device identity is explicit; never replace CUDA with CPU silently.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    device = torch.device('cuda:0')
    data = Windows(a.data)
    expert_ids = require_supervision(data)
    output.mkdir(parents=True)
    source = {str(path.relative_to(ROOT)): sha(path) for path in
              [Path(__file__), *sorted((TASK / 'src/consequence_evaluator').glob('*.py'))]}
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    shape = data.arrays['history'].shape[1:]
    initial = Evaluator(int(np.prod(shape)), width=a.width)
    models = {name: copy.deepcopy(initial).to(device) for name in ARMS}
    statistics = {key: tuple(item.to(device) for item in value)
                  for key, value in data.normalization().items()}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
                  for name, model in models.items()}
    draws = np.random.default_rng(a.seed + 1).choice(data.pair_ids['train'], (a.updates, a.batch))
    expert_draws = np.random.default_rng(a.seed + 2).choice(expert_ids, (a.updates, a.batch))
    torch.save(dict(model=initial.state_dict(), statistics=data.normalization()), output / 'initial.pt')
    manifest = dict(status='RUNNING', task='consequence-evaluator', run_id=output.name,
                    work_version='consequence-evaluator-ref2',arms=list(ARMS),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    physical_gpu=a.gpu, pid=os.getpid(), seed=a.seed,
                    window_manifest_sha256=sha(a.data / 'manifest.json'),
                    windows_sha256=data.manifest['windows_sha256'], sources=source,
                    updates=a.updates, pairs_per_update=a.batch, seconds_budget=a.seconds,
                    expert_windows_per_update=a.batch,
                    parameters=sum(v.numel() for v in initial.parameters()), width=a.width, layers=2,
                    split='train/val only', preference_objective='adapted_independent_scalar_BT',
                    initial_weights_sha256=sha(output / 'initial.pt'),
                    training_pair_draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                    training_expert_draw_sha256=hashlib.sha256(expert_draws.tobytes()).hexdigest())
    started = time.monotonic()
    best = {name: -1. for name in models}
    step = 0
    write(output / 'run_manifest.json', manifest)
    try:
        for step, pair_ids in enumerate(draws, 1):
            if time.monotonic() - started >= a.seconds:
                raise TimeoutError('fixed matched-fit deadline')
            if any(sha(ROOT / name) != value for name, value in source.items()):
                raise RuntimeError('training implementation drift')
            pairs = data.arrays['pairs'][pair_ids]
            left, left_labels = normalized(data, pairs[:, 0], device, statistics)
            right, right_labels = normalized(data, pairs[:, 1], device, statistics)
            expert, expert_labels = normalized(data, expert_draws[step-1], device, statistics)
            record = dict(step=step)
            for name, model in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                loss, terms = matched_loss(model(**left, use_future=future_mode(name)),
                                           model(**right, use_future=future_mode(name)),
                                           left_labels, right_labels,
                                           model(**expert, use_future=future_mode(name)), expert_labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite matched evaluator loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizers[name].step()
                record[name] = dict(loss=float(loss.detach()), **{key: float(value.detach()) for key, value in terms.items()})
            record['elapsed_s'] = time.monotonic() - started
            with (output / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step % 100 == 0 or step == a.updates:
                validation = dict(step=step, **{name: evaluate(model, data, data.pair_ids['val'], device,
                                                            statistics, future_mode(name), a.batch)
                                               for name, model in models.items()})
                with (output / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(validation) + '\n')
                for name, model in models.items():
                    payload = dict(model=model.state_dict(), optimizer=optimizers[name].state_dict(),
                                   statistics=statistics, arm=name, step=step, manifest=manifest)
                    temporary = output / (name + '-latest.partial')
                    torch.save(payload, temporary)
                    temporary.replace(output / (name + '-latest.pt'))
                    accuracy = validation[name]['strict_accuracy']
                    if accuracy > best[name]:
                        best[name] = accuracy
                        torch.save(payload, output / (name + '-best.pt'))
                elapsed = time.monotonic() - started
                print(json.dumps(dict(step=step, validation=validation,
                                      seconds_per_matched_update=elapsed / step,
                                      eta_remaining_s=elapsed / step * (a.updates - step),
                                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      gpu_state=subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                                          '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip())), flush=True)
        if sha(a.data / 'windows.npz') != data.manifest['windows_sha256']:
            raise RuntimeError('dataset drift during fit')
        manifest['status'] = 'COMPLETED'
    except BaseException as error:
        manifest.update(status='TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', error=repr(error))
        raise
    finally:
        manifest.update(step=step, elapsed_s=time.monotonic() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(output / 'run_manifest.json', manifest)


if __name__ == '__main__':
    main()
