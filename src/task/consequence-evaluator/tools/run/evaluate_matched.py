"""Freeze validation-selected weights before a bounded independent test pass."""
import argparse
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

from consequence_evaluator.contracts import is_within
from consequence_evaluator.data import Windows, sha
from consequence_evaluator.evaluation import future_donors, predict, summarize
from consequence_evaluator.model import Evaluator


def write(path, record):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_frozen(fit, data, data_root):
    """Validate fitted input/code identity and select weights solely by val."""
    manifest = json.loads((fit / 'run_manifest.json').read_text())
    if manifest.get('status') != 'COMPLETED' or manifest.get('task') != 'consequence-evaluator':
        raise ValueError('only a completed matched fit can enter test evaluation')
    if (manifest['window_manifest_sha256'] != sha(data_root / 'manifest.json')
            or manifest['windows_sha256'] != data.manifest['windows_sha256']
            or manifest['initial_weights_sha256'] != sha(fit / 'initial.pt')):
        raise ValueError('fitted input or initialization identity changed')
    for name, digest in manifest['sources'].items():
        source = (ROOT / name).resolve()
        if not is_within(source, ROOT) or sha(source) != digest:
            raise ValueError('fitted implementation identity changed: ' + name)
    records = [json.loads(line) for line in (fit / 'validation.jsonl').read_text().splitlines()]
    if not records:
        raise ValueError('validation selection record is missing')
    expected_stats = data.normalization()
    models, statistics, selections = {}, {}, {}
    history_dim = int(np.prod(data.arrays['history'].shape[1:]))
    for arm in ('baseline', 'oracle'):
        accuracies = [item[arm]['strict_accuracy'] for item in records]
        if not all(isinstance(v, (float, int)) and np.isfinite(v) and 0 <= v <= 1 for v in accuracies):
            raise ValueError('invalid validation selection metrics')
        # Training saves only on a strict improvement: select the first maximum.
        selected = records[int(np.argmax(accuracies))]
        path = fit / (arm + '-best.pt')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['arm'] != arm or payload['step'] != selected['step']:
            raise ValueError('checkpoint does not match validation-only selection')
        for key in ('windows_sha256', 'window_manifest_sha256', 'initial_weights_sha256',
                    'training_pair_draw_sha256', 'width', 'layers', 'updates', 'pairs_per_update'):
            if payload['manifest'][key] != manifest[key]:
                raise ValueError('checkpoint matched-fit identity mismatch: ' + key)
        stats = payload['statistics']
        if (set(stats) != set(expected_stats)
                or any(len(stats[key]) != 2 or not all(torch.equal(old.cpu(), new)
                       for old, new in zip(stats[key], expected_stats[key])) for key in expected_stats)):
            raise ValueError('checkpoint statistics are not the train-only statistics')
        model = Evaluator(history_dim, width=manifest['width'], layers=manifest['layers'])
        model.load_state_dict(payload['model'], strict=True)
        if sum(v.numel() for v in model.parameters()) != manifest['parameters']:
            raise ValueError('fitted evaluator capacity changed')
        models[arm], statistics[arm] = model, stats
        selections[arm] = dict(path=path.name, sha256=sha(path), step=payload['step'],
                               validation_strict_accuracy=selected[arm]['strict_accuracy'])
    return models, statistics, selections


def freeze_protocol(fit, protocol):
    """Repeated test passes must retain the first pre-inference protocol."""
    path = fit / 'test_protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('frozen test protocol changed; do not retune against test')
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(protocol, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, required=True, help='fixed future-donor diagnostic seed')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seconds', type=int, default=300)
    a = p.parse_args()
    if not 1 <= a.batch <= 128 or not 1 <= a.seconds <= 300:
        p.error('bounded evaluation: <=128 windows/batch and <=300 seconds')
    fit, output = a.fit.resolve(), a.output.resolve()
    owned = ROOT / 'outputs/consequence-evaluator'
    if not is_within(fit, owned) or not is_within(output, owned):
        p.error('fit and output must be under outputs/consequence-evaluator')
    if output.exists():
        raise FileExistsError('choose a fresh evaluation output directory')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                                       '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU is occupied; no model is loaded: ' + occupied.replace('\n', ', '))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    device = torch.device('cuda:0')
    started = time.monotonic()
    deadline = started + a.seconds
    data = Windows(a.data)
    pairs = data.arrays['pairs'][data.pair_ids['test']]
    if not len(pairs):
        raise ValueError('held-out test preferences are required')
    ids = np.unique(pairs)
    if len(ids) > 20000:
        raise ValueError('bounded evaluation supports at most 20000 unique test windows')
    models, statistics, selections = load_frozen(fit, data, a.data)
    donors = future_donors(data, ids, a.seed)
    protocol = dict(schema='ref2dex.consequence-evaluator.test-protocol.v1',
                    fit_manifest_sha256=sha(fit / 'run_manifest.json'),
                    validation_log_sha256=sha(fit / 'validation.jsonl'),
                    windows_sha256=data.manifest['windows_sha256'],
                    window_manifest_sha256=sha(a.data / 'manifest.json'),
                    selection='first maximum of validation strict accuracy, independently per arm',
                    checkpoints=selections, seed=a.seed, batch=a.batch,
                    primary_metric='test strict preference accuracy; ties count as incorrect',
                    future_control='same task/phase, other test episode, sampling with replacement',
                    sources={str(path.relative_to(ROOT)): sha(path) for path in
                             [Path(__file__), TASK / 'src/consequence_evaluator/evaluation.py']})
    freeze_protocol(fit, protocol)
    output.mkdir(parents=True)
    report = dict(status='RUNNING', task='consequence-evaluator', run_id=output.name,
                  physical_gpu=a.gpu, pid=os.getpid(), seconds_budget=a.seconds, protocol=protocol,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(output / 'report.json', report)
    try:
        predictions = {}
        for arm in ('baseline', 'oracle'):
            model = models[arm].to(device)
            predictions[arm] = predict(model, data, ids, statistics[arm], device, arm == 'oracle',
                                       a.batch, deadline=deadline)
            if arm == 'oracle':
                predictions['oracle_reassigned_future'] = predict(model, data, ids, statistics[arm], device,
                                                                  True, a.batch, donors, deadline)
            model.cpu()
        report.update(summarize(data, ids, predictions))
        # Preserve per-window predictions and donor identities for independent recomputation.
        np.savez_compressed(output / 'predictions.npz', window_ids=ids, pair_ids=data.pair_ids['test'],
                            future_donor_ids=donors,
                            **{name + '_' + key: value for name, packet in predictions.items()
                               for key, value in packet.items()})
        if (sha(a.data / 'windows.npz') != protocol['windows_sha256']
                or sha(a.data / 'manifest.json') != protocol['window_manifest_sha256']
                or sha(fit / 'run_manifest.json') != protocol['fit_manifest_sha256']
                or sha(fit / 'validation.jsonl') != protocol['validation_log_sha256']
                or any(sha(fit / entry['path']) != entry['sha256'] for entry in selections.values())
                or any(sha(ROOT / path) != digest for path, digest in protocol['sources'].items())):
            raise RuntimeError('frozen evaluation inputs or source changed')
        report.update(status='COMPLETED', predictions_sha256=sha(output / 'predictions.npz'))
    except BaseException as error:
        report.update(status='TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', error=repr(error))
        raise
    finally:
        report.update(elapsed_s=time.monotonic() - started,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(output / 'report.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
