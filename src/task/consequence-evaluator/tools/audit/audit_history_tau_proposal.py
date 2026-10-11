"""Replay frozen H-to-tau weights and diagnose train-only input scaling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
sys.path.insert(0, str(TASK / 'tools/run'))
from consequence_evaluator.data import sha
from consequence_evaluator.contracts import is_within
from consequence_evaluator.history_to_tau import HistoryToTau, SCHEMA
from train_history_to_tau import collect, encode, metrics


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seconds', type=int, default=180)
    a = p.parse_args(); started = time.monotonic()
    output = a.output.resolve(); fit = a.fit.resolve()
    if output.exists() or not is_within(output, ROOT / 'outputs/consequence-evaluator'):
        raise ValueError('fresh task-owned output required')
    if not 1 <= a.seconds <= 180:
        raise ValueError('bounded replay required')
    state = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
        '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    util, used, total = map(int, state.strip().split(','))
    if util > 10 or used > 512 or total - used < 20480:
        raise RuntimeError('GPU is not idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    m = json.loads((fit / 'manifest.json').read_text())
    if m.get('schema') != SCHEMA or m.get('status') != 'COMPLETED':
        raise ValueError('completed original fit required')
    checkpoint = fit / 'best.pt'
    if sha(checkpoint) != m['checkpoint_sha256']:
        raise ValueError('checkpoint identity mismatch')
    hashes = {str(checkpoint): sha(checkpoint), str(fit / 'manifest.json'): sha(fit / 'manifest.json'),
              str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for path in (TASK / 'tools/run/train_history_to_tau.py',
                 TASK / 'src/consequence_evaluator/history_to_tau.py',
                 TASK / 'src/consequence_evaluator/hand_execution.py'):
        hashes[str(path)] = sha(path)
    data = {}; source_roots = {}
    for split in ('train', 'val', 'test'):
        sources = [Path(path).parent for path in m['input_sha256']
                   if path.endswith('/manifest.json') and ('bank-' + split + '-') in path]
        if len(sources) != 1:
            raise ValueError('unique original source per split required')
        source_roots[split] = str(sources[0])
        data[split], source_m, source_hash = collect(sources[0], m['stride'])
        if source_m['actor_sha256'] != m['actor_sha256']:
            raise ValueError('source actor drift')
        for path, digest in source_hash.items():
            if m['input_sha256'].get(path) != digest:
                raise ValueError('source changed: ' + path)
        hashes.update(source_hash)
    sets = [set(data[s]['episode'].tolist()) for s in ('train', 'val', 'test')]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('episode overlap')
    packet = torch.load(checkpoint, map_location=device, weights_only=False)
    model = HistoryToTau(packet['architecture']['width']).to(device)
    model.load_state_dict(packet['model']); model.eval()
    stat = packet['statistics']
    masks = {key: np.asarray(stat[key][1]) <= 1.01e-4 for key in ('history', 'current_hand')}
    result = {}; saved_predictions = {}
    output.mkdir(parents=True)
    manifest = dict(schema='ref2dex.history-tau-proposal-audit.v1', status='RUNNING',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256=hashes, physical_gpu=a.gpu, gpu_before=state.strip(), sources=source_roots,
        seconds_cap=a.seconds, interventions=['original', 'zero_train_floor', 'clip10'],
        contract='Frozen old weights, train-only statistics, identical split targets; diagnostic interventions only')
    write(output / 'manifest.json', manifest)
    try:
        for split, arrays in data.items():
            h = encode(arrays['history'], stat['history'])
            c = encode(arrays['current_hand'], stat['current_hand'])
            summary = {}
            for key, values in [('history', h), ('current_hand', c)]:
                flat = np.abs(values).reshape(len(values), -1)
                raw = arrays[key].reshape(len(values), -1)
                channel_max = flat.max(0); worst = np.argsort(channel_max)[-8:][::-1]
                summary[key] = dict(abs_p99=float(np.quantile(flat, .99)), abs_max=float(flat.max()),
                    rms=float(np.sqrt(np.mean(values.astype('float64') ** 2))),
                    floor_channels=int(masks[key].sum()),
                    worst_channels=[dict(index=int(i), abs_normalized_max=float(channel_max[i]),
                        raw_min=float(raw[:, i].min()), raw_max=float(raw[:, i].max()),
                        train_mean=float(np.asarray(stat[key][0]).reshape(-1)[i]),
                        train_scale=float(np.asarray(stat[key][1]).reshape(-1)[i])) for i in worst])
            split_result = dict(input_distribution=summary,
                persistence=metrics(np.repeat(arrays['current_hand'][:, None], 24, axis=1), arrays['target']),
                windows=len(h), episodes=len(np.unique(arrays['episode'])), interventions={})
            for name in ('original', 'zero_train_floor', 'clip10'):
                hh = h.copy(); cc = c.copy()
                if name == 'zero_train_floor':
                    hh[:, masks['history']] = 0; cc[:, masks['current_hand']] = 0
                elif name == 'clip10':
                    hh = hh.clip(-10, 10); cc = cc.clip(-10, 10)
                values = []
                with torch.inference_mode():
                    for begin in range(0, len(hh), 512):
                        if time.monotonic() - started > a.seconds:
                            raise TimeoutError('replay budget')
                        normalized = model(torch.as_tensor(hh[begin:begin+512], device=device),
                                           torch.as_tensor(cc[begin:begin+512], device=device))
                        values.append(normalized.cpu().numpy() * stat['target'][1] + stat['target'][0])
                prediction = np.concatenate(values)
                if not np.isfinite(prediction).all():
                    raise ValueError('nonfinite prediction')
                scores = metrics(prediction, arrays['target'])
                scores['tick_bins'] = {str(lo): metrics(prediction[(arrays['tick'] >= lo) & (arrays['tick'] < hi)],
                    arrays['target'][(arrays['tick'] >= lo) & (arrays['tick'] < hi)])
                    for lo, hi in [(8, 64), (64, 192), (192, 384), (384, 543)]}
                split_result['interventions'][name] = scores
                if split == 'test':
                    saved_predictions[name] = prediction
            result[split] = split_result
            gpu = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
            processes = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
            for row in processes.strip().splitlines():
                pid, memory = map(int, row.split(','))
                if pid != os.getpid() and memory > 512:
                    raise RuntimeError('foreign GPU work')
            print(json.dumps(dict(split=split, elapsed_s=time.monotonic()-started,
                gpu=gpu.strip(), metrics={k:v['point_rmse_m'] for k,v in split_result['interventions'].items()})), flush=True)
        with np.load(fit / 'test-predictions.npz', allow_pickle=False) as old:
            for key in ('target', 'current', 'episode', 'tick'):
                current_key = 'current_hand' if key == 'current' else key
                if not np.array_equal(old[key], data['test'][current_key]):
                    raise ValueError('saved test data alignment differs')
            error = float(np.abs(saved_predictions['original'] - old['prediction']).max())
            if error > 1e-5:
                raise ValueError('original checkpoint replay differs')
        hashes[str(fit / 'test-predictions.npz')] = sha(fit / 'test-predictions.npz')
        result.update(original_replay_max_error=error, status='UNCLEAR',
            claim='Frozen-weight distribution interventions, not repaired model or deployable candidates')
        np.savez_compressed(output / 'test-interventions.npz', **saved_predictions)
        write(output / 'result.json', result)
        manifest.update(status='COMPLETED', elapsed_s=time.monotonic()-started,
                        original_replay_max_error=error)
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic()-started)
        raise
    finally:
        write(output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
