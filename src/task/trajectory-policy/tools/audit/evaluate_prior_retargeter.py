"""Read-only old v1 learned-R baseline on the new Task's fixed validation wave."""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from audit_learned_retargeter import ROOT, TASK, reconstruct, report, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned output required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, before.split(','))
    if utilization > 10 or memory > 512:
        raise ValueError('baseline GPU not idle: '+before)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1')
    import torch
    sys.path.insert(0, str(TASK.parents[0]/'consequence-evaluator/src'))
    from consequence_evaluator.hand_action_retargeter import HandActionRetargeter, Standardizer, SCHEMA, HORIZON, ACTION_DIM
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(status='RUNNING', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before)
    try:
        fit = json.loads((args.fit/'manifest.json').read_text())
        prior = json.loads((args.prior/'manifest.json').read_text())
        prior_result = json.loads((args.prior/'result.json').read_text())
        if prior['status'] != 'COMPLETED' or prior['schema'] != SCHEMA or prior.get('teacher_anchor_sources'):
            raise ValueError('original structured-data-only v1 checkpoint required')
        checkpoint = args.prior/'best.pt'
        if sha(checkpoint) != prior_result['checkpoint_sha256'] or HORIZON != 24 or ACTION_DIM != 18:
            raise ValueError('original checkpoint/contract differs')
        source = TASK.parents[0]/'consequence-evaluator/src/consequence_evaluator/hand_action_retargeter.py'
        relative = str(source.relative_to(ROOT))
        historical = subprocess.check_output(['git', 'show', prior['git_commit']+':'+relative], text=True)
        def classes(text):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                if isinstance(node, ast.ClassDef) and node.name in ('HandActionRetargeter', 'Standardizer')}
        if classes(historical) != classes(source.read_text()) or len(classes(historical)) != 2:
            raise ValueError('old checkpoint class semantics differ from historical source')
        hashes = dict(fit['input_sha256'])
        for path in (args.fit/'manifest.json', args.fit/'result.json', args.fit/'tau-best.pt',
            args.prior/'manifest.json', args.prior/'result.json', checkpoint, source, Path(__file__), Path(__file__).with_name('audit_learned_retargeter.py')):
            hashes[str(path.resolve())] = sha(path)
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('fixed source drift')
        packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if packet['schema'] != SCHEMA or packet['manifest']['input_sha256'] != prior['input_sha256']:
            raise ValueError('old checkpoint provenance differs')
        model = HandActionRetargeter(packet['width']).cuda().eval()
        model.load_state_dict(packet['state_dict'], strict=True)
        statistics = {key: Standardizer(**value) for key, value in packet['statistics'].items()}
        if set(statistics) != {'hand', 'state', 'action'} or not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('old input normalization/nonfinite differs')
        hands, states, labels, ticks = [], [], [], []
        for metadata in fit['metadata']:
            with np.load(Path(metadata['source'])/'low.npz', allow_pickle=False) as f:
                data = {key: f[key] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'applied', 'episode_start', 'episode_tick')}
            state, _, action, indices = reconstruct(data, [2168])
            future = np.stack([data['hand'][indices+k] for k in range(1, 25)], axis=2)
            hands.append((future-data['hand'][indices, :, None]).reshape(-1, 24, 11, 3))
            states.append(state.reshape(-1, 87)[:, :36])
            labels.append(action.reshape(-1, 8, 12))
            ticks.append(np.repeat(data['episode_tick'][indices], 16))
        hand, state, truth = np.concatenate(hands), np.concatenate(states), np.concatenate(labels)
        startup = np.concatenate(ticks) < 64
        if len(hand) != 16576:
            raise ValueError('fixed validation count differs')
        outputs = []
        with torch.no_grad():
            for begin in range(0, len(hand), 512):
                raw = model(torch.as_tensor(statistics['hand'].encode(hand[begin:begin+512]), device='cuda'),
                    torch.as_tensor(statistics['state'].encode(state[begin:begin+512]), device='cuda')).cpu().numpy()
                outputs.append(statistics['action'].decode(raw)[:, :8, [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]])
        prediction = np.concatenate(outputs)
        if not np.isfinite(prediction).all():
            raise ValueError('old-model nonfinite prediction')
        comparison_scale = torch.load(args.fit/'tau-best.pt', map_location='cpu', weights_only=False)['model']['action_scale'].numpy()
        metrics = report(prediction, truth, comparison_scale, fit['native_controller'], startup)
        original = json.loads((args.fit/'result.json').read_text())['tau']
        result = dict(metrics=metrics, new_generic=original,
            prior_new_l1_ratio=metrics['normalized_l1']/original['normalized_l1'],
            prior_training=prior['splits'], checkpoint=str(checkpoint), historical_class_ast_match=True,
            claim='Frozen reuse baseline on same new validation rows/units; distinct training data/budgets, not a matched architecture claim or physical grasp proof')
        processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
        for process in processes.splitlines():
            pid, used = map(int, process.split(','))
            if pid != os.getpid() and used > 512:
                raise RuntimeError('foreign GPU compute detected')
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('source drift during baseline')
        write(args.output/'result.json', result)
        manifest.update(status='COMPLETED', input_sha256=hashes, validation_samples=len(hand),
            elapsed_s=time.monotonic()-started, torch_peak_bytes=torch.cuda.max_memory_allocated())
        write(args.output/'manifest.json', manifest)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', elapsed_s=time.monotonic()-started, error=repr(error))
        write(args.output/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    def deadline(signum, frame):
        raise TimeoutError('old retargeter frozen baseline exceeded60s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    main()
