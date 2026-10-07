#!/usr/bin/env python3
"""Matched cached-batch timing and semantic parity; no production-run changes."""
import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
spec = importlib.util.spec_from_file_location('reference_train', TASK / 'tools/run/train_oakink2_pointworld_temporal.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
from oakink_wm.pointworld_performance import install_fused_hilbert, restore_hilbert


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-sweep', action='store_true', help='Engineering throughput only; larger batches change the reference objective')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    train.configure_numerics()
    config = json.loads((TASK / 'configs/pointworld_temporal_wm24.json').read_text())
    data = train.Windows(a.data, 'train')
    draws = train.balanced_indices(data, 16, 218)
    batches = [train.device_batch(cpu) for cpu in train.loader(data, draws, 2)]
    torch.manual_seed(217)
    model = train.model_from_config(json.loads(a.stats.read_text()), config).cuda()

    parity = {}
    for arm in ('history', 'action', 'shuffle'):
        # shuffle arm consumes the same donated chunk in both executions.
        batch = train.shuffled_with_donors(batches[0], data, 217) if arm == 'shuffle' else batches[0]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = model(batch, arm)
        loss, _ = model.loss(expected, batch)
        loss.backward()
        grads = {k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}
        expected_cpu_rng, expected_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        expected_loss = loss.detach().clone()
        expected = {k: v.detach().clone() for k, v in expected.items()}
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        # Sparse scatter reductions can vary at BF16 rounding boundaries even
        # between two unchanged reference executions. Measure that control.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            repeated = model(batch, arm)
        repeated_loss, _ = model.loss(repeated, batch)
        repeated_loss.backward()
        reference_prediction_noise = {k:float((repeated[k]-expected[k]).abs().max()) for k in expected}
        reference_gradient_noise = max(float((v.grad-grads[k]).abs().max()) for k, v in model.named_parameters() if k in grads)
        reference_loss_noise = float((repeated_loss-expected_loss).abs())
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        original = install_fused_hilbert()
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                actual = model(batch, arm)
            actual_loss, _ = model.loss(actual, batch)
            actual_loss.backward()
            assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
            assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_rng)
            delta = 0.
            squared_delta, squared_reference = 0., 0.
            for k, v in model.named_parameters():
                assert (v.grad is None) == (k not in grads), (arm, k)
                if k in grads:
                    delta = max(delta, float((v.grad-grads[k]).abs().max()))
                    squared_delta += float((v.grad-grads[k]).square().sum())
                    squared_reference += float(grads[k].square().sum())
            relative_gradient_l2 = (squared_delta/max(squared_reference, 1e-20))**.5
            assert relative_gradient_l2 < 1e-3, (arm, relative_gradient_l2)
            assert float((actual_loss-expected_loss).abs()) < 1e-5, arm
            # Physical tolerances follow the existing BF16 checkpoint/evaluator
            # smoke bounds, rather than requiring bitwise floating-point replay.
            assert float((actual['translation']-expected['translation']).abs().max()) < 5e-5, arm
            assert float((actual['rotation']-expected['rotation']).abs().max()) < 3e-4, arm
            parity[arm] = dict(loss_max_abs=float((actual_loss-expected_loss).abs()),
                               prediction_max_abs={k:float((actual[k]-expected[k]).abs().max()) for k in expected},
                               gradient_max_abs=delta, cpu_rng_exact=True, cuda_rng_exact=True,
                               gradient_relative_l2=relative_gradient_l2,
                               unchanged_reference_repeat=dict(prediction_max_abs=reference_prediction_noise,
                                                               gradient_max_abs=reference_gradient_noise,
                                                               loss_max_abs=reference_loss_noise))
            print('parity', arm, parity[arm], flush=True)
        finally:
            restore_hilbert(original)
        del grads

    model.zero_grad(set_to_none=True)
    def update():
        for batch in batches:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(batch, 'action')
            loss, _ = model.loss(pred, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite diagnostic loss')
            (loss/8).backward()
            float(loss.detach())
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        float(grad)
    times = {}
    for backend in ('reference', 'fused_hilbert'):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        original = install_fused_hilbert() if backend == 'fused_hilbert' else None
        try:
            update(); torch.cuda.synchronize()  # warm kernels and optimizer state
            values = []
            for _ in range(3):
                start = time.perf_counter(); update(); torch.cuda.synchronize()
                values.append(time.perf_counter()-start)
            times[backend] = values
            print('timing', backend, values, flush=True)
        finally:
            if original is not None: restore_hilbert(original)
    summary = dict(engineering_only=True, parity=parity, update_seconds=times,
                   microbatch=2, accumulation=8, effective_batch=16,
                   median_speedup=statistics.median(times['reference'])/statistics.median(times['fused_hilbert']),
                   note='Same real cached batches; briefly shares a GPU with our action worker. Not an isolated hardware utilization measurement.',
                   implementation_sources=train.implementation_sources(),
                   performance_source_sha256=train.digest(TASK/'src/oakink_wm/pointworld_performance.py'),
                   script_sha256=train.digest(Path(__file__)))
    if a.batch_sweep:
        sweep = {}
        original = install_fused_hilbert()
        try:
            for microbatch in (2, 4, 8, 16):
                batches = [train.device_batch(cpu) for cpu in train.loader(data, draws, microbatch)]
                accumulation = 16 // microbatch
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
                def sweep_update():
                    for batch in batches:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            pred = model(batch, 'action')
                        loss, _ = model.loss(pred, batch)
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite sweep loss')
                        (loss/accumulation).backward(); float(loss.detach())
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); float(grad)
                sweep_update(); torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                values = []
                for _ in range(3):
                    start = time.perf_counter(); sweep_update(); torch.cuda.synchronize()
                    values.append(time.perf_counter()-start)
                sweep[str(microbatch)] = dict(accumulation=accumulation, update_seconds=values,
                                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                print('sweep', microbatch, sweep[str(microbatch)], flush=True)
        finally:
            restore_hilbert(original)
        summary['batch_sweep'] = sweep
        summary['batch_sweep_limitation'] = 'Equal16 windows/update does not preserve independently normalized pair losses, voxel origins or RNG layout; not an equivalent continuation recipe.'
    (a.output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
