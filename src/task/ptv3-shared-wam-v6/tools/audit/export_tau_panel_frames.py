#!/usr/bin/env python3
"""Export missing paired frame records for the EXISTING B1/B0a comparison only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
sys.path.insert(0, str(TASK/'tools/run'))
from evaluate_tau_fixed_panel import read, fixed_indices, validate_panel, Accumulator
from score_tau_panel_trivial_baselines import frame_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel', type=Path, default=TASK/'configs/tau-fixed-panel-v1.json')
    p.add_argument('--gpu', type=int, default=7)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists(): raise FileExistsError(a.output)
    if shutil.disk_usage(ROOT).free < 20*2**30: raise RuntimeError('storage floor')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import torch
    from ptv3_shared_wam_v6.data import V6Windows, collate
    from ptv3_shared_wam_v6.model import SharedWAMV6
    panel = read(a.panel); anchor = read(ROOT/panel['anchor'])
    rows = validate_panel(a.panel, panel, anchor)
    dataset = V6Windows(ROOT/panel['event_recipe'], 'train', cache_size=8)
    indices = fixed_indices(dataset, rows)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    assert config['flow_eval_seed'] == panel['flow_eval_seed']
    torch.manual_seed(panel['seed']); np.random.seed(panel['seed']); torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True); torch.cuda.set_device(0)
    model = SharedWAMV6(read(ROOT/panel['stats']), config).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    step = int(saved['step']); del saved
    arrays = {k: np.zeros((len(rows), 24), dtype=dtype) for k,dtype in
              [('b1_error_mm', np.float32), ('b0a_error_mm', np.float32),
               ('frame_valid', bool), ('contact_frame', bool)]}
    totals = [Accumulator(), Accumulator()]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for begin in range(0, len(rows), 8):
            if time.monotonic()-started > 120: raise TimeoutError('bounded export >120s')
            end = min(begin+8, len(rows))
            batch = {k:v.cuda(non_blocking=True) for k,v in collate([dataset[indices[i]] for i in range(begin,end)]).items()}
            with torch.autocast('cuda', enabled=False):
                prediction = model(batch)['trajectory']
            target = batch['trajectory_target'].float()
            valid = batch['trajectory_valid'].bool() & batch['start_hand_valid'].bool()[:,None]
            errors = [(prediction.float()-target).norm(dim=-1)*1000, target.norm(dim=-1)*1000]
            for name, error, total in zip(('b1_error_mm','b0a_error_mm'), errors, totals):
                values, known = frame_errors(error, valid)
                arrays[name][begin:end] = values
                total.add(error[valid].cpu().numpy())
            arrays['frame_valid'][begin:end] = known
            arrays['contact_frame'][begin:end] = ((batch['interaction_valid'].bool() &
                (batch['contact_target'].float()>panel['metrics']['contact_confidence_threshold'])).any(dim=(1,3))).cpu().numpy()
    reference = read(a.reference)
    metrics = dict(zip(('B1','B0a'), (t.finish() for t in totals)))
    for method in metrics:
        assert abs(metrics[method]['mean_mm']-reference['metrics'][method]['overall']['mean_mm']) < 1e-4
    arrays.update(sequence_id=np.array([r['sequence_id'] for r in rows]),
                  slot=np.array([r['slot'] for r in rows]), tick=np.array([r['tick'] for r in rows]),
                  phase=np.array([r['phase'] for r in rows]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output, **arrays)
    meta = dict(schema='ref2dex.tau-panel-frame-records.v1', anchor_id=anchor['anchor_id'],
        purpose='recover unpersisted paired errors for clustered CI; same historical B1/B0a, no new baseline',
        checkpoint=str(a.checkpoint), step=step, rows=len(rows), sequence_count=len(np.unique(arrays['sequence_id'])),
        metrics=metrics, reference_mean_equivalence_tolerance_mm=1e-4, training_updates=0,
        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            physical_gpu=a.gpu, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),
        timing=dict(elapsed_s=time.monotonic()-started), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__ == '__main__': main()
