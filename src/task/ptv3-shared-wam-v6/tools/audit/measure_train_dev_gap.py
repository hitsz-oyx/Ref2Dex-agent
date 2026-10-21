"""Bounded fixed-TRAIN checkpoint evaluation with the production DEV evaluator."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
sys.path.insert(0, str(TASK/'tools/run'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.data import V6Windows
from ptv3_shared_wam_v6.model import SharedWAMV6
from train_pretrain import evaluate_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'stats', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    config = json.loads(args.config.read_text())
    if not config.get('strict_fp32') or config.get('amp') or config.get('tf32'):
        raise ValueError('this diagnostic compares checkpoints under strict FP32')
    if not torch.cuda.is_available():
        raise ValueError('GPU inference required')
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(170)
    if torch.cuda.mem_get_info()[0] < 8*2**30:
        raise ValueError('at least8GiB free on the explicitly selected GPU required')
    stats = json.loads(args.stats.read_text())
    dataset = V6Windows(args.manifest, 'train',
                        geometry_gt_object=config.get('geometry_object_source') == 'gt_dense2048')
    dataset.pools = {'1': dataset.pools['1']}
    eval_config = dict(config, validation_seed=170, validation_samples=32,
                       validation_microbatch_size=2)
    indices = np.random.default_rng(170).choice(dataset.pools['1'], 32, replace=False).tolist()
    report = dict(status='RUNNING', scope='fixed32 GRAB TRAIN examples, inference only; actual production evaluator',
                  seed=170, indices=indices, python=sys.version, torch=torch.__version__,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  checkpoints=[])
    start = time.monotonic()
    for path in args.checkpoints:
        model = SharedWAMV6(stats, config).cuda().eval()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        step = int(saved['step']);del saved;gc.collect()
        torch.cuda.reset_peak_memory_stats()
        evaluation = evaluate_condition(model, dataset, eval_config, 'cuda:0')
        source = evaluation['sources']['1']
        metrics = source['metrics']
        summary = dict(hand_mm=source['hand_epe_mm'],
            object_h08_mm=metrics['scene/h08']['mean'], object_h24_mm=metrics['scene/h24']['mean'],
            object_prefix8_mm=sum(metrics[f'scene/h{t:02d}']['mean'] for t in range(1, 9))/8,
            hand_prefix8_mm=sum(metrics[f'hand/h{t:02d}']['mean'] for t in range(1, 9))/8,
            geometry_relative_mm=metrics['geometry/contact_relative']['mean'],
            objective=source['objective'], components=source['components'])
        report['checkpoints'].append(dict(path=str(path.resolve()), step=step,
            precision_runtime=model.precision_runtime, evaluation=evaluation, summary=summary,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        print(json.dumps(dict(step=step, summary=summary)), flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    report.update(status='COMPLETED', elapsed_s=time.monotonic()-start)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
