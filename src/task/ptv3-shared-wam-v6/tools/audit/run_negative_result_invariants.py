"""CPU invariant checks. PASS requires both a positive control and a rejected mutant."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6 import paths
import torch
import numpy as np
from ptv3_shared_wam_v6.negative_result_invariants import basic_checks, v6_model_checks, verified, legacy_qualification_timing
from audit_action_reset_cpu import action_audit, reset_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = dict(device='cpu', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  python=sys.executable, torch=torch.__version__, checks=[], errors={})
    for name, fn in [('basic', basic_checks), ('v6_model', v6_model_checks)]:
        try:
            report['checks'].extend(fn())
        except Exception:
            report['errors'][name] = traceback.format_exc()
    try:
        report['action'], check = action_audit()
        report['checks'].append(check)
    except Exception:
        report['errors']['action'] = traceback.format_exc()
    try:
        report['reset'] = reset_replay()
        report['reset_mutant'] = reset_replay(mutate=True)
        report['checks'].append(verified('two-env interleaved reset isolation at steps 2/4',
            report['reset']['status'], 'PASS', report['reset_mutant']['status'],
            'clear untouched environment history during indexed reset'))
    except Exception:
        report['errors']['reset'] = traceback.format_exc()
    try:
        report['legacy_p0_timing'], check = legacy_qualification_timing()
        report['checks'].append(check)
    except Exception:
        report['errors']['legacy_timing'] = traceback.format_exc()
    report['status'] = 'PASS' if not report['errors'] and all(c['status'] == 'PASS' for c in report['checks']) else 'FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')
    print(json.dumps(dict(status=report['status'], checks=len(report['checks']),
                         failed=[c['name'] for c in report['checks'] if c['status'] != 'PASS'],
                         errors=report['errors'], output=str(args.output))))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
