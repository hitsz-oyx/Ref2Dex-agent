"""Compare warm-start/current weights on a larger DEV panel unused in selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Retain the actual completed file even if production later prunes aliases.
    snapshot = args.output / 'current-snapshot.pt'
    os.link(args.run.joinpath('latest.pt').resolve(strict=True), snapshot)
    sys.path.insert(0, str(args.code_root / 'src/task/ptv3-shared-wam-v4/src'))
    from ptv3_shared_wam_v4 import paths
    from ptv3_shared_wam_v4.data import NineWindows, DevInputCache
    from ptv3_shared_wam_v4.model import SharedWAMV4
    from ptv3_shared_wam_v4.training import evaluate
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    current = torch.load(snapshot, map_location='cpu', weights_only=False)
    config, stats = current['config'], current['stats']
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']
    recipe = stats['recipe']
    dataset = NineWindows(recipe['main3_root'], recipe['native_root'], 'dev', recipe)
    selection_panel = json.loads((args.run / 'validation-panel.json').read_text())
    candidates = dataset.panel(1024, config['validation_seed'])
    panel = {}
    for source in selection_panel:
        excluded, selected = set(selection_panel[source]), []
        for index in candidates[source]:
            if index not in excluded:
                selected.append(index)
                excluded.add(index)
            if len(selected) == 128:
                break
        if len(selected) != 128:
            raise ValueError('128 unique non-selection DEV windows per source required')
        panel[source] = selected
    (args.output / 'panel.json').write_text(json.dumps(panel, indent=2) + '\n')
    microbatch = config.get('validation_microbatch_size', 8)
    cache = DevInputCache(dataset, panel, microbatch)
    model = SharedWAMV4(stats, config).cuda().eval().requires_grad_(False)
    baseline = torch.load(args.run / 'step0.pt', map_location='cpu', weights_only=False)
    for name in ('hand', 'scene'):
        if baseline['stats'][name] != stats[name]:
            raise ValueError('normalization changed between weights')
    model.load_state_dict(baseline['model'], strict=True)
    del baseline
    original = evaluate(model, dataset, selection_panel, config, torch.device('cuda:0'))
    baseline_score = json.loads((args.run / 'curve.json').read_text())[0]['objective']
    if abs(original['objective'] - baseline_score) > 1e-5:
        raise ValueError('original selection baseline did not reproduce')
    reports = {'baseline': evaluate(model, dataset, panel, config, torch.device('cuda:0'), input_cache=cache)}
    model.load_state_dict(current['model'], strict=True)
    reports['current'] = evaluate(model, dataset, panel, config, torch.device('cuda:0'), input_cache=cache)
    result = dict(work_version='wam-v4-disjoint-dev-panel.v1', current_step=current['step'],
                  source_counts={s: len(v) for s, v in panel.items()}, reports=reports,
                  original_baseline_reproduced=original['objective'], cache_bytes=cache.bytes,
                  code_root=str(args.code_root), dependency_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=args.code_root, text=True).strip(),
                  python=sys.version, torch=torch.__version__,
                  elapsed_s=time.monotonic() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  scope='same DEV split; disjoint from selection sample IDs; no optimizer or selection changes')
    with (args.output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(step=current['step'], baseline=reports['baseline']['objective'],
                          current=reports['current']['objective'], elapsed_s=result['elapsed_s'])))


if __name__ == '__main__':
    main()
