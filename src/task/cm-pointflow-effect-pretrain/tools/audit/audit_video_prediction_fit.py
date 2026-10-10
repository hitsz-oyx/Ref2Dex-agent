#!/usr/bin/env python3
"""Read-only checkpoint diagnosis: per-clip train/development fits and controls."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_points import VideoWindows, collate_video, flow_metrics
from oakink_wm.video_pointworld import VideoPointWorldWM
from oakink_wm.pointworld import VENDOR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    import hashlib
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(args.data / 'manifest.json') != checkpoint['identity']['data_manifest_sha256']:
        raise ValueError('checkpoint data identity mismatch')
    for path, expected in checkpoint['identity']['implementation_sha256'].items():
        if sha(TASK / path) != expected:
            raise ValueError('checkpoint implementation source drift')
    for path, expected in checkpoint['identity']['vendor_sources_sha256'].items():
        if sha(VENDOR / path) != expected:
            raise ValueError('checkpoint vendor source drift')
    stats = checkpoint['identity']['normalization']
    model = VideoPointWorldWM(stats['scene_mean'], stats['scene_std']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    result = dict(schema='ref2dex.video-fit-audit.v1',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  checkpoint_sha256=sha(args.checkpoint), script_sha256=sha(Path(__file__)),
                  fits={}, gpu_samples=[],
                  limitation='current-only ablation is an input intervention on the trained model, not a matched training arm')
    started, last_monitor = time.monotonic(), 0
    with torch.no_grad():
        for split in ('train', 'dev'):
            dataset = VideoWindows(args.data, split)
            for number in range(0, len(dataset), 2):
                samples = [dataset[i] for i in range(number, min(number + 2, len(dataset)))]
                batch = {key: value.cuda() for key, value in collate_video(samples).items()}
                inputs = {key: batch[key] for key in ('xyz', 'features', 'point_valid')}
                current = {key: value.clone() for key, value in inputs.items()}
                current['features'][..., 3:12] = 0
                current['features'][..., 13] = 0
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                    pred, ablated = model(inputs), model(current)
                controls = dict(model=pred, current_only_intervention=ablated,
                                static=torch.zeros_like(pred),
                                cv=batch['velocity'][:, :, None] * batch['elapsed'][:, None, :, None])
                for j in range(len(samples)):
                    scene = dataset.index[number + j][0]
                    for name, values in controls.items():
                        one = {key: value[j:j + 1] for key, value in batch.items()}
                        for key, (total, count) in flow_metrics(values[j:j + 1], one).items():
                            group = result['fits'].setdefault(split + '/' + scene, {})
                            pair = group.setdefault(name + '/' + key, [0., 0])
                            pair[0] += total
                            pair[1] += count
                if time.monotonic() - last_monitor > 3:
                    monitor = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                                      '--format=csv,noheader,nounits'], text=True)
                    result['gpu_samples'].append(dict(elapsed_seconds=time.monotonic() - started,
                                                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                                       nvml_gpu_rows=monitor.splitlines()))
                    last_monitor = time.monotonic()
                if time.monotonic() - started > 90:
                    raise TimeoutError('fit diagnosis GPU budget')
    result['fits'] = {scene: {key: dict(epe_m=value / count if count else None,
                                      supported_point_horizons=count)
                             for key, (value, count) in metrics.items()}
                      for scene, metrics in result['fits'].items()}
    result['elapsed_seconds'] = time.monotonic() - started
    result['peak_memory_mib'] = torch.cuda.max_memory_allocated() / 1024 ** 2
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('fits', 'gpu_samples')}, indent=2))


if __name__ == '__main__':
    main()
