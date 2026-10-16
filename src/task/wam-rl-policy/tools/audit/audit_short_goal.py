"""Replay saved short-goal models and independently recompute held-out errors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from wam_rl_policy.short_goal import ShortGoal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = args.run / 'audit.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    # Match the recorded run's matmul precision and batch shape.
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.run / 'manifest.json').read_text())
    packet = torch.load(Path(manifest['parent']) / 'features/features.pt', weights_only=False)
    data = torch.load(args.run / 'dataset.pt', weights_only=False)
    summary = json.loads((args.run / 'summary.json').read_text())
    forecasts = torch.load(args.run / 'predicted-goal-forecasts.pt', weights_only=False)
    codes = np.array([packet['objects'][i]['object_code'] for i in data['object_id']])
    held = ~data['fit'].numpy()
    assert not set(codes[held]) & set(codes[~held])
    assert np.isfinite(data['target'].numpy()).all()
    ids = torch.arange(128)
    rows = data['rows'][ids]
    report = {'status': 'PASS', 'held_objects': len(set(codes[held])),
              'replay_samples': len(ids), 'arms': {}}

    def macro(errors):
        errors = errors.numpy()
        return float(np.mean([errors[held & (codes == code)].mean() for code in sorted(set(codes[held]))]))

    for arm in ['raw-query', 'wam-mean']:
        saved = torch.load(args.run / (arm + '.pt'), weights_only=False)
        stats = {k: v.cuda() for k, v in saved['normalization'].items()}
        state = (packet['state'][rows].cuda() - stats['state_mean']) / stats['state_scale']
        mean = arm == 'wam-mean'
        tokens = packet['mean' if mean else 'raw'][rows].cuda()
        valid = packet['raw_valid'][rows].cuda()
        if mean:
            tokens = F.layer_norm(tokens.reshape(-1, 3, 768), (768,)).flatten(1)
        else:
            tokens = torch.where(valid[..., None], (tokens - stats['token_mean']) / stats['token_scale'], 0)
        model = ShortGoal(state.shape[-1], tokens.shape[-1], mean=mean).cuda().eval()
        model.load_state_dict(saved['model'])
        with torch.inference_mode():
            replay = model(state, tokens, valid, data['query'][ids].cuda())
        prediction = saved['prediction']
        differences = {}
        for key, value in replay.items():
            expected = prediction[key][ids]
            differences[key] = float((value.cpu() - expected).abs().max())
            torch.testing.assert_close(value.cpu(), expected, atol=2e-5, rtol=2e-5)
        rotation = prediction['rotation']
        orthogonal_error = float((rotation.transpose(-1, -2) @ rotation - torch.eye(3)).abs().max())
        assert orthogonal_error < 1e-4
        assert float(torch.linalg.det(rotation).min()) > .9999
        reconstructed = data['query'] @ rotation.transpose(-1, -2) + prediction['translation'][:, None] - data['query']
        torch.testing.assert_close(reconstructed, prediction['delta'], atol=1e-6, rtol=1e-6)
        endpoint = macro((reconstructed - data['target']).norm(dim=-1).mean(-1) * 1000)
        assert abs(endpoint - summary['models'][arm]['endpoint']['heldout']['all']['point_mm']['macro']) < 1e-4
        metrics = {}
        for domain, start, end, points in [('hand', 0, 66, 11), ('scene', 66, 258, 32)]:
            error = (forecasts[arm][:, start:end] - data['future_labels'][:, start:end]).reshape(-1, 2, points, 3)
            errors = error.norm(dim=-1).mean(-1) * 1000
            for i, horizon in enumerate([16, 32]):
                name = domain + str(horizon)
                metrics[name] = macro(errors[:, i])
                assert abs(metrics[name] - summary['models'][arm]['forecast'][name]['heldout']['all']['macro']) < 1e-4
        report['arms'][arm] = dict(replay_max_abs=differences, rotation_error=orthogonal_error,
                                  endpoint_macro_mm=endpoint, forecast_macro_mm=metrics)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
