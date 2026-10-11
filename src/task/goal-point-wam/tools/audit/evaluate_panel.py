#!/usr/bin/env python3
"""Evaluate WAM and identity effects on saved source-specific diagnostic windows."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import GoalPointWAM, GoalWindows, evaluate, atomic_json, digest
from oakink_wm.multisource import MixedWindows, source_draw


class StaticWorld(torch.nn.Module):
    def forward(self, batch):
        b, o = batch['object_valid'].shape
        return dict(trajectory=torch.zeros_like(batch['trajectory_target']),
                    translation=batch['points'].new_zeros((b, o, 24, 3)),
                    rotation=torch.eye(3, device=batch['points'].device).expand(b, o, 24, 3, 3))


class HistoricalMeanAction(torch.nn.Module):
    """Replay the old checkpoint's original 24-token contract for comparison."""
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
    def forward(self, action, valid):
        encoded, _ = self.adapter(action, valid)
        encoded = encoded.reshape(len(action), 24, 2, 11, 128)
        counts = valid.sum((2, 3)).clamp_min(1).to(encoded.dtype)
        return encoded.sum((2, 3)) / counts[..., None], valid.any((2, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=250)
    parser.add_argument('--samples', type=int, default=96)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    stats = json.loads(args.stats.read_text())
    if state['identity']['stats_sha256'] != digest(args.stats): raise ValueError('stats drift')
    model = GoalPointWAM(stats, state['config']).to(device)
    model.load_state_dict(state['model'], strict=True)
    old = torch.load(args.previous, map_location='cpu', weights_only=False)
    previous = GoalPointWAM(stats, old['config']).to(device)
    previous.load_state_dict(old['model'], strict=True)
    previous.action = HistoricalMeanAction(previous.action)
    del state['model'], old['model']
    static = StaticWorld().to(device)
    bases = {split: MixedWindows(args.data, split) for split in ('val', 'test')}
    rng = np.random.default_rng(args.seed)
    result = dict(schema='goal-point-wam.matched-diagnostic.v1', seed=args.seed,
                  samples_per_source=args.samples, checkpoint=str(args.checkpoint),
                  checkpoint_sha256=digest(args.checkpoint), checkpoint_step=state['step'],
                  previous_sha256=digest(args.previous), data_sha256=digest(args.data / 'processed/manifest.json'),
                  evaluator_sha256=digest(Path(__file__)), panels={})
    source_names = tuple(source['name'] for source in bases['test'].meta['sources'])
    for source_id, name in enumerate(source_names):
        base = bases['test']
        split = 'test'
        if base.offsets[source_id + 1] == base.offsets[source_id]:
            base, split = bases['val'], 'val'
        indices = base.offsets[source_id] + source_draw(base.sources[source_id], args.samples, rng)
        dataset = GoalWindows(base)
        panel = dict(split=split, indices=indices.tolist())
        for label, learner in (('aligned', model), ('previous', previous), ('static', static)):
            measured = evaluate(learner, dataset, indices, 2, 0, device)
            panel[label] = {key: value for key, value in measured.items() if key.startswith('source/' + name + '/')}
            if label == 'static': panel[label].pop('source/' + name + '/trajectory_epe_m')
        result['panels'][name] = panel
        print(json.dumps(dict(source=name, split=split, metrics={k: v for k,v in panel.items() if k != 'indices'})), flush=True)
    atomic_json(args.output, result)


if __name__ == '__main__': main()
