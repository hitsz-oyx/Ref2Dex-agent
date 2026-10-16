"""Audit independent action execution and actual initial states across motion arms."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs=3, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = dict(status='PASS', arms={})
    initial = None
    for run in a.runs:
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['status'] == 'COMPLETED'
        assert manifest['bc_kind'] == 'none' and manifest['base_checkpoint'] is None
        assert manifest['action_kind'] == 'independent-gaussian' and manifest['bc_weight'] == 0
        assert manifest['official_actor_loaded'] is False and manifest['frozen_bc'] is False
        source = (run/'runtime_replay.py').read_text()
        assert 'actor = None' in source and 'state_based_model.pt' not in source
        curves = json.loads((run/'curve.json').read_text())
        execution = max(v for row in curves for v in row['native_execution_error'].values())
        segments = [s for row in curves for s in row['ppo_segments']]
        assert execution < 1e-4
        assert max(s['behavior_logprob_error'] for s in segments) <= 1e-4
        assert all(s['bc_regularization'] == 0 for s in segments)
        assert sum(s['actor_steps'] for s in segments) > 0
        with np.load(run/'first-episode.npz') as trace:
            assert all(np.isfinite(trace[k]).all() for k in trace.files)
            assert not trace['base'].any()
            np.testing.assert_array_equal(trace['action'], np.clip(trace['latent'], -1, 1))
            current = {k: trace[k][0] for k in ['hand', 'object', 'q', 'dq', 'hand_root', 'observation', 'goal', 'success', 'reset']}
        if initial is None:
            initial = current
        differences = {k: float(np.max(np.abs(v.astype(float)-initial[k].astype(float)))) for k, v in current.items()}
        # Exact actual state matching, not merely shared RNG seeds.
        assert max(differences.values()) == 0, differences
        reference_files = sorted(run.glob('motion-input-*.pt'))
        assert len(reference_files) in (13, 15)
        anchor = None
        clips, maxima = [], []
        for path in reference_files:
            packet = torch.load(path, map_location='cpu', weights_only=False)
            assert packet['reference_tick'] == 1 + ((packet['tick']-1)//16)*16
            if packet['tick'] == 1:anchor = packet
            if packet['tick'] in (2, 16):
                assert torch.equal(packet['hand_target_world'], anchor['hand_target_world'])
                assert torch.equal(packet['scene_target_world'], anchor['scene_target_world'])
                torch.testing.assert_close(packet['observation'][:, 561], torch.full((manifest['envs'],), (packet['tick']-1)/48))
            assert float(packet['delta_world'].norm(dim=-1).max()) <= .050001
            assert torch.isfinite(packet['observation']).all() and not packet['observation'][:, -24:].any()
            assert packet['observation'].shape[1] == 586  # saved before remaining-clock append
            if manifest['observation_kind'] == 'motion-none':
                assert not packet['observation'][:, 303:561].any()
            clips.append(packet['clipped_fraction']); maxima.append(packet['unclipped_max_m'])
        if manifest['observation_kind'] == 'motion-wam':
            assert json.loads((run/'motion-freeze-audit.json').read_text())['status'] == 'PASS'
        medians = [float(np.median([s['value_explained_variance'] or 0 for s in row['ppo_segments']])) for row in curves]
        gate = len(curves) >= 25 and sum(v > .05 for v in medians[-5:]) >= 4
        result['arms'][manifest['observation_kind']] = dict(run=str(run), execution_max_abs=execution,
            initial_state_max_abs=differences, updates=len(curves), actor_steps=sum(s['actor_steps'] for s in segments),
            critic_median_ev=medians, learning_qualified=gate,
            all_episode_max_clip_fraction=max(r.get('motion_max_clip_fraction', 0) for r in curves),
            first_reference_max_abs_m=max(maxima), first_reference_max_clip_fraction=max(clips),
            returns=[r['mean_native_episode_return'] for r in curves],
            held=[r['native']['held60_with_lift10cm'] for r in curves],
            reached=[r['native']['official_success'] for r in curves],
            lifted=[sum(x >= .1 for x in r['native']['max_lift_m']) for r in curves])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: {n: v[n] for n in ['updates', 'actor_steps', 'learning_qualified', 'execution_max_abs',
        'first_reference_max_clip_fraction']} for k, v in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
