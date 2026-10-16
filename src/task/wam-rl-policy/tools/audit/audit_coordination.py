"""Recompute actual-state coordination bonuses and paired held-out outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK.parent/'shadow-finetune/src')]
from shadow_finetune import paths
from video_data_preparation.unidexgrasp_data import object_poses
from wam_rl_policy.coordination import lifted_nearby


def inspect(run):
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED'
    assert manifest['bc_kind'] == 'mlp' and manifest['action_kind'] == 'reference-gaussian'
    assert manifest['bc_weight'] == 0 and manifest['official_actor_loaded'] is False
    assert 'state_based_model.pt' not in (run/'runtime_replay.py').read_text()
    assert json.loads((run/'freeze-audit.json').read_text())['status'] == 'PASS'
    if manifest['coordination_kind'] == 'wam':
        assert json.loads((run/'motion-freeze-audit.json').read_text())['status'] == 'PASS'
    curves = json.loads((run/'curve.json').read_text())
    assert max(v for row in curves for v in row['native_execution_error'].values()) < 1e-4
    if manifest['mode'] == 'train':
        segments = [s for row in curves for s in row['ppo_segments']]
        assert max(s['behavior_logprob_error'] for s in segments) <= 1e-4
        assert sum(s['actor_steps'] for s in segments) > 0
    with np.load(run/'first-episode.npz') as data:
        trace = {k: data[k] for k in data.files}
    assert all(np.isfinite(value).all() for value in trace.values())
    assert np.max(np.abs(trace['reward']-trace['native_reward']-trace['coordination_bonus'])) < 2e-6
    poses = torch.as_tensor(object_poses(trace['object']), dtype=torch.float32)
    hands = torch.as_tensor(trace['hand'])
    initial_z = torch.as_tensor(trace['object'][0, :, 2])
    weight = 0 if manifest['coordination_kind'] == 'none' else manifest['coordination_weight']
    reference = {tick: torch.load(run/('coordination-input-%03d.pt' % tick), weights_only=False, map_location='cpu')
                 for tick in range(1, 200, 16)}
    recomputed = []; eligible = []; phase_admitted = []; reference_changes = []
    for tick in range(1, 200):
        ref = reference[1+((tick-1)//16)*16]
        assert ref['reference_tick'] == 1+((tick-1)//16)*16
        now = (hands[tick]-poses[tick, :, None, :3, 3]) @ poses[tick, :, :3, :3]
        active = ref['enabled'] & lifted_nearby(hands[tick], poses[tick, :, :3, 3], initial_z)
        # Independent formula, fixed reference, actual post-transition state.
        score = torch.exp(-(now-ref['relative_target']).square().sum(-1).mean(-1)/.0018)
        recomputed.append((weight*score*active).numpy()); eligible.append(active.numpy())
        if tick == ref['reference_tick']:
            phase_admitted.extend(ref['admitted'][ref['phase']].tolist())
            reference_changes.extend(ref['target_change'][ref['enabled']].tolist())
    error = float(np.max(np.abs(np.stack(recomputed)-trace['coordination_bonus'])))
    assert error < 2e-5
    np.testing.assert_array_equal(np.stack(eligible), trace['coordination_enabled'])
    for tick in (2, 16):
        packet = torch.load(run/('coordination-input-%03d.pt' % tick), weights_only=False, map_location='cpu')
        torch.testing.assert_close(packet['relative_target'], reference[1]['relative_target'], atol=0, rtol=0)
    held = np.asarray(curves[-1]['native']['longest_goal_hold_with_lift_frames']) >= 60
    with np.load(run/'last-episode.npz') as final_trace:
        reached = final_trace['success'].any(0)
    initial = {k: trace[k][0] for k in ['hand', 'object', 'q', 'dq', 'hand_root', 'observation', 'goal', 'success', 'reset']}
    return dict(run=str(run), mode=manifest['mode'], condition=manifest['coordination_kind'],
        held=int(held.sum()), reached=int(reached.sum()), reach_without_hold=int((reached & ~held).sum()),
        lifted=sum(v >= .1 for v in curves[-1]['native']['max_lift_m']),
        bonus_replay_max_abs=error, first_enabled_fraction=float(trace['coordination_enabled'].mean()),
        first_phase_admission=float(np.mean(phase_admitted)) if phase_admitted else None,
        first_target_change_mean_m=float(np.mean(reference_changes)) if reference_changes else None,
        mean_bonus_by_episode=[row['mean_coordination_bonus'] for row in curves],
        held_by_episode=[row['native']['held60_with_lift10cm'] for row in curves]), initial, held


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--paired', action='store_true', help='First run is the BC baseline on the same evaluation panel')
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    result = dict(status='PASS', runs=[])
    anchor = None
    for run in a.runs:
        report, initial, held = inspect(run)
        if a.paired:
            if anchor is None:anchor = (initial, held)
            differences = {k: float(np.abs(initial[k].astype(float)-anchor[0][k].astype(float)).max()) for k in initial}
            assert max(differences.values()) == 0, differences
            report.update(initial_state_max_abs=differences, both=int((held & anchor[1]).sum()),
                          reference_only=int((anchor[1] & ~held).sum()), new_only=int((held & ~anchor[1]).sum()))
        result['runs'].append(report)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
