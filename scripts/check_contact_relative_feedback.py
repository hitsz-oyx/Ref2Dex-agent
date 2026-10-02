#!/usr/bin/env python3
"""GPU engineering on recorded observations; never treats countercommands as truth."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_paired_evaluator_resolution import gpu_admission, sha


def run(args):
    admission = gpu_admission(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = admission['uuid']
    import torch
    from src.task.CmResidual.contact_relative_feedback import (
        feedback_candidates, relative_anchor, verify_translation_asset)
    torch.set_num_threads(2)
    begin = time.monotonic()
    asset = ROOT / 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    contract = verify_translation_asset(asset)
    base = ROOT / 'src/task/CmResidual/research/contact_consequence/output'
    fixtures = [base / 'P-20261002-structured-opportunity-engineering-r1/seed623/records.pt',
                base / 'P-20261002-optimized-contact-native-engineering-r2/seed590/records.pt']
    reports = []
    for path in fixtures:
        b = torch.load(path, map_location='cuda', weights_only=False)
        state = torch.cat((b['state'][:, None], b['future_state'][:, :-1]), 1)
        n = len(state)
        bank = b['expert_bank'].flatten(0, 1)
        anchors = b['hold_target'][:, None].expand(-1, 10, -1).flatten(0, 1)
        relative = relative_anchor(b['state'])[:, None].expand(-1, 10, -1).flatten(0, 1)
        offset, scale = b['pd_offset'], b['pd_scale']
        actual, correction = feedback_candidates(bank, state.flatten(0, 1), anchors, relative,
                                                  offset, scale, 1 / 30)
        # Scalar, metric independent reconstruction of all actuated channels.
        expected = bank[:, 1, None].expand(-1, 4, -1).clone(); expected[:, 0] = bank[:, 4]
        pre = state.flatten(0, 1)
        expected[:, 1:, 3:6] = ((anchors[:, None, 3:6] - pre[:, None, 3:6]
                                - offset[None, None, 3:6]) / scale[None, None, 3:6]).clamp(-1, 1)
        expected[:, 2, :3] = (bank[:, 1, :3] + .5 / 30 * (pre[:, 43:46] - pre[:, 18:21]) / scale[:3]).clamp(-1, 1)
        expected[:, 3, :3] = (bank[:, 1, :3] + .5 * (pre[:, 36:39] - pre[:, :3] - relative) / scale[:3]).clamp(-1, 1)
        error = float((actual - expected).abs().max())
        if error > 2e-6 or actual.abs().max() > 1:
            raise AssertionError('independent native feedback reconstruction')
        sequence = actual.reshape(n, 10, 4, 18)
        if not torch.equal(sequence[:, 0, 3], sequence[:, 0, 1]):
            raise AssertionError('initial position anchor must agree with Cup')
        if not torch.equal(actual[:, 1:, 6:], bank[:, 1, None, 6:].expand(-1, 3, -1)):
            raise AssertionError('nominal fingers changed')
        translated = pre.clone(); translated[:, :3] += .25; translated[:, 36:39] += .25
        shifted, _ = feedback_candidates(bank, translated, anchors, relative, offset, scale, 1 / 30)
        if float((shifted - actual).abs().max()) > 3e-5:
            raise AssertionError('relative world translation invariance')
        # Current-only call has no future-state parameter: poisoning later observations
        # cannot affect the separately evaluated initial/current command.
        poisoned = state.clone(); poisoned[:, 1:] = float('nan')
        now, _ = feedback_candidates(b['expert_bank'][:, 0], poisoned[:, 0], b['hold_target'],
                                    relative_anchor(b['state']), offset, scale, 1 / 30)
        if not torch.equal(now, sequence[:, 0]):
            raise AssertionError('future observation entered initial command')
        changes = (sequence[:, :, 2:4] - sequence[:, :, 1, None]).abs().amax(-1) > 1e-5
        reports.append(dict(path=str(path), sha256=sha(path), rows=n, independent_command_error=error,
            velocity_changed_windows=int(changes[:, :, 0].any(-1).sum()),
            position_changed_windows=int(changes[:, :, 1].any(-1).sum()),
            requested_correction_max_mm=float(correction.abs().max() * 1000),
            first_position_exact_cup=True, fingers_preserved=True, future_poison_passed=True))
    result = dict(run_status='COMPLETED', engineering_passed=True, gpu=admission,
                  asset_sha256=sha(asset), asset_contract=contract, fixtures=reports,
                  elapsed_seconds=time.monotonic()-begin,
                  scope='observed-trajectory countercommands only; no candidate outcome or utility inference')
    if args.output.exists(): raise ValueError('unique engineering result')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=5)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
