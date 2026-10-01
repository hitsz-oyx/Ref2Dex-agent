#!/usr/bin/env python3
"""Independently reconstruct macro timing, physical labels, and fit normalizers."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha, gpu_admission


def run(args):
    if args.output.exists(): raise ValueError('unique audit output required')
    begin = time.monotonic(); m = json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status'] != 'COMPLETED': raise ValueError('terminal fit required')
    try:
        os.kill(m['pid'], 0)
        raise ValueError('fit owner still running')
    except ProcessLookupError:
        pass
    if any(sha(Path(k)) != v for k, v in m['input_sha256'].items()): raise ValueError('input drift')
    checkpoint = args.run/'contact_to_lift_macro.pt'
    if sha(checkpoint) != m['result']['checkpoint_sha256']: raise ValueError('checkpoint drift')
    admission = gpu_admission(args.gpu); os.environ['CUDA_VISIBLE_DEVICES'] = admission['uuid']
    import torch
    from qualify_contact_to_lift_source import qualify
    from src.task.CmResidual.contact_to_lift_macro import windows
    from src.task.CmResidual.executable_contact_options import INDEPENDENT
    torch.set_num_threads(2)
    p = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data, splits, counts, adequate, _ = qualify(args.source, args.additional, args.additional_audit)
    if not adequate or counts != m['counts']: raise ValueError('supervision/count drift')
    max_error = 0.; audited_rows = 0
    for directory in [args.source, args.additional]:
        source = json.loads((directory/'run_manifest.json').read_text())
        for phase in source['phases']:
            b = torch.load(Path(phase['directory'])/'records.pt', map_location='cpu', weights_only=False)
            d = windows(b); n = len(b['state']); state = b['state']; future = b['future_state']; rest = b['rest_z']
            if not torch.equal(d['history'], b['history']) or not torch.equal(d['native'], b['native_observation'][:, 0]): raise ValueError('pre-input timing')
            raw = torch.cat((b['initial_hand_force'].flatten(-2), b['initial_object_force']), -1)
            raw = raw/(b['mass_kg'][:, None]*b['gravity_magnitude'])
            phys = torch.cat((raw.sign()*raw.abs().log1p(), b['initial_clearance'][:, None],
                              (state[:, 38]-rest).clamp_min(0)[:, None]), -1)
            goal = b['actual_pd_targets'][:, 0, list(INDEPENDENT)] - state[:, list(INDEPENDENT)]
            choice = b['assignment']; actor = choice.clone(); actor[choice == 6] = 4; actor[choice == 7] = 1
            law = torch.zeros(n, 7); law[torch.arange(n), actor] = 1.; law[:, 6] = (choice >= 6).float()
            # Re-derive net-force presence from actual logged forces, rather than copied bits.
            weight = b['mass_kg'][:, None]*b['gravity_magnitude']
            hand = b['future_hand_force'].norm(dim=-1).amax(-1)/weight
            obj = b['future_object_force'].norm(dim=-1)/weight
            pair = (hand > .1) & (obj > .1)
            clear = b['future_clearance'] >= .002
            support = pair[:, -3:].all(-1) & clear[:, -3:].all(-1)
            last_height = (future[:, -3:, 38].amin(-1)-rest).clamp_min(0)
            loss_event = torch.zeros(n, dtype=torch.bool); seen = b['initial_clearance'] >= .002
            for step in range(10):
                loss_event |= seen & ~clear[:, step]; seen |= clear[:, step]
            target = torch.cat(((future[:, :, 38]-state[:, None, 38])/.01,
                                (b['future_clearance']-b['initial_clearance'][:, None])/.01,
                                pair.float(), support[:, None].float(), loss_event[:, None].float(),
                                (support & (last_height >= .03))[:, None].float(),
                                ((last_height*support-(state[:, 38]-rest).clamp_min(0))/.01)[:, None]), -1)
            for name, expected in [('physical', phys), ('goal', goal), ('law', law), ('target', target)]:
                error = float((d[name]-expected).abs().max()); max_error = max(max_error, error)
                if error > 2e-6: raise ValueError('macro reconstruction '+name)
            audited_rows += n
    normalizer_error = 0.
    for name in ['history', 'physical', 'native', 'goal', 'law']:
        x = data[name][splits['fit']].cuda(); dims = (0, 1) if name == 'history' else 0
        mean = x.mean(dims); std = x.std(dims, unbiased=False).clamp_min(.001)
        for key, value in [(name+'_mean', mean), (name+'_std', std)]:
            error = float((p['normalization'][key]-value.cpu()).abs().max()); normalizer_error = max(normalizer_error, error)
            if error > 2e-6: raise ValueError('fit-only normalizer '+key)
    if any(len(p['models'][mode]) != 3 for mode in ['cm', 'state_only', 'shuffled']): raise ValueError('ensemble size')
    if len(m['models']) != 9 or any(r['updates'] != 1000 for r in m['models']): raise ValueError('matched fit budget')
    result = dict(status='COMPLETED', own_fit_pid_exited=True, input_hashes_unchanged=True,
                  rows=audited_rows, counts=counts, reconstruction_max_error=max_error,
                  fit_only_normalization_max_error=normalizer_error, checkpoint_sha256=sha(checkpoint),
                  elapsed_seconds=time.monotonic()-begin, cumulative_seconds=m['result']['cumulative_seconds']+time.monotonic()-begin,
                  scoped_bytes=m['result']['scoped_bytes'], gpu=admission, information_result=m['result'],
                  scope='pre-only timing, actual force labels, feedback law and fit-only normalization; no controller/RL/stable grasp claim')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['gpu', 'information_result', 'counts']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'source', 'additional', 'additional-audit', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0); run(p.parse_args())
