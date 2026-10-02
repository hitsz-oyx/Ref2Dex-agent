#!/usr/bin/env python3
"""Offline command contract against archived real PD; no new dynamics labels."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def run(args):
    import torch
    from src.task.CmResidual.contact_geometry_actions import sample_programs, executable_candidates, ALLOCATION_TO_OPTION
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError('unique output required')
    begin = time.monotonic()
    record = torch.load(args.records, map_location='cpu', weights_only=False)
    n = len(record['state'])
    bank = record['candidate_actions'][:, :6]
    position = record['state'][:, :18]
    anchor, offset, scale = record['hold_target'], record['pd_offset'], record['pd_scale']
    proposer = torch.Generator().manual_seed(15570+30000)
    weights = sample_programs(n, 'cpu', proposer)
    again = sample_programs(n, 'cpu', torch.Generator().manual_seed(15570+30000))
    if not torch.equal(weights, again):
        raise ValueError('proposal reproducibility')
    before = bank.clone()
    actions = executable_candidates(bank, weights, anchor, position, offset, scale)
    errors = dict(base_reference=float((actions[:, 0]-bank[:, 4]).abs().max()),
                  cup_reference=float((actions[:, 1]-record['candidate_actions'][:, 7]).abs().max()),
                  weight_sum=float((weights.sum(-1)-1).abs().max()))
    if not torch.equal(bank, before):
        raise ValueError('expert bank mutated')
    lower, upper = bank.min(1).values, bank.max(1).values
    for channels in [slice(0, 3), slice(6, 18)]:
        if (actions[:, 2:, channels] < lower[:, None, channels]-2e-6).any() or (actions[:, 2:, channels] > upper[:, None, channels]+2e-6).any():
            raise ValueError('mixture leaves expert coordinate range')
    # Independent native mapping, checked against previously executed real PD.
    raw = actions.clone()
    raw[..., 6:] = (1+raw[..., 6:])*.5
    targets = offset+scale*raw
    targets[..., :6] += position[:, None, :6]
    for dst, src, ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        targets[..., dst] = targets[..., src]*ratio
    errors['fixed_rotation'] = float((targets[:, 1:, 3:6]-anchor[:, None, 3:6]).abs().max())
    real_cup = record['assignment'] == 7
    real_base = record['assignment'] == 4
    if not real_cup.any() or not real_base.any():
        raise ValueError('archived executed reference support required')
    errors['archived_actual_cup_pd'] = float((targets[real_cup, 1]-record['actual_pd_targets'][real_cup, 0]).abs().max())
    errors['archived_actual_base_pd'] = float((targets[real_base, 0]-record['actual_pd_targets'][real_base, 0]).abs().max())
    merged = torch.bincount(torch.tensor(ALLOCATION_TO_OPTION), minlength=8)/10
    if not torch.allclose(merged, torch.tensor([.2,.2,.1,.1,.1,.1,.1,.1])):
        raise ValueError('duplicate slot propensity contract')
    # A corrupt optimizer/proposal must fail before producing executable commands.
    invalid = weights.clone()
    invalid[0, 2, 0, 0] = -1
    try:
        executable_candidates(bank, invalid, anchor, position, offset, scale)
    except ValueError:
        pass
    else:
        raise ValueError('nonconvex programme accepted')
    passed = max(errors.values()) <= 2e-5
    result = dict(run_status='COMPLETED', engineering_passed=passed, source_rows=n,
                  archived_real_cup_matches=int(real_cup.sum()), archived_real_base_matches=int(real_base.sum()),
                  mixture_commands_tested=n*6, max_errors=errors, merged_propensity=merged.tolist(),
                  elapsed_seconds=time.monotonic()-begin,
                  input_sha256={str(args.records.resolve()):sha(args.records),
                                str(Path(__file__).resolve()):sha(Path(__file__)),
                                str(ROOT/'src/task/CmResidual/contact_geometry_actions.py'):sha(ROOT/'src/task/CmResidual/contact_geometry_actions.py')},
                  device_reason='CPU small command algebra and archived PD checks; no NN or simulation',
                  scope='offline command engineering only; no new physical transitions, candidate opportunity or Cm utility evidence')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'input_sha256'}, indent=2))
    if not passed:
        raise ValueError('offline command contract failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
