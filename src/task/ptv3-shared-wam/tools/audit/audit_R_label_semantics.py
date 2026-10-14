"""Read-only RL660 PD versus realized-state audit; no model or simulator.

This checks label/controller contracts and statistics, not a causal noise floor
or counterfactual physical replay. Outputs are fresh, bounded JSON artifacts.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED, encode_targets
from ptv3_shared_wam.rl_robot import RLRobotWindows
from trajectory_policy.retargeter_data import ACTIVE
from trajectory_policy.pd_retargeter import COUPLING


def summary(value):
    value = np.asarray(value, dtype=np.float64)
    return dict(count=len(value), mean=float(value.mean()),
                median=float(np.median(value)), p95=float(np.percentile(value, 95)),
                maximum=float(value.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    manifest = json.loads((args.source / 'manifest.json').read_text())
    controller = manifest['native_controller']
    offset, scale = (np.asarray(controller[key], dtype=np.float32) for key in ('offset', 'scale'))
    report = dict(scope='label/controller algebra and recorded-state statistics only; no physics replay',
                  device='cpu', CPU_reason='file/label statistics; no neural computation',
                  source=str(args.source.resolve()), splits={})
    for split in ('train', 'val'):
        rows = {}
        def add(key, values):
            rows.setdefault(key, []).append(np.asarray(values))
        records = [r for r in manifest['records'] if r['split'] == split]
        for record in records:
            with np.load(args.source / record['path'] / 'trajectory.npz') as data:
                q, pd, applied = (data[key] for key in ('q', 'pd_targets', 'applied'))
                ticks = np.asarray(record['admitted_ticks'], dtype=np.int64)
                if not len(ticks):
                    continue
                # Same PD converter as Inspire, expressed independently in NumPy.
                reconstructed = offset + scale * np.concatenate((applied[:, :6], (applied[:, 6:] + 1) / 2), -1)
                reconstructed[:, :6] += q[:-1, :6]
                for distal, parent, ratio in COUPLING:
                    reconstructed[:, distal] = reconstructed[:, parent] * ratio
                assert np.max(np.abs(reconstructed - pd)) < 5e-6, record['sequence']
                norm_mm = lambda value: np.linalg.norm(value, axis=-1) * 1000
                add('PD_to_current_wrist_mm', norm_mm(pd[ticks, :3] - q[ticks, :3]))
                add('PD_to_next_wrist_mm', norm_mm(pd[ticks, :3] - q[ticks + 1, :3]))
                add('PD_to_wrist_after8_mm', norm_mm(pd[ticks, :3] - q[ticks + 8, :3]))
                add('wrist_one_tick_motion_mm', norm_mm(q[ticks + 1, :3] - q[ticks, :3]))
                hand_motion = norm_mm(data['hand'][ticks + 1] - data['hand'][ticks]).mean(-1)
                add('hand_one_tick_motion_mm', hand_motion)
                still = hand_motion < 2
                if still.any():
                    add('PD_offset_when_hand_motion_under2mm', norm_mm(pd[ticks[still], :3] - q[ticks[still], :3]))
                later = ticks[ticks >= 1]
                add('PD_target_change_mm_excluding_tick0', norm_mm(pd[later, :3] - pd[later - 1, :3]))
                add('native_wrist_command_change_mm_excluding_tick0', norm_mm(
                    (applied[later, :3] - applied[later - 1, :3]) * scale[:3]))
                add('native_applied_saturated_any_independent', (np.abs(applied[ticks][:, ACTIVE]) >= .999999).any(-1))
                # Realized 18-D q need not obey the controller's target coupling.
                realized = q[ticks + 1]
                projected = realized.copy()
                for distal, parent, ratio in COUPLING:
                    projected[:, distal] = projected[:, parent] * ratio
                add('realized_q_coupling_projection_max_error_rad', np.abs(projected[:, 6:] - realized[:, 6:]).max(-1))
                add('realized_q_coupling_projection_exceeds_001rad', (np.abs(projected[:, 6:] - realized[:, 6:]).max(-1) > .01))
                assert np.array_equal(data['action'], applied)
        report['splits'][split] = dict(sequences=len(records), metrics={
            key: summary(np.concatenate(values)) for key, values in rows.items()})
    # Exercise the actual live dataset seam, rather than reconstructing its labels.
    dataset = RLRobotWindows(args.source, 'val', action_representation=GEOMETRY_LIFTED)
    checks = []
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        tick, sequence = dataset.keys[index]
        data, _ = dataset.sequence(sequence)
        expected = encode_targets(data['pd_targets'][tick + np.arange(8)], data['q'][tick], GEOMETRY_LIFTED)
        assert np.array_equal(sample['action_target'], expected)
        assert np.array_equal(sample['future_q'], data['q'][tick + np.arange(8)])
        assert np.array_equal(sample['native_state'][:18], data['q'][tick])
        checks.append(dict(index=index, sequence=dataset.records[sequence]['sequence'], tick=tick))
    report.update(pass_=True, contract_checks=checks,
                  future_q_semantics='q[t:t+8]: pre-control states used for PD conversion, not q[t+1:t+9] targets',
                  saved_fields=list(data), limitations=[
                      'PD offsets and command changes do not establish irreducible label noise',
                      'coupling projection is not a physical tracking error',
                      'no contact forces or simulator solver-state snapshots in these trajectory arrays'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
