#!/usr/bin/env python3
"""Derived FK actual0/4/8 dense-flow sidecar; original assets stay immutable."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/task/cm-interaction-oracle/src'), str(Path(__file__).parent)]
from package_rolling_oracle_asset import Snapshot, sha


def validate_slots(round_record):
    observed = torch.tensor([p is not None for p in round_record['candidates']])
    if len(observed) != 7 or not torch.equal(observed, round_record['candidate_observed']):
        raise ValueError('observed-mask/join mismatch')
    if round_record['actual'] is None:
        raise ValueError('executed-record join absent')
    for k, record in enumerate(round_record['candidates']):
        if record is not None and record['candidate'] != k:
            raise ValueError('candidate join identity mismatch')
    return observed


def to_packet(record):
    h = record['H']
    if not torch.equal(h['native_q0'], h['history'][:, -1, :18]):
        raise ValueError('current q0/history identity mismatch')
    value = {key: h[key] for key in ('history', 'hand_root', 'before',
                                    'before_fingertip_positions', 'before_hand_base_pose')}
    value.update(native_q=record['native_q32'], fingertip_positions=record['tip_positions32'],
                 hand_base_pose=record['hand_base_pose32'])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-run', type=Path, required=True)
    parser.add_argument('--sampling-reference', type=Path,
                        default=ROOT / 'outputs/cm-interaction-oracle/oracle-y-utility-sync-s263-s264/paired_actual_flow.pt')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic(); torch.set_num_threads(2)
    snapshot = Snapshot(); groups = []; checks = []
    report = dict(status='STARTED', physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  device='cuda:0', no_models_or_simulation=True, groups=[], checks=checks)
    output = args.run_dir / 'derived_dense_flows.pt'
    try:
        from geometric_consequence import NominalSurfaceActions
        from oracle_hand_flow import measured_flow_inputs
        if not torch.cuda.is_available():
            raise ValueError('GPU FK required')
        manifest = snapshot.read(args.asset_run / 'manifest.json')
        if manifest['status'] != 'COMPLETED':
            raise ValueError('completed input asset manifest required')
        sampling = snapshot.read(args.sampling_reference)
        old_report = snapshot.read(args.sampling_reference.with_suffix('.json'))
        if old_report['status'] != 'PASS' or old_report['artifact_sha256'] != sha(args.sampling_reference):
            raise ValueError('old sampling archive audit/hash mismatch')
        visual = snapshot.read(ROOT / 'outputs/cm-interaction-oracle/geometric-innovation-s241-r2/manifest.json')['hand_visual_sha256']
        asset_dir = ROOT / 'third_party/DExplore/dexplore/data/assets'
        urdf = asset_dir / 'inspire_hand_new/inspire_hand_right.urdf'
        asset_hashes = {**visual, str(urdf.resolve()): sha(urdf)}
        # Exact equality against the historical audited assets, not merely current hashes.
        for path, expected in sampling['asset_sha256'].items():
            if sha(Path(path)) != expected:
                raise ValueError('historical visual/URDF asset drift: ' + path)
            snapshot.hashes[str(Path(path).resolve())] = expected
        for path, expected in asset_hashes.items():
            if sha(Path(path)) != expected:
                raise ValueError('visual/URDF asset drift: '+path)
            snapshot.hashes[str(Path(path).resolve())] = expected
        code_paths = [Path(__file__).resolve(), Path(__file__).with_name('package_rolling_oracle_asset.py')]
        code_paths += [ROOT / path for path in (
            'src/task/cm-interaction-oracle/src/oracle_hand_flow.py',
            'src/task/cm-interaction-oracle/src/geometric_consequence.py',
            'src/task/CmResidual/dexplore_cm_geometry.py',
            'src/task/CmResidual/surface_execution.py', 'src/task/CmResidual/v118_planner.py')]
        code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
        snapshot.hashes.update(code_hashes)
        bridge = NominalSurfaceActions(urdf, 'cuda:0', seed=sampling['sampling_seed'])
        if set(bridge.samples) != set(sampling['samples']) or any(
                not torch.equal(value.cpu(), sampling['samples'][key]) for key, value in bridge.samples.items()):
            raise ValueError('historical corresponding-point sampling table mismatch')
        for source in manifest['groups']:
            packet = snapshot.read(source['asset'], source['sha256'])
            group = dict(seed=packet['seed'], group=packet['group'], rows=packet['rows'], rounds=[])
            groups.append(group)
            for rd in packet['rounds']:
                if time.monotonic()-begin >= 280:
                    raise TimeoutError('300second GPU budget stop margin')
                observed = validate_slots(rd)
                target = dict(offset=rd['offset'], candidate_observed=observed,
                              candidates=[None]*7, actual=None)
                group['rounds'].append(target)
                joins = [(k, record) for k, record in enumerate(rd['candidates']) if record is not None]
                joins.append(('actual', rd['actual']))
                for key, record in joins:
                    geometry, action, error = measured_flow_inputs(to_packet(record), bridge)
                    normalized = action['Chunk'].cpu().reshape(len(packet['rows']), 2, 120, 3)
                    value = dict(join=dict(seed=packet['seed'], group=packet['group'], offset=rd['offset'], candidate=key),
                                 points_current=geometry[:, :360].cpu().reshape(-1, 120, 3)*.1,
                                 flow_metres=normalized*.02, flow_normalized_002m=normalized,
                                 fk_live_checks=error,
                                 semantics='FK-derived post-treatment actual0→4/4→8; CURRENT object frame')
                    if key == 'actual':
                        value.update(executed_mask32=record['executed_mask32'], in_Z90_mask32=record['in_Z90_mask32'])
                        target['actual'] = value
                    else:
                        target['candidates'][key] = value
                    checks.append(dict(**value['join'], **error))
        for path, expected in snapshot.hashes.items():
            if sha(Path(path)) != expected:
                raise ValueError('immutable source changed: '+path)
        payload = dict(schema='ref2dex.rolling_derived_dense_flow.v1', groups=groups,
                       samples={k:v.cpu() for k,v in bridge.samples.items()}, sampling_seed=bridge.seed,
                       sampling_table_exact_match=True, input_sha256=snapshot.hashes,
                       code_sha256=code_hashes, asset_sha256=asset_hashes,
                       contract='Corresponding120handpoints FK actual0→4/4→8; current object frame; metres and /.02m; never desired/planned flow')
        torch.save(payload, output)
        if output.stat().st_size > 200*1024**2:
            raise ValueError('200MiB sidecar cap exceeded')
        report.update(status='PASS', sampling_table_exact_match=True, input_sha256=snapshot.hashes,
                      code_sha256=code_hashes, asset_sha256=asset_hashes,
                      artifact_sha256=sha(output), bytes=output.stat().st_size,
                      groups=[dict(seed=g['seed'], group=g['group'], anchors=len(g['rows']), rounds=len(g['rounds'])) for g in groups])
    except Exception as exc:
        torch.save(dict(schema='ref2dex.rolling_derived_dense_flow.partial.v1', status='FAILED', groups=groups,
                        checks=checks, input_sha256=snapshot.hashes, error=str(exc)), args.run_dir / 'failed_partial_dense_flows.pt')
        report.update(status='FAILED', error=str(exc), input_sha256=snapshot.hashes)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-begin
        (args.run_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('checks','input_sha256','code_sha256','asset_sha256')}, indent=2))


if __name__ == '__main__':
    main()
