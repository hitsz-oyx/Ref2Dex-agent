#!/usr/bin/env python3
"""Snapshot completed rolling groups into a read-only, explicit-observation asset.

CPU file/label processing only. No simulator, fitting, future-label imputation,
or writes into the source run. A group completion file is the admission gate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
from intervention import physical_targets
from oracle_y_utility import CANDIDATES, stable_grasp_z, utility
from rolling_control import OFFSETS, TOLERANCE
from rolling_y_audit import failure_events, short_y


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Snapshot:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        before = sha(path)
        if expected is not None and before != expected:
            raise ValueError('recorded input hash mismatch: ' + str(path))
        value = (torch.load(path, map_location='cpu', weights_only=False)
                 if path.suffix == '.pt' else json.loads(path.read_text()))
        if sha(path) != before:
            raise ValueError('input changed while reading: ' + str(path))
        self.hashes[str(path)] = before
        return value


def panel_record(panel, rows, candidate, semantics):
    """Retain the raw physics and exact inherited readouts without new thresholds."""
    if not panel['valid_steps'][rows].all():
        raise ValueError('incomplete candidate window')
    if not (panel['full_world_prefix_errors'] <= TOLERANCE).all():
        raise ValueError('prefix replay failure')
    before = panel['before'][rows]
    trajectory = panel['trajectory'][rows]
    if not torch.isfinite(trajectory).all():
        raise ValueError('nonfinite physical trajectory')
    y, risk = short_y(before[:, 2], before[:, 71] > .5,
                      panel['height'][rows, :32], panel['pair'][rows, :32],
                      panel['rest_height'][rows])
    ei = physical_targets(before, trajectory[:, 7])
    tips = panel['fingertip_positions'][rows]
    initial_tips = panel['before_fingertip_positions'][rows]
    # These are five measured tips in WORLD axes, not sampled hand-surface flow.
    flow = tips[:, :8] - initial_tips[:, None]
    record = dict(candidate=candidate, semantics=semantics,
                  H={k: panel[k][rows].clone() for k in
                     ('before', 'history', 'actor_obs', 'hand_root',
                      'before_fingertip_positions', 'before_hand_base_pose')},
                  a_base=panel['base_actions'][rows].clone(),
                  native_actions=panel['actions'][rows].clone(),
                  pd_targets=panel['pd_targets'][rows].clone(),
                  delta_a=panel['delta'][candidate].clone() if candidate is not None else None,
                  F_actual_tip_displacement=flow.clone(),
                  F_actual_surface=None,
                  E8=ei[:, :12], I8=ei[:, 12:],
                  physical32=trajectory.clone(),
                  native_q32=panel['native_q'][rows].clone(),
                  tip_positions32=tips.clone(),
                  hand_base_pose32=panel['hand_base_pose'][rows].clone(),
                  height=panel['height'][rows].clone(), pair=panel['pair'][rows].clone(),
                  valid_steps=panel['valid_steps'][rows].clone(),
                  Y32=y, current_risk=risk, U=utility(y),
                  clipped_steps=panel['clipped_steps'][rows].clone(),
                  Z_candidate=None,
                  missing_fields=['sampled_hand_surface_positions_or_flow',
                                  'candidate_90step_Z'],
                  provenance={key: panel[key] for key in
                              ('initial_fingerprint', 'simulation_contract',
                               'model_fingerprint', 'rms_fingerprint')},
                  control_dt=panel['control_dt'])
    return record


def validate_plan(plan, records, rows):
    certified = bool(plan['baseline_upper_bound_certificate'])
    observed = [k for k, r in enumerate(records) if r is not None]
    if observed != ([0] if certified else list(range(7))):
        raise ValueError('candidate missingness does not match certificate')
    y = torch.stack([records[k]['Y32'] for k in observed], 1)
    scores = utility(y)
    if y.tolist() != plan['y'] or scores.tolist() != plan['utility']:
        raise ValueError('exact Y/U replay mismatch')
    choices = torch.tensor(plan['choices'])[rows]
    if certified:
        if not (scores == 1.25).all() or choices.any():
            raise ValueError('invalid baseline certificate')
    elif not torch.equal(choices, scores.argmax(1)):
        raise ValueError('selection replay mismatch')
    return choices


def package_group(source, group_path, snapshot):
    group = snapshot.read(group_path)
    rows = torch.tensor(group['rows'], dtype=torch.long)
    if [d['offset'] for d in group['decisions']] != list(OFFSETS):
        raise ValueError('group has incomplete replan sequence')
    rounds = []
    reference = Path(group['baseline_dir'])
    for decision in group['decisions']:
        offset = decision['offset']
        prefix = f"s{group['seed']}-g{group['group']}-t{offset:02d}"
        plan = snapshot.read(source / (prefix + '-plan.json'))
        if plan['rows'] != rows.tolist():
            raise ValueError('plan cohort mismatch')
        if sha(reference / 'trace.pt') != plan['source_prefix_sha256']:
            raise ValueError('actual predecessor trace hash mismatch')
        snapshot.hashes[str((reference / 'trace.pt').resolve())] = plan['source_prefix_sha256']
        records = [None] * 7
        for path, expected in plan['candidate_panel_sha256'].items():
            panel = snapshot.read(path, expected)
            k = panel['candidate']
            if records[k] is not None or panel['rolling_offset'] != offset:
                raise ValueError('duplicate/wrong candidate identity')
            snapshot.read(Path(path).parent / 'run_manifest.json')
            result = snapshot.read(Path(path).parent / 'result.json')
            if result['output_sha256'] != expected:
                raise ValueError('candidate completion hash mismatch')
            records[k] = panel_record(panel, rows, k, 'speculative_same_current_state_fork32')
        choices = validate_plan(plan, records, rows)
        for record in records:
            if record is not None:
                if record['provenance'] != records[0]['provenance']:
                    raise ValueError('candidate simulation/model provenance mismatch')
                for key in ('before', 'history', 'actor_obs', 'hand_root'):
                    if not torch.allclose(record['H'][key], records[0]['H'][key], atol=1e-4, rtol=0):
                        raise ValueError('candidate current H mismatch')
        actual_dir = Path(decision['actual_dir'])
        actual_panel = snapshot.read(actual_dir / 'panel.pt')
        actual = panel_record(actual_panel, rows, None, 'executed_first8_then_speculative_suffix')
        actual['delta_a'] = actual_panel['delta'][choices].clone()
        if not torch.equal((actual['a_base'] + actual['delta_a'][:, None]).clamp(-1, 1), actual['native_actions']):
            raise ValueError('native chosen action wiring mismatch')
        for key in ('before', 'history', 'actor_obs', 'hand_root'):
            if not torch.allclose(actual['H'][key], records[0]['H'][key], atol=1e-4, rtol=0):
                raise ValueError('executed current H mismatch')
        actual['executed_mask32'] = torch.arange(32) < 8
        actual['in_Z90_mask32'] = torch.arange(32) < min(8, 90-offset)
        rounds.append(dict(offset=offset, choices=choices, candidates=records,
                           candidate_observed=torch.tensor([r is not None for r in records]),
                           missing_reason=[None if r is not None else 'baseline_U_max_certificate_pruned'
                                           for r in records],
                           actual=actual, plan_path=str(source / (prefix + '-plan.json'))))
        reference = actual_dir
    trace = snapshot.read(group['final_trace'])
    origin = group['origin']
    factual = {k: trace[k][origin:origin+90, rows].transpose(0, 1).clone()
               for k in ('physical', 'after_physical', 'dof', 'root', 'action', 'base_action', 'done')}
    if factual['after_physical'].shape[1] != 90 or factual['done'].any():
        raise ValueError('incomplete factual Z trajectory')
    rest = torch.tensor(group['rest'])
    z, details = stable_grasp_z(factual['after_physical'][:, :, 2],
                              factual['after_physical'][:, :, 71] > .5, rest)
    if z.tolist() != group['rolling'] or details['first_stable_step'].tolist() != group['rolling_qualification']:
        raise ValueError('final factual Z mismatch')
    if details['dropped_after_qualification'].tolist() != group['rolling_dropped']:
        raise ValueError('final factual drop mismatch')
    for rd in rounds:
        start = rd['offset']; end = min(start+8, 90)
        rd['actual']['H']['native_q0'] = factual['dof'][:, start, :18].clone()
        for record in rd['candidates']:
            if record is not None:
                record['H']['native_q0'] = factual['dof'][:, start, :18].clone()
        if not torch.equal(factual['action'][:, start:end], rd['actual']['native_actions'][:, :end-start]):
            raise ValueError('final actual trace action continuity mismatch')
        if not torch.equal(factual['after_physical'][:, start:end, 2], rd['actual']['height'][:, :end-start]):
            raise ValueError('final actual trace height continuity mismatch')
        if not torch.equal(factual['after_physical'][:, start:end], rd['actual']['physical32'][:, :end-start]):
            raise ValueError('final actual trace full physical continuity mismatch')
    events = [failure_events(h.numpy(), p.numpy(), float(r), int(q))
              for h, p, r, q in zip(factual['after_physical'][:, :, 2],
                                    factual['after_physical'][:, :, 71] > .5,
                                    rest, details['first_stable_step'])]
    return dict(schema='ref2dex.rolling_oracle_asset.v1', seed=group['seed'],
                group=group['group'], rows=rows, origin=origin, rest_height=rest,
                motion_id=torch.tensor(group['motion']), rounds=rounds,
                factual90=factual, Z_rolling=z, Z_baseline=torch.tensor(group['baseline']),
                qualification=details, failure_events=events,
                source_group_result=str(group_path.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = args.source_run.resolve()
    paths = sorted(source.glob('s*-g*-result.json'))
    if not paths:
        raise ValueError('no completed groups; no partial groups admitted')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    snapshot = Snapshot(); summaries = []
    code_paths = [Path(__file__).resolve()] + [ROOT / 'src/task/cm-interaction-oracle/src' / name
                                              for name in ('intervention.py', 'oracle_y_utility.py',
                                                           'rolling_control.py', 'rolling_y_audit.py')]
    for path in code_paths:
        snapshot.hashes[str(path)] = sha(path)
    for path in paths:
        if time.monotonic()-started > 540:
            raise TimeoutError('bounded CPU packaging deadline')
        packet = package_group(source, path, snapshot)
        out = args.run_dir / (path.stem.replace('-result', '') + '-asset.pt')
        torch.save(packet, out)
        if sum(p.stat().st_size for p in args.run_dir.iterdir() if p.is_file()) > 1024**3:
            raise ValueError('1GiB asset cap exceeded')
        summaries.append(dict(seed=packet['seed'], group=packet['group'],
                              anchors=len(packet['rows']), rounds=len(packet['rounds']),
                              observed_candidate_slots=sum(int(r['candidate_observed'].sum()) for r in packet['rounds']),
                              asset=str(out.resolve()), sha256=sha(out)))
    # Completed source files must still be immutable at the end of the snapshot.
    for path, expected in snapshot.hashes.items():
        if sha(Path(path)) != expected:
            raise ValueError('completed input changed during snapshot: '+path)
    report = dict(schema='ref2dex.rolling_oracle_asset.v1', status='COMPLETED',
                  snapshot_utc=datetime.now(timezone.utc).isoformat(), source_run=str(source),
                  admission='completed_group_result_only; source may still be running',
                  groups=summaries, input_sha256=snapshot.hashes,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  tool_sha256=sha(Path(__file__)), device='cpu',
                  code_sha256={str(path): snapshot.hashes[str(path)] for path in code_paths},
                  device_reason='file packaging and exact labels only; no neural computation',
                  elapsed_seconds=time.monotonic()-started,
                  limitations=['post-treatment measured tips are not causal planning inputs',
                               'sampled surface flow absent; raw q/root retained for later FK',
                               'candidate Z absent: only32step speculative lookahead',
                               'future actor feedback a_base[:,1:] is post-treatment',
                               'no scientific gate or predictor/noise fitting performed'])
    (args.run_dir / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
