"""Recount all preregistered blocks from saved raw/actions; no new simulation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.action_boundary_audit import audit_action_boundaries
from wam_rl_policy_v3.tau_replay_contract import bouts, require_saved_clock
from wam_rl_policy_v3.inspire_contract import ACTIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scoring', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, help='Optional six-case production source-PD cross-check')
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    scoring = json.loads((args.scoring / 'summary.json').read_text())
    physical = json.loads((args.scoring / 'physical-rows.json').read_text())
    if (manifest['status'] != 'COMPLETED' or len(manifest['runs']) != 25
            or scoring['status'] != 'PASS' or scoring['n_cases'] != 25):
        raise ValueError('complete frozen n5 run and independent scoring required')
    scored = {r['label']: r for r in scoring['records']}
    records = []
    for run in sorted(manifest['runs'], key=lambda r: (r['block'], r['chain'])):
        with np.load(args.run / run['label'] / 'trajectory.npz') as archive:
            data = {k: archive[k] for k in archive.files}
        require_saved_clock(data)
        record = audit_action_boundaries(data['raw_action'], data['action'], data['q'][:, 0],
                    manifest['native_initial_audit']['physical_joint_limits'],
                    reported_clipped=run['clipped_controls'],
                    reported_source_rejections=run['source_clipped_controls'], source_chain=run['chain']=='source-pd')
        rows = physical[run['label']]
        if len(rows) != len(data['q']) or [r['tick'] for r in rows] != list(range(len(rows))):
            raise ValueError('physical scoring state clock mismatch')
        spans = bouts([r['held'] for r in rows])
        duration = max(((hi-lo-1)/30 for lo, hi in spans), default=0.)
        ever = any(hi-lo >= 31 for lo, hi in spans)
        final = bool(spans and spans[-1][1] == len(rows) and spans[-1][1]-spans[-1][0] >= 31)
        if (abs(duration-run['longest_hold_elapsed_s']) > 1e-9
                or ever != run['ever_success'] or final != run['final_success']
                or abs(duration-scored[run['label']]['longest_elapsed_s']) > 1e-9):
            raise ValueError('hold/success log mismatch')
        if bool(record['joint_abort_states']) != (run['physical_safety_abort']=='physical_joint_limit'):
            raise ValueError('joint abort reason disagrees with measured q')
        longest = max(spans, key=lambda pair: pair[1]-pair[0]) if spans else None
        # state[t] is before action[t]; action at the last held state is reported
        # separately from controls with both endpoints still inside the hold.
        hold_controls = [] if longest is None else list(range(longest[0], min(longest[1]-1, len(data['action']))))
        clipped_set = set(record['clipped_ticks'])
        record.update(label=run['label'], block=run['block'], chain=run['chain'],
                      longest_hold_elapsed_s=duration, ever_success=ever, final_success=final,
                      physical_abort=run['physical_safety_abort'],
                      longest_hold_state_ticks=None if longest is None else [longest[0], longest[1]-1],
                      longest_hold_internal_controls=len(hold_controls),
                      longest_hold_internal_clipped_controls=sum(t in clipped_set for t in hold_controls),
                      first_clipped_tick=record['clipped_ticks'][0] if record['clipped_ticks'] else None)
        records.append(record)
    chains = {}
    for name in ('source-pd','baseline-gt','calibrated-gt','baseline-wam','calibrated-wam'):
        part = [r for r in records if r['chain']==name]
        if len(part)!=5 or sorted(r['block'] for r in part)!=list(range(5)):
            raise ValueError('five preregistered blocks required for every chain')
        controls = sum(r['controls'] for r in part)
        chains[name] = dict(n=5, controls=controls,
            holds_s=[r['longest_hold_elapsed_s'] for r in part],
            ever_successes=sum(r['ever_success'] for r in part), final_successes=sum(r['final_success'] for r in part),
            **{key:sum(r[key] for r in part) for key in ('clipped_controls','boundary_controls','boundary_controls_cartesian_or_wrist','nominal_joint_outside_states','joint_abort_states')},
            joint_abort_episodes=sum(r['joint_abort_states']>0 for r in part),
            physical_abort_episodes=sum(r['physical_abort'] is not None for r in part))
        chains[name]['clipped_fraction'] = chains[name]['clipped_controls']/controls
    report = dict(status='PASS',input_run=str(args.run.resolve()),input_scoring=str(args.scoring.resolve()),
        protocol=manifest['protocol'],unit='control tick, not action coordinate; joint states include terminal T+1',
        total_controls=sum(r['controls'] for r in records),n=25,chains=chains,records=records,
        limitation='Hold predicates reused from saved independent physical scoring; not a new ground-truth grasp sensor.')
    if args.source_run:
        source_manifest = json.loads((args.source_run/'manifest.json').read_text())
        with np.load(args.source_run/'trajectory.npz') as z:
            source_q = z['q'].copy()
        with np.load(args.source_run/'controls.npz') as z:
            active = z['active'].copy(); target = z['target'].copy(); actual_action = z['action'].copy()
        controller = source_manifest['native_initial_audit']['controller']
        offset = np.asarray(controller['offset'], dtype=np.float64)
        scale = np.asarray(controller['scale'], dtype=np.float64)
        # Independent algebra on recorded nearest-angle targets and live q;
        # do not trust the old clipped_envcontrols counter, which was initialized
        # to zero and relies on source_pd_action rejecting overflow upstream.
        raw = (target.astype(np.float64)-offset)/scale
        raw[..., :6] = (target[..., :6]-source_q[:-1, :, :6]-offset[:6])/scale[:6]
        raw[..., 6:] = 2*raw[..., 6:]-1
        raw = raw[..., ACTIVE][active]
        applied = actual_action[active].astype(np.float64)
        if not np.allclose(applied,np.clip(raw,-1,1),atol=1e-7,rtol=0):
            raise ValueError('production source raw/clip mismatch')
        report['production_source_crosscheck'] = dict(
            run=str(args.source_run.resolve()),active_controls=int(active.sum()),
            substantive_clipped_controls=int((np.abs(raw-applied)>1e-6).any(axis=1).sum()),
            inverse_overflow_controls=int((np.abs(raw)>1+1e-6).any(axis=1).sum()),
            boundary_controls=int((np.abs(applied)>=1-1e-6).any(axis=1).sum()),
            boundary_cartesian_or_wrist_controls=int((np.abs(applied[:,:6])>=1-1e-6).any(axis=1).sum()),
            maximum_roundoff_excess=float(np.maximum(np.abs(raw)-1,0).max()),
            logged_saturation_rejections=source_manifest['source_pd_saturation_rejections'])
        if report['production_source_crosscheck']['substantive_clipped_controls'] or report['production_source_crosscheck']['inverse_overflow_controls']:
            raise ValueError('production source no-clipping claim contradicted')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['| block / chain | controls | longest hold s | ever/final | clip >1e-6 | endpoint ≥1−1e-6 | q-abort states/episodes; DOFs | first clip / q-abort tick | hold-internal clip/controls |',
           '|---|---:|---:|---|---:|---:|---|---|---|']
    for r in records:
        lines.append(f"| {r['label']} | {r['controls']} | {r['longest_hold_elapsed_s']:.6f} | {int(r['ever_success'])}/{int(r['final_success'])} | {r['clipped_controls']} | {r['boundary_controls']} | {r['joint_abort_states']}/{int(r['joint_abort_states']>0)}; {r['joint_abort_indices']} | {r['first_clipped_tick']} / {r['first_joint_abort_tick']} | {r['longest_hold_internal_clipped_controls']}/{r['longest_hold_internal_controls']} |")
    (args.output/'blocks.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('status','total_controls','n','chains','production_source_crosscheck') if k in report}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
