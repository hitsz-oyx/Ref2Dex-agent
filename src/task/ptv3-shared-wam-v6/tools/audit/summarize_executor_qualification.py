"""Summarize paired executor runs; unmeasured quantities remain JSON null.

No GPU, policy fitting or simulation is performed. Metrics use resulting states
after actually completed controls, censored at each robot reference prefix.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ARMS = ('replay', 'gt', 'healthy', 'flow')
THRESHOLDS = dict(hz=30, lift_m=.05, palm_object_distance_m=.20,
                  contact_body_force_n=.10, minimum_contact_bodies=2,
                  hold_controls=30, horizontal_push_m=.05, tilt_degrees=45.,
                  early_contact_lead_controls=3, loss_controls=3,
                  arrival_error_m=.05, replay_success_gate=.80,
                  gt_fraction_of_replay_gate=.80, predicted_deficit_pp=20.)


def runs(values):
    values = np.asarray(values, bool)
    padded = np.r_[False, values, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def longest(values):
    return max((b-a for a, b in runs(values)), default=0)


def first_true(values):
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else None


def natural_recovery(hold):
    """A qualified hold, >=3 lost controls, then another qualified hold."""
    bouts = [(a, b) for a, b in runs(hold) if b-a >= THRESHOLDS['hold_controls']]
    opportunities = 0
    recoveries = 0
    for index, (_, end) in enumerate(bouts):
        next_start = bouts[index+1][0] if index+1 < len(bouts) else len(hold)
        # Brief contact flickers must not erase a later sustained loss before
        # the next qualified hold. Count each qualified-to-qualified interval
        # once, plus a possible unrecovered terminal loss.
        if longest(~hold[end:next_start]) >= THRESHOLDS['loss_controls']:
            opportunities += 1
            recoveries += int(index+1 < len(bouts))
    return dict(loss_opportunities=opportunities, recovered_losses=recoveries,
                recovery_rate=recoveries/opportunities if opportunities else None,
                perturbation_recovery=None, perturbation_reason='no forced perturbation was performed')


def wilson(successes, count):
    if not count: return None
    z = 1.959963984540054
    p = successes/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [float(center-half), float(center+half)]


def compare_initials(root):
    loaded = {}
    for arm in ARMS:
        path = root/arm/'initial-state.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                loaded[arm] = {k: z[k].copy() for k in z.files}
    base = 'replay' if 'replay' in loaded else next(iter(loaded), None)
    result = dict(reference_arm=base, comparisons={}, all_available_arms_identical=None)
    if base is None: return result
    for arm, arrays in loaded.items():
        checks = {}
        for key in sorted(set(loaded[base]) | set(arrays)):
            a, b = loaded[base].get(key), arrays.get(key)
            same_shape = a is not None and b is not None and a.shape == b.shape
            exact = same_shape and a.dtype == b.dtype and np.array_equal(a, b)
            delta = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if same_shape and a.size and a.dtype.kind in 'biuf' else None
            checks[key] = dict(exact=bool(exact), reference_shape=list(a.shape) if a is not None else None,
                               shape=list(b.shape) if b is not None else None, maximum_absolute_difference=delta)
        result['comparisons'][arm] = dict(all_arrays_exact=all(c['exact'] for c in checks.values()), arrays=checks)
    result['all_available_arms_identical'] = all(x['all_arrays_exact'] for x in result['comparisons'].values())
    result['four_arms_available'] = len(loaded) == len(ARMS)
    return result


def case_metrics(trace, index, row, completed):
    controls = min(int(row['reference_controls']), completed, len(trace['objects'])-1)
    if controls <= 0: raise ValueError('no completed control transition')
    # State[t+1] is the observed result of command[t]. The final stored decision
    # command[completed] was not applied and is never included in action metrics.
    window = slice(1, controls+1)
    objects = trace['objects'][:controls+1, index]
    hand = trace['hand'][:controls+1, index]
    if hand.shape[1:] != (11, 3): raise ValueError('native world-coordinate hand11 required')
    lift_all = objects[:, 2] - objects[0, 2]
    lift = lift_all[window]
    palm_distance = np.linalg.norm(hand[window, 0]-objects[window, :3], axis=-1)
    contact = trace['contact'][window, index]
    if contact.shape[1:] != (5, 3): raise ValueError('five native finger contact bodies required')
    if 'contact_force_valid' in trace:
        valid = trace['contact_force_valid'][window, index].astype(bool)
        validity_source = 'recorded trace.contact_force_valid'
    else:
        valid = np.ones(controls, bool)
        validity_source = 'inferred host contract: reset step0 invalid; no reset and each post-step contact valid; field absent in this loaded runner'
    contact_count = (np.linalg.norm(contact, axis=-1) > THRESHOLDS['contact_body_force_n']).sum(-1)
    multi_contact = valid & (contact_count >= THRESHOLDS['minimum_contact_bodies'])
    supported = trace['supported'][window, index].astype(bool)
    hold = (lift >= THRESHOLDS['lift_m']) & ~supported & (palm_distance <= THRESHOLDS['palm_object_distance_m']) & multi_contact
    lifted = lift >= THRESHOLDS['lift_m']
    command = trace['command'][:controls, index]
    if command.shape[-1] != 12: raise ValueError('12 actual requested active controls required')
    xyz = objects[window, :3]
    horizontal = np.linalg.norm(xyz[:, :2]-objects[0, :2], axis=-1)
    rotations = Rotation.from_quat(objects[:, 3:7]).as_matrix()
    # Tilt measures the body's initial local z direction versus its current
    # direction in world space. Yaw alone does not count as tipping.
    cosine = np.clip(np.sum(rotations[1:, :, 2]*rotations[0, :, 2], axis=-1), -1., 1.)
    tilt = np.degrees(np.arccos(cosine))
    dropped = np.maximum.accumulate(lift)-lift >= THRESHOLDS['lift_m']
    loss_after_lift = bool(np.any(dropped & (np.maximum.accumulate(lift) >= THRESHOLDS['lift_m'])))
    task_complete = bool(trace['complete'][window, index].any()) if 'complete' in trace else None
    task_end = int(row['spec']['source_frames'][-1])
    truncated = controls < task_end and not task_complete
    target = np.asarray(row['spec']['object_path'][-1])[:3, 3]
    arrival_error = float(np.linalg.norm(xyz[-1]-target))

    with np.load(Path(row['source'])/'trajectory.npz', allow_pickle=False) as archive:
        receipt = json.loads(Path(row['collection_batch_manifest']).read_text())
        ids = receipt['observation_contract']['contact_body_ids']
        archive_valid = archive['contact_force_valid'].astype(bool)
        archive_contact = (np.linalg.norm(archive['body_contact_forces'][:, ids], axis=-1) > THRESHOLDS['contact_body_force_n']).sum(-1) >= THRESHOLDS['minimum_contact_bodies']
        archive_contact &= archive_valid
        archive_ticks = archive['frame_tick']
        actual_ticks = trace['tick'][window, index].astype(int) if 'tick' in trace else np.arange(1, controls+1)
        if 'tick' in trace and not np.array_equal(actual_ticks, np.arange(1, controls+1)):
            raise ValueError('native time changed or reset; reference contact comparison invalid')
        reference_first_index = first_true(archive_contact[:controls+1])
        reference_first_tick = int(archive_ticks[reference_first_index]) if reference_first_index is not None else None
        actual_first_index = first_true(multi_contact)
        actual_first_tick = int(actual_ticks[actual_first_index]) if actual_first_index is not None else None
        early = (actual_first_tick <= reference_first_tick-THRESHOLDS['early_contact_lead_controls']) if actual_first_tick is not None and reference_first_tick is not None else None
        contact_agreement = float(np.mean(multi_contact == archive_contact[actual_ticks]))
        reference = archive['hand'][:controls+1]
        hand_error = np.linalg.norm(hand-reference, axis=-1)
        object_error = np.linalg.norm(objects[:, :3]-archive['obj'][:controls+1, :3, 3], axis=-1)

    return dict(sequence=row['sequence'], object=row['object'], evaluated_controls=controls,
        available_completed_controls=completed, reference_prefix_controls=int(row['record']['training_prefix_commands']),
        native_hand_contract='world metres; 11 native bodies, palm index0; no canonical column reorder for these geometric metrics',
        contact_validity_source=validity_source, post_control_contact_valid_fraction=float(valid.mean()),
        proxy_success=bool(longest(hold) >= THRESHOLDS['hold_controls']),
        lift_only_success=bool(longest(lifted) >= THRESHOLDS['hold_controls']),
        max_lift_m=float(lift.max()), final_lift_m=float(lift[-1]),
        longest_hold_controls=longest(hold), longest_hold_seconds=longest(hold)/THRESHOLDS['hz'],
        hold_controls=int(hold.sum()), longest_lift_only_controls=longest(lifted),
        maximum_contact_body_count=int(contact_count[valid].max()) if valid.any() else None,
        multi_finger_contact_controls=int(multi_contact.sum()),
        natural_recovery=natural_recovery(hold),
        task_complete_observed=task_complete, task_endpoint_tick=task_end,
        task_endpoint_truncated=bool(truncated),
        task_completion_for_uncensored_rate=None if truncated else task_complete,
        final_path_progress=float(trace['progress'][controls, index]) if 'progress' in trace else None,
        max_path_progress=float(trace['progress'][:controls+1, index].max()) if 'progress' in trace else None,
        final_endpoint_position_error_m=arrival_error,
        applied_command_rows=controls, applied_command_max_abs=float(np.abs(command).max()),
        applied_command_saturation_fraction=float((np.abs(command) >= .999).mean()),
        reference_hand11_epe_mean_mm=float(hand_error[1:].mean()*1000),
        reference_hand11_epe_last_mm=float(hand_error[-1].mean()*1000),
        reference_object_position_epe_mean_mm=float(object_error[1:].mean()*1000),
        reference_contact_first_tick=reference_first_tick, observed_contact_first_tick=actual_first_tick,
        reference_contact_tick_agreement=contact_agreement,
        maximum_horizontal_drift_m=float(horizontal.max()), maximum_tilt_degrees=float(tilt.max()),
        failure_proxies=dict(no_lift=not bool(lifted.any()),
            failed_sustained_lift=not bool(longest(lifted) >= THRESHOLDS['hold_controls']),
            loss_after_lift=loss_after_lift,
            no_finger_closure_contact=not bool(multi_contact.any()) if valid.all() else None,
            pushed_or_tipped=bool(np.any(supported & ((horizontal > THRESHOLDS['horizontal_push_m']) | (tilt > THRESHOLDS['tilt_degrees'])))),
            arrival_failure=None if truncated else arrival_error > THRESHOLDS['arrival_error_m'],
            early_contact=early, penetration=None),
        limitations=['force bodies provide net-force evidence, not object-specific contact pairs',
                     'loss_after_lift may include intentional release; no causal slip certificate',
                     'full mesh penetration and forced perturbation recovery unmeasured'])


def summarize_arm(root, arm, panel):
    manifest_path = root/arm/'manifest.json'
    trace_path = root/arm/'trajectory.npz'
    if not manifest_path.exists() or not trace_path.exists():
        return dict(status='NOT_AVAILABLE', cases=[], missing=str(trace_path))
    manifest = json.loads(manifest_path.read_text())
    completed = int(manifest.get('controls_completed', 0))
    if completed <= 0:
        return dict(status='NO_COMPLETED_CONTROLS', cases=[], manifest=str(manifest_path), runner_status=manifest.get('status'))
    with np.load(trace_path, allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    if trace['objects'].shape[1] != len(panel['records']): raise ValueError('panel size mismatch')
    if manifest.get('arm') != arm: raise ValueError('arm manifest mismatch')
    native_path = Path(manifest['native_output'])/'manifest.json' if manifest.get('native_output') else None
    native = json.loads(native_path.read_text()) if native_path and native_path.exists() else {}
    if native.get('source_sequence_names') and native['source_sequence_names'] != panel['sequences']:
        raise ValueError('native sequence order differs from panel')
    engineering = dict(manifest=str(native_path) if native_path else None,
                       status=native.get('status'), steps=native.get('steps'),
                       maximum_pd_error=native.get('maximum_pd_error'),
                       episode_action_mismatches=native.get('episode_action_mismatches'),
                       indexed_resets=native.get('indexed_resets'),
                       extended_state=native.get('extended_state'))
    cases = [case_metrics(trace, i, row, completed) for i, row in enumerate(panel['records'])]
    successes = sum(c['proxy_success'] for c in cases)
    completion = [c['task_completion_for_uncensored_rate'] for c in cases if c['task_completion_for_uncensored_rate'] is not None]
    failure_counts = {}
    for name in cases[0]['failure_proxies']:
        values = [c['failure_proxies'][name] for c in cases]
        known = [v for v in values if v is not None]
        failure_counts[name] = dict(observed_true=sum(known), known_cases=len(known),
                                    unknown_cases=len(values)-len(known), rate=sum(known)/len(known) if known else None)
    opportunities = sum(c['natural_recovery']['loss_opportunities'] for c in cases)
    recovered = sum(c['natural_recovery']['recovered_losses'] for c in cases)
    return dict(status=manifest['status'], manifest=str(manifest_path), trace=str(trace_path),
        native_engineering=engineering,
        samples=len(cases), completed_controls_per_environment=completed,
        metric_controls=sum(c['evaluated_controls'] for c in cases),
        proxy_successes=successes, proxy_success_rate=successes/len(cases),
        proxy_success_wilson95=wilson(successes, len(cases)),
        lift_only_successes=sum(c['lift_only_success'] for c in cases),
        lift_only_success_rate=sum(c['lift_only_success'] for c in cases)/len(cases),
        hold_seconds_mean=float(np.mean([c['longest_hold_seconds'] for c in cases])),
        hold_seconds_median=float(np.median([c['longest_hold_seconds'] for c in cases])),
        hold_seconds_max=max(c['longest_hold_seconds'] for c in cases),
        task_complete_observed_count=sum(c['task_complete_observed'] is True for c in cases),
        task_uncensored_count=len(completion), task_censored_count=sum(c['task_endpoint_truncated'] for c in cases),
        task_completion_uncensored_rate=sum(completion)/len(completion) if completion else None,
        path_progress_mean=float(np.mean([c['final_path_progress'] for c in cases])) if all(c['final_path_progress'] is not None for c in cases) else None,
        natural_recovery=dict(opportunities=opportunities, recovered=recovered, rate=recovered/opportunities if opportunities else None,
                              forced_perturbation=None),
        failure_proxies=failure_counts,
        inference_seconds=manifest.get('inference_seconds'), inference_samples=manifest.get('inference_samples'),
        inference_ms_per_sample=manifest['inference_seconds']/manifest['inference_samples']*1000 if manifest.get('inference_samples') else None,
        elapsed_seconds=manifest.get('elapsed_s'), cases=cases)


def paired_difference(a, b):
    by_a = {x['sequence']: x for x in a['cases']}
    by_b = {x['sequence']: x for x in b['cases']}
    names = sorted(set(by_a) & set(by_b))
    if not names: return None
    values = np.asarray([int(by_a[k]['proxy_success'])-int(by_b[k]['proxy_success']) for k in names])
    rng = np.random.default_rng(185)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return dict(samples=len(names), a_minus_b_pp=float(values.mean()*100),
                paired_sequence_bootstrap95_pp=[float(v*100) for v in np.quantile(draws, [.025, .975])],
                a_only_successes=int((values == 1).sum()), b_only_successes=int((values == -1).sum()),
                limitation='one fixed executor/physics seed, selected reference panel; bootstrap reflects sequence sampling only')


def decisions(arms, initials):
    replay, gt = arms['replay'], arms['gt']
    if not replay.get('samples') or not gt.get('samples'):
        return dict(gt_stably_executable=None, prediction_clearly_worse=None, conclusion='PENDING replay and GT traces')
    if any(a['status'] != 'COMPLETED' for a in (replay, gt)):
        return dict(gt_stably_executable=None, prediction_clearly_worse=None,
                    conclusion='Incomplete replay or GT run; partial metrics do not establish executor qualification')
    for name, a in (('replay', replay), ('gt', gt)):
        native = a.get('native_engineering', {})
        if native.get('episode_action_mismatches') not in (0, None) or native.get('indexed_resets') not in (0, None):
            return dict(gt_stably_executable=None, prediction_clearly_worse=None,
                        conclusion=name+' has action/episode or reset mismatches; qualification is confounded')
    paired_initials = initials.get('comparisons', {}).get('gt', {}).get('all_arrays_exact')
    if initials.get('reference_arm') != 'replay' or not paired_initials:
        return dict(gt_stably_executable=None, prediction_clearly_worse=None,
                    conclusion='Initial states are not exactly paired; qualification comparison is confounded')
    r, g = replay['proxy_success_rate'], gt['proxy_success_rate']
    replay_ok = r >= THRESHOLDS['replay_success_gate']
    gt_ok = replay_ok and g >= THRESHOLDS['gt_fraction_of_replay_gate']*r
    contrasts = {arm: paired_difference(gt, arms[arm]) for arm in ('healthy', 'flow')}
    deficits = {arm: (x['a_minus_b_pp'] >= THRESHOLDS['predicted_deficit_pp'] and x['paired_sequence_bootstrap95_pp'][0] > 0)
                if x is not None and arms[arm]['status'] == 'COMPLETED' and
                initials.get('comparisons', {}).get(arm, {}).get('all_arrays_exact') else None
                for arm, x in contrasts.items()}
    if not replay_ok:
        conclusion = 'Robot command replay did not reach the 80% gate; stored reference behavior is not certified reproducible in this host. Do not attribute failure to the generator.'
    elif not gt_ok:
        conclusion = 'The fixed executor did not qualify on executable robot GT tau. The execution bridge is a blocker; generator decoding cannot be identified as the primary bottleneck.'
    elif all(v is True for v in deficits.values()):
        conclusion = 'GT qualifies and both prediction arms have a paired deficit on this selected panel. This supports a prediction-to-execution gap for this bridge, not a universal generator bottleneck.'
    else:
        conclusion = 'GT qualifies; a clear deficit for both prediction arms is not established.'
    return dict(replay_qualified=replay_ok, gt_stably_executable=gt_ok,
                gt_required_success_rate=THRESHOLDS['gt_fraction_of_replay_gate']*r,
                prediction_clearly_worse=deficits, gt_minus_predictions=contrasts,
                conclusion=conclusion)


def markdown(report):
    def fmt(value, digits=3): return 'UNKNOWN' if value is None else f'{value:.{digits}f}'
    lines = ['# Executor qualification summary', '', report['decision']['conclusion'], '',
             'Metrics use post-control states 1..C, C=min(completed controls, admitted prefix, 300). Final unused decisions are excluded.', '',
             '|arm|cases|success|lift-only|hold mean / median / max (s)|task observed / uncensored / censored|',
             '|---|---:|---:|---:|---|---|']
    for arm in ARMS:
        a = report['arms'][arm]
        if not a.get('samples'):
            lines.append(f'|{arm}|0|UNKNOWN|UNKNOWN|UNKNOWN|UNKNOWN|'); continue
        n = a['samples']
        lines.append(f"|{arm}|{n}|{a['proxy_successes']}/{n} ({100*a['proxy_success_rate']:.1f}%)|{a['lift_only_successes']}/{n}|{a['hold_seconds_mean']:.3f} / {a['hold_seconds_median']:.3f} / {a['hold_seconds_max']:.3f}|{a['task_complete_observed_count']} / {a['task_uncensored_count']} / {a['task_censored_count']}|")
    lines += ['', 'Success is 30 consecutive controls with lift>=5cm, no measured table support, palm distance<=20cm, and >=2 finger net-force bodies >0.1N. Force evidence is not object-specific contact-pair proof.', '',
              f"Initial arrays exactly equal across available arms: {report['initial_state_pairing']['all_available_arms_identical']}. See JSON for each array and each case.", '',
              '|arm|natural losses / recoveries|push or tip|loss after lift|no lift|no finger contact|early contact|penetration|',
              '|---|---|---|---|---|---|---|---|']
    for arm in ARMS:
        a = report['arms'][arm]
        if not a.get('samples'): continue
        def cell(key):
            d = a['failure_proxies'][key]
            return 'UNKNOWN' if not d['known_cases'] else f"{d['observed_true']}/{d['known_cases']} (unknown {d['unknown_cases']})"
        recovery = a['natural_recovery']
        cells = [cell(k) for k in ('pushed_or_tipped', 'loss_after_lift', 'no_lift', 'no_finger_closure_contact', 'early_contact', 'penetration')]
        lines.append('|'+ '|'.join([arm, f"{recovery['opportunities']} / {recovery['recovered']}"]+cells)+'|')
    lines += ['', 'Labels overlap and are observable proxies. Natural object lowering may be deliberate release; penetration and forced-disturbance recovery remain UNKNOWN. No natural loss opportunity means recovery rate UNKNOWN, not zero.', '',
              'Early contact compares >=2 force bodies >0.1N against the archived measured robot contacts at the same native ticks; >=3 controls earlier is the flag. Reset step0 is excluded. Missing reference onset gives UNKNOWN.', '',
              'All rates concern a reference-selected 16-case robot DEV panel, one executor/physics seed. Robot TRAIN is sequence/source-disjoint; WAM lifetime exposure is uncertified. This executor was adapted to v4 tau, not certified as v6-adapted.', '',
              'GT minus predicted proxy success (paired sequence bootstrap; not independent training seeds):']
    for arm, contrast in report['decision'].get('gt_minus_predictions', {}).items():
        if contrast:
            lo, hi = contrast['paired_sequence_bootstrap95_pp']
            lines.append(f"- GT − {arm}: {contrast['a_minus_b_pp']:.1f} pp, 95% CI [{lo:.1f}, {hi:.1f}], n={contrast['samples']}.")
    lines += ['', 'Exact thresholds, per-case metrics, source files, failure denominators and censoring are in summary.json.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--panel', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    panel_path = a.panel.resolve() if a.panel else root/'panel.json'
    panel = json.loads(panel_path.read_text())
    initials = compare_initials(root)
    arms = {arm: summarize_arm(root, arm, panel) for arm in ARMS}
    report = dict(schema='ref2dex.wam-v6-executor-summary.v1',
                  experiment_id=panel['experiment_id'], run_id=panel['run_id'],
                  panel=str(panel_path), python=sys.executable, numpy=np.__version__,
                  thresholds=THRESHOLDS, initial_state_pairing=initials,
                  arms=arms, decision=decisions(arms, initials),
                  limits='Observable selected-panel probe; no mesh penetration, forced perturbation, independent executor seed or WAM lifetime-heldout certification.')
    for name, body in [('summary.json', json.dumps(report, indent=2, allow_nan=False)+'\n'),
                       ('summary.md', markdown(report))]:
        part = root/(name+'.part')
        part.write_text(body)
        part.replace(root/name)
    print(json.dumps(dict(decision=report['decision'], available_arms=[k for k, v in arms.items() if v.get('samples')]), indent=2))


if __name__ == '__main__': main()
