"""Merge continuous split runs and generate auditable, abstaining supervision."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))

import numpy as np

from consequence_evaluator.contracts import (K, K_EXEC, is_within, MIN_PREFERENCE_PAIRS,
                                             HISTORY_MATCH_MAX_RELATIVE_RMS)
from consequence_evaluator.data import sha, validate_rigid
from consequence_evaluator.supervision import (
    CONTACT_SEMANTICS, RULE, expert_anchor, local_event, local_preferences, pair_coverage_ready,
    physical_trace,
)


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def label(sources, output, route_config, extra_preferences=None, seconds=600, *, audit_only=False):
    """Preserve raw H/A/poses and split groups; only supervision is replaced."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= seconds <= 600:
        raise ValueError('bounded label preparation: <=600 seconds')
    started = time.monotonic()
    route = json.loads(Path(route_config).read_text())
    qualified = route.get('all_experts_operationally_qualified') is True
    route_trainable = route.get('training_allowed') is True and qualified
    if not audit_only and not route_trainable:
        raise ValueError('expert route is observational-only; labeling for evaluator training is blocked')
    expected_experts = route['experts']
    pair_coverage_required = route.get('pair_coverage_required', True)
    route_digest = sha(Path(route_config))
    rule_digest = sha(TASK / 'src/consequence_evaluator/supervision.py')
    contracts_digest = sha(TASK / 'src/consequence_evaluator/contracts.py')
    producer_digest = sha(Path(__file__))
    expected_annotation_provenance = dict(
        rule=RULE, contact_semantics=CONTACT_SEMANTICS,
        rule_sha256=rule_digest, contracts_sha256=contracts_digest,
        history_match_relative_rms=HISTORY_MATCH_MAX_RELATIVE_RMS,
        minimum_pair_coverage=MIN_PREFERENCE_PAIRS, route_sha256=route_digest,
        all_experts_operationally_qualified=qualified)
    if len(expected_experts) != 6 or len({v['sha256'] for v in expected_experts.values()})!=6:
        raise ValueError('fixed six self-trained experts required')
    expected_hashes = {item['sha256'] for item in expected_experts.values()}
    output.mkdir(parents=True)
    report = dict(status='RUNNING', rule=RULE, contact_semantics=CONTACT_SEMANTICS,
                  rule_sha256=rule_digest,
                  contracts_sha256=contracts_digest,
                  producer_sha256=producer_digest,
                  route_sha256=route_digest, route_training_allowed=route_trainable,
                  audit_only=bool(audit_only),
                  source_manifests=[], episode_anchors={},
                  event_counts={}, abstention_counts={}, sources_cpu_reason='file/label statistics only; no model computation')
    from consequence_evaluator.contracts import EPISODE_SCHEMA,ACTION_SEMANTICS
    manifest = dict(schema=EPISODE_SCHEMA,action_semantics=ACTION_SEMANTICS, status='LABELING',
                    rollout_kind='continuous', fps=30, units='m', training_allowed=False,
                    contact_semantics=CONTACT_SEMANTICS, episodes=[], label_rule=RULE,
                    audit_only=bool(audit_only))
    records, windows, groups, episode_ids, phases = {}, [], {}, set(), {}
    try:
        for directory in map(Path, sources):
            source_manifest = directory / 'manifest.json'
            m = json.loads(source_manifest.read_text())
            # Older raw runs pinned the route in sources rather than a named
            # route_sha256 field. Accept that identity only for read-only audit.
            source_route_hash = m.get('route_sha256')
            if audit_only and source_route_hash is None:
                source_route_hash = m.get('sources', {}).get(str(Path(route_config).resolve()))
            if (m.get('schema') != manifest['schema'] or m.get('status') != 'COMPLETED'
                    or m.get('rollout_kind') != 'continuous'
                    or (not audit_only and (m.get('training_allowed') is not True
                                            or m.get('all_experts_operationally_qualified') is not True))
                    or source_route_hash != route_digest
                    or m.get('action_semantics')!=ACTION_SEMANTICS
                    or m.get('fps') != 30 or m.get('units') != 'm' or not m.get('history_contract')
                    or m.get('contact_semantics') != CONTACT_SEMANTICS
                    or not expected_hashes.issubset(set(m.get('sources', {}).values()))):
                raise ValueError('completed native six-expert collection provenance required')
            if 'history_contract' in manifest and manifest['history_contract'] != m['history_contract']:
                raise ValueError('raw policy H contract differs across source runs')
            manifest['history_contract'] = m['history_contract']
            report['source_manifests'].append(dict(path=str(source_manifest.resolve()), sha256=sha(source_manifest)))
            for record in m['episodes']:
                if time.monotonic() - started >= seconds:
                    raise TimeoutError('fixed label-preparation deadline')
                if len(records) >= 256:
                    raise ValueError('at most256 source episodes in this bounded probe')
                episode = record['episode']
                if episode in episode_ids or record['quality'] != 'unlabeled' or record['expert'] not in expected_experts:
                    raise ValueError('duplicate episode or previously labeled/non-native input')
                episode_ids.add(episode)
                split, group = record['split'], record['split_group']
                if split not in ('train', 'val', 'test') or (group in groups and groups[group] != split):
                    raise ValueError('source seed group crosses splits')
                groups[group] = split
                path = (directory / record['path']).resolve()
                sidecar = (directory / record['diagnostics']).resolve()
                if sha(path) != record['sha256'] or sha(sidecar) != record['diagnostics_sha256']:
                    raise ValueError('raw episode or physical diagnostic identity changed')
                with np.load(path, allow_pickle=False) as packet:
                    expected = {'history', 'action','residual_plan','plan_known','hand_keypoints', 'object_pose', 'timestamps', 'phase', 'progress', 'progress_mask'}
                    if set(packet.files) != expected:
                        raise ValueError('raw episode whitelist mismatch')
                    arrays = {key: packet[key].copy() for key in packet.files}
                with np.load(sidecar, allow_pickle=False) as packet:
                    diagnostics = {key: packet[key].copy() for key in packet.files}
                steps = len(arrays['action'])
                if (not K <= steps <= 1200 or arrays['action'].shape != (steps, 18)
                        or len(arrays['history']) != steps+1 or arrays['phase'].shape != (steps,)
                        or arrays['timestamps'].shape != (steps+1,)
                        or not np.allclose(np.diff(arrays['timestamps']), 1/30, atol=1e-6, rtol=0)
                        or arrays['progress_mask'].shape != (steps+1,) or arrays['progress_mask'].any()
                        or not np.isfinite(arrays['history']).all() or not np.isfinite(arrays['action']).all()
                        or np.abs(arrays['action']).max() > 1+1e-6):
                    raise ValueError('invalid raw continuous observation/action/clock or preexisting dense labels')
                validate_rigid(arrays['object_pose'])
                if (arrays['residual_plan'].shape!=(steps,K,18) or arrays['plan_known'].shape!=(steps,)
                        or arrays['plan_known'].dtype!=np.bool_
                        or not np.isfinite(arrays['residual_plan']).all()
                        or np.abs(arrays['residual_plan']).max()>.2+1e-6
                        or arrays['hand_keypoints'].shape!=(steps+1,11,3)
                        or not np.isfinite(arrays['hand_keypoints']).all()):
                    raise ValueError('decision-known residual plan fields required')
                trace = physical_trace(arrays, diagnostics)
                audit=report.setdefault('force_proxy_geometry_audit',dict(valid_frames=0,proxy_frames=0,
                    proxy_and_near_frames=0,proxy_far_frames=0,elevated_proxy_frames=0,elevated_proxy_near_frames=0))
                valid=trace['valid'];force=trace['force_proxy'];near=trace['surface_gap']<=.01
                audit['valid_frames']+=int(valid.sum());audit['proxy_frames']+=int((valid&force).sum())
                audit['proxy_and_near_frames']+=int((valid&force&near).sum())
                audit['proxy_far_frames']+=int((valid&force&(trace['surface_gap']>.03)).sum())
                audit['elevated_proxy_frames']+=int((valid&force&(trace['height']>=.03)).sum())
                audit['elevated_proxy_near_frames']+=int((valid&force&near&(trace['height']>=.03)).sum())
                quality, progress, mask, completion = expert_anchor(trace, record)
                arrays['progress'], arrays['progress_mask'] = progress, mask
                # 8-step sample cadence; K stays24. Every chosen sample is a full window.
                ticks = [t for t in range(0, steps-K+1, K_EXEC) if arrays['plan_known'][t]]
                trigger=record['perturbation_tick']
                if 0<=trigger<=steps-K and arrays['plan_known'][trigger]:
                    ticks=sorted(set(ticks)|{trigger})
                for tick in ticks:
                    if len(windows) >= 20000:
                        raise ValueError('at most20000 selected windows; explicitly subset source episodes')
                    phase = str(arrays['phase'][tick])
                    event = local_event(trace, phase, tick)
                    window = dict(episode=episode, tick=tick, split=split, task=record['task'], phase=phase,
                                  expert=record['expert'],motion=record['motion'], history=arrays['history'][tick],
                                  object_pose=arrays['object_pose'][tick],hand_keypoints=arrays['hand_keypoints'][tick],
                                  initial_relative_height=float(trace['height'][tick]), event=event)
                    windows.append(window)
                    counts = report['event_counts'] if event else report['abstention_counts']
                    key = split + '/' + record['task'] + '/' + phase + '/' + str(event)
                    counts[key] = counts.get(key, 0) + 1
                # Avoid using external episode identity strings as filesystem paths.
                target = output / ('episode-' + str(len(records)) + '.npz')
                np.savez_compressed(target, **arrays)
                new_record = dict(record, quality=quality, path=target.name, sha256=sha(target), steps=steps,
                                  diagnostics=str(sidecar), window_ticks=ticks,
                                  raw_episode_path=str(path), raw_episode_sha256=record['sha256'])
                records[episode] = new_record
                phases[episode] = arrays['phase'].copy()
                manifest['episodes'].append(new_record)
                report['episode_anchors'][episode] = dict(quality=quality, verified_completion_tick=completion,
                    progress_frames=int(mask.sum()), criterion='45 elevated force-proxy+<=1cm surface-gap frames; no later drop',
                    raw_episode_sha256=record['sha256'], diagnostics_sha256=record['diagnostics_sha256'])
                if sum(p.stat().st_size for p in output.glob('*') if p.is_file()) > 2*2**30:
                    raise ValueError('label output exceeded2GiB')
        pairs = local_preferences(windows)
        if extra_preferences:
            extra = json.loads(Path(extra_preferences).read_text())
            extra_provenance = extra.get('label_provenance') or {}
            if (extra.get('scope') != 'local_window'
                    or any(extra_provenance.get(key) != value
                           for key, value in expected_annotation_provenance.items())):
                raise ValueError('additional local preference provenance required')
            report['extra_preferences'] = dict(path=str(Path(extra_preferences).resolve()),
                                               sha256=sha(extra_preferences), provenance=extra['label_provenance'])
            # Explicit reviewed windows may lie between the regular sampling ticks.
            for pair in extra['pairs']:
                for name in ('chosen', 'rejected'):
                    endpoint = pair[name]
                    if endpoint['episode'] not in records:
                        raise ValueError('additional preference refers to an unknown episode')
                    record = records[endpoint['episode']]
                    tick = endpoint['tick']
                    steps = record['steps']
                    if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick <= steps-K:
                        raise ValueError('additional preference needs a full24-step window')
                    record['window_ticks'] = sorted(set(record['window_ticks']) | {tick})
                pairs.append(pair)
        lookup = {episode: record for episode, record in records.items()}
        split_pairs = dict(train=0, val=0, test=0)
        pair_groups = {split: set() for split in ('train', 'val', 'test')}
        seen = set()
        unique = []
        for pair in pairs:
            chosen, rejected = pair['chosen'], pair['rejected']
            left, right = lookup[chosen['episode']], lookup[rejected['episode']]
            if (left['episode'] == right['episode'] or left['split'] != right['split'] or left['task'] != right['task']
                    or left['expert']!=right['expert'] or left['motion']!=right['motion']
                    or phases[left['episode']][chosen['tick']] != phases[right['episode']][rejected['tick']]
                    or not str(pair.get('annotation', '')).strip()):
                raise ValueError('invalid cross-episode local preference')
            from consequence_evaluator.supervision import states_match
            current=[]
            for record,endpoint in [(left,chosen),(right,rejected)]:
                with np.load(output/record['path'],allow_pickle=False) as packet:
                    tick=endpoint['tick']
                    if not packet['plan_known'][tick]:raise ValueError('unknown residual schedule in local preference')
                    current.append(dict(history=packet['history'][tick],
                                        object_pose=packet['object_pose'][tick],
                                        hand_keypoints=packet['hand_keypoints'][tick]))
            if not states_match(*current, require_history=True):
                raise ValueError('local preference current H/object/hand states differ')
            identity = ((chosen['episode'], chosen['tick']), (rejected['episode'], rejected['tick']))
            if identity[::-1] in seen:
                raise ValueError('contradictory local preference directions')
            if identity in seen:
                continue
            seen.add(identity)
            split = left['split']
            split_pairs[split] += 1
            pair_groups[split].add(tuple(sorted((left['episode'], right['episode']))))
            unique.append(pair)
        selected_windows = sum(len(r['window_ticks']) for r in records.values())
        if selected_windows > 20000:
            raise ValueError('additional annotations exceeded the20000-window budget')
        train_anchors = sum(report['episode_anchors'][r['episode']]['progress_frames']
                            for r in records.values() if r['split'] == 'train')
        unique_pair_counts = {split: len(groups) for split, groups in pair_groups.items()}
        ready = pair_coverage_ready(unique_pair_counts) if pair_coverage_required else bool(all(split_pairs.values()))
        report.update(status='READY' if ready else 'INSUFFICIENT_PREFERENCES',
                      pairs=split_pairs, episodes=len(records), selected_windows=selected_windows,
                      train_progress_anchor_frames=train_anchors,
                      unique_episode_pair_groups=unique_pair_counts,
                      minimum_pair_coverage=MIN_PREFERENCE_PAIRS,
                      pair_coverage_required=pair_coverage_required,
                      limitation='automatic preferences cover unambiguous held/drop and grasp/lift events only; '
                                 'approach, miss and recovery comparisons need separately grounded local annotations')
        if not train_anchors:
            report['status'] = 'INSUFFICIENT_EXPERT_PROGRESS'
        for source in report['source_manifests']:
            if sha(source['path']) != source['sha256']:
                raise RuntimeError('collection source manifest changed during labeling')
        for record in records.values():
            if (sha(record['raw_episode_path']) != record['raw_episode_sha256']
                    or sha(record['diagnostics']) != record['diagnostics_sha256']):
                raise RuntimeError('raw episode or diagnostic drift during labeling')
        if (sha(Path(route_config)) != route_digest
                or sha(TASK / 'src/consequence_evaluator/supervision.py') != rule_digest
                or sha(TASK / 'src/consequence_evaluator/contracts.py') != contracts_digest
                or sha(Path(__file__)) != producer_digest):
            raise RuntimeError('labeling route or implementation inputs changed during labeling')
        report['training_allowed'] = not audit_only and route_trainable and report['status'] == 'READY'
        manifest.update(status='COMPLETED', training_allowed=report['training_allowed'],
                        minimum_pair_coverage=MIN_PREFERENCE_PAIRS,
                        pair_coverage_required=pair_coverage_required,
                        route_sha256=route_digest,
                        all_experts_operationally_qualified=qualified,
                        label_report_sha256=None)
        report['elapsed_s'] = time.monotonic() - started
        write(output / 'label_report.json', report)
        manifest['label_report_sha256'] = sha(output / 'label_report.json')
        provenance = dict(expected_annotation_provenance,
                          report_sha256=manifest['label_report_sha256'])
        write(output / 'preferences.json', dict(scope='local_window',
              label_provenance=provenance, pairs=unique))
        write(output / 'manifest.json', manifest)
    except BaseException as error:
        report.update(status='TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILED', error=repr(error),
                      elapsed_s=time.monotonic()-started)
        manifest['status'] = report['status']
        write(output / 'label_report.json', report)
        write(output / 'manifest.json', manifest)
        raise
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--route-config', type=Path, default=ROOT/'src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json')
    p.add_argument('--extra-preferences', type=Path)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--audit-only', action='store_true',
                   help='audit existing continuous raw episodes; output is never trainable')
    a = p.parse_args()
    if not is_within(a.output, ROOT/'outputs/consequence-evaluator'):
        p.error('output must be under outputs/consequence-evaluator')
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('free disk below20GiB preparation reserve')
    print(json.dumps(label(a.sources, a.output, a.route_config, a.extra_preferences, a.seconds,
                          audit_only=a.audit_only), indent=2))


if __name__ == '__main__':
    main()
