#!/usr/bin/env python3
"""Bounded owner-requested data gates. Never creates an optimizer or trains.

Stages have separate immutable receipts so an interrupted expensive stage does
not erase completed evidence. Full means all ordered tuples/shards and file
stats; it does not mean materializing eleven million overlapping point clouds.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit'), str(ROOT/'src/task/wam-rl-policy-v3/src')]
from ptv3_shared_wam_v6.prepared_reader import PreparedWindows, collate, read, sha256
from ptv3_shared_wam_v5.cache_contract import array_identity, metadata_identity, identity
from ptv3_shared_wam_v5.event_labels import annotate_events, proximity_counts, OBSERVATION_STRIDE_BY_SOURCE
from ptv3_shared_wam_v6.goal_policy import UNKNOWN_GOAL_POLICY
from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY

RUN = ROOT/'outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1'
MANIFEST = RUN/'readiness-manifest-r1/manifest.json'
REGISTRY = ROOT/'outputs/ptv3-shared-wam-v5/object-surfaces-20261019-r2/manifest.json'
FRESH = ROOT/'outputs/ptv3-shared-wam-v6/rigid-source-bound-pilot-20261023-r2'
PINNED = '4e40b4ec0bd58d390f8d5cee6047530d1832c3d5acb6080232251236d934c4c3'
SEED = 20261023
START_GIT_COMMIT = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
START_CODE_SHA256 = {str(p.relative_to(ROOT)):sha256(p) for p in
    [Path(__file__).resolve(),TASK/'src/ptv3_shared_wam_v6/prepared_reader.py']}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def progress(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def datasets():
    return {s: PreparedWindows(MANIFEST, s, REGISTRY, PINNED) for s in ('train', 'dev')}


def full(out):
    ds = datasets()
    report = dict(reader='PreparedWindows (data-only; production recipe still blocked)',
                  materialized_all_point_clouds=False, by_source={}, split_leaks=[], files=0,
                  fixed_conversion_measured_bytes=0, fixed_conversion_receipt_bytes=0,
                  fixed_conversion_expected_bytes=0, tuple_rows_read=Counter(), sequences_read=Counter(),
                  metadata_checked=0, shard_hashes_checked=0, coverage=Counter(),
                  certified_goal_windows=Counter(), binding_errors=[])
    seen = {kind: {} for kind in ('mother', 'hand_path', 'native_uri', 'hand_payload')}
    # Independent from summary: consume each actual shard, compare every tuple
    # to the reader mmap in order, check all file sizes and metadata bindings.
    rows = ds['train'].rows
    file_ledger = (out/'all-fixed-file-stats.jsonl').open('x')
    seq_ledger = (out/'all-sequence-reconciliation.jsonl').open('x')
    for number, (sid, row) in enumerate(sorted(rows.items())):
        split, source = row['split'], row['source_id']
        dataset = ds[split]
        receipt = read(RUN/'receipts'/(row['key']+'.json'))
        assert receipt['sequence_id'] == sid and receipt['window_shard_sha256'] == row['window_shard_sha256']
        assert sha256(row['window_shard']) == row['window_shard_sha256']
        report['shard_hashes_checked'] += 1
        with np.load(row['window_shard'], allow_pickle=False) as z:
            tuples, phase, geom = z['tuples'], z['phase'], z['geometry_class']
        offset, count = row['normalized_offset'], len(tuples)
        assert count == receipt['admitted_windows'] == row['admitted_windows']
        assert offset == report['tuple_rows_read'][split]
        assert np.array_equal(dataset.index[offset:offset+count], tuples)
        assert tuples.dtype == np.int32 and (tuples[:, 0] == sid).all()
        assert len(np.unique(tuples, axis=0)) == count
        assert (tuples[:, 1] >= 0).all() and (tuples[:, 1] < len(row['objects'])).all()
        measured = 0
        metadata = {}
        for field, expected in [('path', 'hand_meta_identity'), ('contact_path', 'contact_meta_identity'),
                                ('event_path', 'event_meta_identity')]:
            folder = Path(row[field])
            assert metadata_identity(folder/'meta.json') == row[expected]
            report['metadata_checked'] += 1
            metadata[field] = read(folder/'meta.json')
            files = sorted(f for f in folder.iterdir() if f.is_file())
            for f in files:
                size = f.stat().st_size
                if row['conversion_started']:
                    measured += size
                    report['files'] += 1
                    file_ledger.write(json.dumps(dict(sequence_id=sid, path=str(f), bytes=size))+'\n')
        hm, cm, em = (metadata[f] for f in ('path', 'contact_path', 'event_path'))
        assert cm['hand_identity'] == row['hand_meta_identity']
        assert em['contact_meta_identity'] == row['contact_meta_identity']
        assert (tuples[:, 2] >= 3).all() and (tuples[:, 2] < hm['frames']-24).all()
        for slot in np.unique(tuples[:, 1]):
            ticks = tuples[tuples[:, 1] == slot, 2]
            assert (np.diff(ticks) > 0).all()
        assert measured == receipt['conversion_actual_bytes'] == receipt['conversion_expected_bytes']
        report['fixed_conversion_measured_bytes'] += measured
        report['fixed_conversion_receipt_bytes'] += receipt['conversion_actual_bytes']
        report['fixed_conversion_expected_bytes'] += receipt['conversion_expected_bytes']
        for kind, key in [('mother', (source, row['source_sequence'])),
                          ('hand_path', str(Path(row['path']).resolve())),
                          ('native_uri', (source, row.get('native_evidence', {}).get('uri'))),
                          ('hand_payload', hm.get('payload_identity'))]:
            if key is None or (kind == 'native_uri' and key[1] is None):
                continue
            previous = seen[kind].setdefault(key, (split, row['key']))
            if previous[0] != split:
                report['split_leaks'].append(dict(kind=kind, previous=previous, current=[split, row['key']]))
        report['tuple_rows_read'][split] += count
        report['sequences_read'][split] += 1
        target = report['by_source'].setdefault(str(source), Counter())
        target[split] += count
        if row.get('native_evidence', {}).get('complete_certificate_fields'):
            report['binding_errors'].append(row['key'] + ': certificate candidate requires adjudication')
        report['certified_goal_windows'][split] += 0
        values, counts = np.unique(np.stack([phase, geom], 1), axis=0, return_counts=True)
        for (p, g), n in zip(values, counts):
            report['coverage']['%s/%s/%s/%s' % (split, source, int(p), int(g))] += int(n)
        seq_ledger.write(json.dumps(dict(key=row['key'], sequence_id=sid, split=split, source_id=source,
            manifest_windows=count, reader_rows=count, offset=offset, measured_bytes=measured,
            receipt_bytes=receipt['conversion_actual_bytes']))+'\n')
        if number % 2048 == 0:
            progress(stage='full', mothers=number, rows=dict(report['tuple_rows_read']))
    file_ledger.close(); seq_ledger.close()
    assert report['fixed_conversion_measured_bytes'] == 182331193978
    for split, dataset in ds.items():
        assert report['tuple_rows_read'][split] == len(dataset)
        for source in dataset.manifest['sources']:
            assert report['by_source'][str(source['source_id'])][split] == source[split]['admitted_windows']
    report['zero_train_cells'] = [dict(source_id=s, phase=['approach','close','contact','hold'][p],
         geometry=['flat','elongated','compact'][g]) for s in sorted(map(int, report['by_source']))
         for p in range(4) for g in range(3) if not report['coverage']['train/%s/%s/%s' % (s, p, g)]]
    report['manifest_diff_from_brief47'] = dict(same_manifest_path=True, same_manifest_sha256=sha256(MANIFEST),
        added_sequences=0, removed_sequences=0, added_windows=0, removed_windows=0,
        fresh_partition_included=False, ordered_tuples_compared='all shard rows versus full reader arrays')
    report['status'] = 'PASS' if not report['split_leaks'] and not report['binding_errors'] else 'FAIL'
    return report


def selected_rows(ds):
    rng = np.random.default_rng(SEED)
    result = []
    sources = sorted(s['source_id'] for s in ds['train'].manifest['sources'])
    for source in sources:
        splits = ['train', 'dev'] if source != 8 else ['train']
        for split in splits:
            dataset = ds[split]
            mothers = [r for r in dataset.rows.values() if r['source_id'] == source and r['split'] == split]
            # Uniform mothers then uniform admitted rows, to avoid source0 and
            # long trajectories dominating the independent content sample.
            chosen = rng.choice(len(mothers), 32//len(splits), replace=False)
            for ordinal in chosen:
                row = mothers[int(ordinal)]
                index = row['normalized_offset'] + int(rng.integers(row['admitted_windows']))
                result.append((split, index))
    assert len(result) == 288 and len(set(result)) == 288
    return result


def check_sample(dataset, index):
    from scipy.spatial.transform import Rotation
    from ptv3_shared_wam_v6.rigid_dataset import admission_reasons
    from ptv3_shared_wam_v5.contracts import JOINT_NAMES, PART_NAMES, LABEL_SEMANTICS
    sid, slot, tick = map(int, dataset.index[index]); row = dataset.rows[sid]
    seq, contact, event = dataset.sequence(sid); geo = dataset.geometry_for(sid, slot)
    sample = dataset[index]
    for key, value in sample.items():
        if np.asarray(value).dtype.kind in 'fc':
            assert np.isfinite(value).all(), key
    evidence = dict(joint_semantics=list(JOINT_NAMES), part_semantics=list(PART_NAMES),
        label_semantics=LABEL_SEMANTICS, geometry_binding_verified=True,
        binding_verified=True, event_binding_verified=True)
    assert not admission_reasons(seq, geo, contact, event, evidence, tick, slot)
    # Independent scipy SE3 evaluation from recorded poses (not sample_window).
    poses = np.asarray(seq['object_poses'][tick-3:tick+25, slot], np.float64)
    rotation = Rotation.from_matrix(poses[3, :3, :3])
    inv_r = rotation.inv().as_matrix(); origin = poses[3, :3, 3]
    hands = (seq['joints'][tick-3:tick+25].astype(np.float64)-origin) @ inv_r.T
    hv = seq['joint_valid'][tick-3:tick+1].all(0)
    valid = seq['joint_valid'][tick+1:tick+25] & hv[None]
    target = np.where(valid[..., None], hands[4:]-hands[3], 0)
    assert np.array_equal(sample['trajectory_valid'], valid)
    assert np.allclose(sample['trajectory_target'], target, atol=2e-5, rtol=1e-5)
    q = geo['query'].astype(np.float64)
    world_q = q[None] @ np.swapaxes(poses[4:, :3, :3], -1, -2) + poses[4:, None, :3, 3]
    local_q = (world_q-origin) @ inv_r.T
    assert np.allclose(sample['scene_target'][0], local_q-q, atol=2e-5, rtol=1e-5)
    assert np.array_equal(sample['contact_target'][0], contact['contact'][tick+1:tick+25, slot].astype('f4'))
    assert np.array_equal(sample['part_target'][0], contact['part'][tick+1:tick+25, slot].astype('i8'))
    assert sample['trajectory_target'].shape == (24, 2, 21, 3)
    assert sample['contact_target'].shape == (1, 24, 2048)
    assert sample['trajectory_target'].dtype == sample['contact_target'].dtype == np.float32
    assert sample['trajectory_valid'].dtype == sample['object_goal_valid'].dtype == bool
    assert sample['part_target'].dtype == np.int64
    assert all(not np.count_nonzero(sample[k]) for k in ('goal_object_delta','goal_object_points',
        'object_goal_valid','event_type','event_delta_steps','event_valid'))
    ne = row.get('native_evidence', {})
    clock_scope = 'relative native frame grid; no absolute timestamp claimed'
    if row['source_id'] not in (0, 1, 2):
        with np.load(ne['uri'], allow_pickle=False) as z:
            clock = z['time'][tick-3:tick+25]
            assert clock.shape == (28,) and np.isfinite(clock).all()
            assert (np.diff(clock) > 0).all() and np.allclose(np.diff(clock), 1/30, rtol=0, atol=1e-5)
            if 'segment' in z:
                assert (np.diff(z['segment'][tick-3:tick+25]) == 0).all()
            clock_scope = 'native time array: strictly increasing at 30Hz'
    else:
        assert seq['observed_frame'].shape == (len(seq['joints']),)
        clock = np.arange(tick-3, tick+25, dtype=np.float64)/30
        assert (np.diff(clock) > 0).all()
    raw_goal = local_q[-1]-q
    return sample, dict(split=dataset.split, index=index, source_id=row['source_id'],
        sequence_key=row['key'], tuple=[sid, slot, tick], clock_scope=clock_scope,
        native_absolute_timestamp_available=False if row['source_id'] in (0,1,2) else None,
        raw_goal_label_nonzero_fraction=float(np.count_nonzero(raw_goal)/raw_goal.size),
        raw_event_type=int(event['event_type'][tick,slot]), raw_event_valid=bool(event['event_valid'][tick,slot]),
        raw_event_delta_steps=int(event['event_delta_steps'][tick,slot]),
        fields={k:dict(shape=list(np.shape(v)), dtype=str(np.asarray(v).dtype),
                        nonzero=int(np.count_nonzero(v)), elements=int(np.size(v))) for k,v in sample.items()})


def sample(out):
    ds = datasets(); selected = selected_rows(ds); stats = defaultdict(Counter); details = []
    for split, index in selected:
        value, detail = check_sample(ds[split], index)
        details.append(detail)
        for key, data in detail['fields'].items():
            stats[key]['nonzero'] += data['nonzero']; stats[key]['elements'] += data['elements']
        if len(details) % 32 == 0:
            progress(stage='sample', checked=len(details))
    write(out/'sample-windows.json', details)
    return dict(status='PASS', windows=288, seed=SEED, sampling='32/source; 16/split except source8 TRAIN32; uniform mother then admitted row',
        by_source=dict(Counter(str(r['source_id']) for r in details)),
        by_split=dict(Counter(r['split'] for r in details)),
        raw_event_categories=dict(Counter('%s/%s' % (r['raw_event_type'],r['raw_event_valid']) for r in details)),
        field_nonzero_rates={k:dict(v, rate=v['nonzero']/v['elements']) for k,v in stats.items()},
        independent_checks=['SciPy pose transform vs hand/scene labels', 'exact recorded contact/part values',
            'raw event bounds', 'H4/K24 admission and numerical SE3', 'native time/segment where exported',
            'canonical unknown input masks', 'dtype/shape/finiteness'],
        absolute_timestamp_limitation='Sources0/1/2 export frame-rate contracts, not absolute timestamp arrays; not fabricated.')


def hashes(out):
    ds = datasets(); rng = np.random.default_rng(SEED); rows = list(ds['train'].rows.values())
    selected = []
    for source in sorted({r['source_id'] for r in rows}):
        pool = [r for r in rows if r['source_id'] == source and r['conversion_started']]
        selected.extend(pool[int(i)] for i in rng.choice(len(pool), math.ceil(len(pool)*.015), replace=False))
    report = dict(status='PASS', seed=SEED, sampled_sequences=len(selected), files_hashed=0,
        bytes_hashed=0, payload_bundles_compared=0, events_recomputed=0, missing_prior_payload_identity=[],
        fraction_scope='all 22 fixed files of 1.5% newly converted sequences, stratified by source; excludes read-only reused roots')
    with (out/'sample-file-sha256.jsonl').open('x') as ledger:
        for number, row in enumerate(selected):
            sid = row['sequence_id']; seq, contact, event = ds[row['split']].sequence(sid)
            for field, values in [('path',seq),('contact_path',contact)]:
                meta = read(Path(row[field])/'meta.json')
                expected = meta.get('payload_identity')
                if expected is None:
                    report['missing_prior_payload_identity'].append(str(Path(row[field])/'meta.json'))
                else:
                    assert array_identity(values) == expected, row['key']
                    report['payload_bundles_compared'] += 1
            enter, leave = proximity_counts(contact['contact'],contact['valid'])
            regenerated = annotate_events(enter,leave,contact['valid'],
                observation_stride=OBSERVATION_STRIDE_BY_SOURCE.get(row['source_id'],1))
            regenerated.pop('transitions'); regenerated.update(enter_count=enter,leave_count=leave,raw_valid=np.asarray(contact['valid']))
            assert set(regenerated) == set(event)
            for name, value in regenerated.items():
                assert np.array_equal(value, event[name]), (row['key'],name)
                stream = io.BytesIO(); np.save(stream,value)
                assert hashlib.sha256(stream.getvalue()).hexdigest() == sha256(Path(row['event_path'])/(name+'.npy'))
            report['events_recomputed'] += 1
            for field in ('path','contact_path','event_path'):
                for f in sorted(Path(row[field]).iterdir()):
                    if f.is_file():
                        digest = sha256(f)
                        assert digest == sha256(f), 'input mutated during sample hashing'
                        ledger.write(json.dumps(dict(path=str(f),sha256=digest,bytes=f.stat().st_size,
                            reference='recorded canonical metadata/payload identity; events independently regenerated'))+'\n')
                        report['files_hashed'] += 1; report['bytes_hashed'] += f.stat().st_size
            if number % 64 == 0:
                progress(stage='hash', sequences=number, files=report['files_hashed'])
    report['total_new_fixed_files'] = 38350*22
    report['sampled_file_fraction'] = report['files_hashed']/report['total_new_fixed_files']
    assert .01 <= report['sampled_file_fraction'] <= .02
    assert not report['missing_prior_payload_identity']
    return report


def geometry(out):
    """Independent brute-force nearest-vertex check, without the writer KD tree."""
    from scipy.spatial.distance import cdist
    ds = datasets(); checked = invalid = 0; maximum = 0.; ties = 0
    for split, index in selected_rows(ds):
        dataset = ds[split]; sid, slot, tick = map(int,dataset.index[index])
        seq, contact, _ = dataset.sequence(sid); geo = dataset.geometry_for(sid,slot)
        for frame in (tick,tick+12,tick+24):
            sides = seq['surface_valid'][frame]
            expected = seq['expected_hand_presence'][frame]
            valid = bool((~expected|sides).all() and sides.any() and seq['observed_frame'][frame]
                         and seq['object_pose_valid'][frame,slot])
            assert valid == bool(contact['valid'][frame,slot])
            if not valid:
                invalid += 1
                assert not np.count_nonzero(contact['contact'][frame,slot])
                continue
            vertices = (seq['surface_offsets'][frame].astype('f4')+seq['joints'][frame,:,None,0])[sides].reshape(-1,3)
            labels = seq['surface_parts'][sides].reshape(-1)
            pose = seq['object_poses'][frame,slot]
            queries = geo['surface']@pose[:3,:3].T+pose[:3,3]
            near = ((queries >= vertices.min(0)-.033)&(queries <= vertices.max(0)+.033)).all(-1)
            distances = cdist(queries[near],vertices)
            which = distances.argmin(-1); distance = distances[np.arange(len(distances)),which]
            expected_c = np.zeros(2048,np.float16); expected_p = np.zeros(2048,np.uint8)
            confidence = np.exp(-np.maximum(distance-.003,0)/.010); confidence[distance>.033] = 0
            expected_c[near] = confidence.astype('f2'); expected_p[near] = labels[which]
            saved_c, saved_p = contact['contact'][frame,slot],contact['part'][frame,slot]
            maximum = max(maximum,float(np.abs(expected_c.astype('f4')-saved_c.astype('f4')).max()))
            assert np.array_equal(expected_c,saved_c), (sid,frame,'contact')
            assert not np.any(saved_p[~near]), (sid,frame,'outside bounding box')
            different = np.flatnonzero(expected_p[near]!=saved_p[near])
            for j in different:
                # Different solvers may choose different exactly tied vertices;
                # accept only an actual equal-distance vertex with saved part.
                matching = labels == saved_p[near][j]
                assert matching.any() and np.min(distances[j,matching]) <= distance[j]+1e-12
                ties += 1
            checked += 1
    return dict(status='PASS',windows=288,frames_checked=checked,invalid_frames_mask_checked=invalid,
        maximum_contact_error=maximum,exact_distance_part_ties=ties,
        algorithm='SciPy brute-force cdist; independent of writer surface_contact/cKDTree',
        scope='saved pose + saved surface -> saved contact/part; does not independently establish raw MANO joint convention')


def goal(out):
    from wam_rl_policy_v3.goal_channel import validate_instruction
    from ptv3_shared_wam_v6.fresh_rigid import reset_bound_sample
    candidate = read(FRESH/'candidate-manifest.json')
    tuples = np.load(candidate['tuple_index'], allow_pickle=False)
    rows = sorted(candidate['sequences'], key=lambda r:r['sequence_ordinal'])
    counts = Counter(); event_counts = Counter(); checked = 0; consumer_checks = 0
    for row in rows:
        ordinal = row['sequence_ordinal']; binding = read(row['reset_binding'])
        instruction = read(row['goal_instruction']); validate_instruction(instruction)
        assert row['goal_certification'] == 'CERTIFIED_LEGAL'
        assert sha256(binding['intent_artifact']) == binding['source_artifact_sha256']
        # This field is the signed canonical instruction ID, not its JSON
        # file-byte SHA. validate_instruction verifies the unsigned payload.
        assert instruction['instruction_sha256'] == binding['instruction_sha256']
        assert sha256(row['geometry']) == row['geometry_sha256']
        assert binding['source_available_timestamp_ns'] < binding['reset_timestamp_ns']
        assert instruction['reset'] == binding['reset']
        intents = read(binding['intent_artifact'])
        intent = next(r for r in intents if r['target_execution_id'] == binding['reset']['execution_id'])
        assert intent['source_role'] == 'explicit_task_instruction'
        assert intent['source_execution_id'] != intent['target_execution_id']
        assert intent['available_at_ns'] == binding['source_available_timestamp_ns']
        own = tuples[tuples[:,0] == ordinal]
        assert len(own) == 274 and np.array_equal(own[:,2],np.arange(3,277))
        seq = {f.stem:np.load(f,mmap_mode='r',allow_pickle=False) for f in Path(row['sequence']).glob('*.npy')}
        contact = {f.stem:np.load(f,mmap_mode='r',allow_pickle=False) for f in Path(row['contact']).glob('*.npy')}
        with np.load(row['geometry'],allow_pickle=False) as z: geo = {k:z[k] for k in z.files}
        with np.load(Path(row['sequence']).parent/'raw.npz',allow_pickle=False) as raw:
            assert (np.diff(raw['observed_at_ns']) > 0).all()
            assert binding['reset_timestamp_ns'] <= int(raw['observed_at_ns'][0])
        # The immutable instruction is shared by the rollout. Validate it once
        # and check every current-pose encoding in a batch; repeatedly hashing
        # the same 2048-point JSON at every overlapping window adds no evidence.
        assert (own[:,1] == instruction['object_slot']).all()
        query = np.asarray(geo['query'],np.float64)
        assert np.array_equal(query,np.asarray(instruction['sources']['query']['payload'],np.float64))
        poses = np.asarray(seq['object_poses'][own[:,2],instruction['object_slot']],np.float64)
        relative = np.linalg.inv(poses) @ np.asarray(instruction['task_goal_pose'],np.float64)
        delta = np.einsum('tij,pj->tpi',relative[:,:3,:3],query)+relative[:,None,:3,3]-query
        assert np.isfinite(delta).all() and delta.shape == (274,32,3)
        for position in (0,len(own)-1):
            _,slot,tick = own[position]
            sampled = reset_bound_sample(seq,geo,contact,int(tick),int(slot),4,[ordinal,int(slot),int(tick)],instruction,binding['reset'])
            assert sampled['object_goal_valid'].all()
            assert np.allclose(sampled['goal_object_delta'],delta[position],rtol=1e-5,atol=1e-7)
            consumer_checks += 1
        checked += len(own)
        counts[instruction.get('source_role', binding['source_role'])] += len(own)
        event_counts['UNKNOWN_NO_INDEPENDENT_EVENT_INTENT'] += len(own)
        if len(counts) and checked % (274*16) == 0:
            progress(stage='goal',certified_windows=checked)
    assert checked == 26304
    return dict(status='PASS', old6000_certified_goal=0, full_legacy_certified_goal=0,
        full_legacy_windows=11197332, new_certified_goal_windows=checked, new_rollouts=len(rows),
        by_goal_role=dict(counts), by_event_certification=dict(event_counts),consumer_windows_checked=consumer_checks,
        fresh_candidate_manifest_sha256=sha256(FRESH/'candidate-manifest.json'),
        combined_catalog_exists=False, fresh_split='unassigned candidate; official TRAIN assets only',
        legal_goal_comparison_ready=False,
        missing='frozen paired TRAIN/DEV recipe and independent intended-event certificate; legacy cannot be recertified from future poses')


def smoke(out):
    import torch
    from ptv3_shared_wam_v6.model import SharedWAMV6
    from ptv3_shared_wam_v6.objectives import forecast_loss
    from ptv3_shared_wam_v6.goal_policy import assert_unknown_inputs
    from ptv3_shared_wam_v6.data import V6Windows
    ds = datasets(); config_path = out/'smoke-config.json'
    config = read(ROOT/'outputs/ptv3-shared-wam-v6/full-scale-20261023-r1/config-b96-w4.json')
    config.update(goal_input_policy=UNKNOWN_GOAL_POLICY,event_input_policy=UNKNOWN_EVENT_POLICY,
                  batch_size=1,training_allowed=False, purpose='forward/loss smoke only; zero optimizer updates')
    write(config_path,config)
    recipe = read(ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json')
    stats_path = Path(recipe['stats']); stats = read(stats_path)
    production_rejection = None
    try:
        V6Windows(MANIFEST,'train',goal_input_policy=UNKNOWN_GOAL_POLICY,event_input_policy=UNKNOWN_EVENT_POLICY)
    except ValueError as exc:
        production_rejection = str(exc)
    assert production_rejection
    torch.set_num_threads(2); torch.manual_seed(170); torch.cuda.manual_seed_all(170)
    torch.cuda.set_device(0)
    model = SharedWAMV6(stats,config).cuda().eval().requires_grad_(False)
    details=[]
    # Nine actual full-reader samples traverse the complete production model,
    # including its real PTv3 backbone and all enabled supervision objectives.
    with torch.no_grad():
        for source in sorted(s['source_id'] for s in ds['train'].manifest['sources']):
            row = next(r for r in ds['train'].rows.values() if r['source_id']==source and r['split']=='train')
            index = row['normalized_offset']
            batch = {k:v.cuda() for k,v in collate([ds['train'][index]]).items()}
            assert_unknown_inputs(batch)
            prediction = model(batch)
            assert all(torch.isfinite(v).all() for v in prediction.values() if torch.is_tensor(v) and v.is_floating_point())
            loss,terms = forecast_loss(prediction,batch,config,model.normalizer)
            assert torch.isfinite(loss)
            components={k:dict(numerator=float(n),denominator=float(d)) for k,(n,d) in terms['components'].items()}
            assert all(math.isfinite(v['numerator']) and math.isfinite(v['denominator']) for v in components.values())
            assert any(v['denominator'] > 0 for v in components.values())
            details.append(dict(source_id=source,index=index,tuple=ds['train'].index[index].tolist(),loss=float(loss),components=components))
            progress(stage='smoke',source_id=source,loss=float(loss))
    return dict(status='PASS', device=torch.cuda.get_device_name(0), physical_gpu_env='CUDA_VISIBLE_DEVICES=4',
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda, numpy=np.__version__,
        config_sha256=sha256(config_path),stats_path=str(stats_path),stats_sha256=sha256(stats_path),
        seed=170, seed_sha256=identity(dict(torch=170,numpy_sample=SEED)),
        production_reader_rejection=production_rejection, production_recipe_ready=False,
        parameter_count=sum(p.numel() for p in model.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), examples=details,
        weights='seeded random initialization; contract smoke, not quality evaluation',
        optimizer_created=False,backward_calls=0,training_updates=0,checkpoints_written=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['full','sample','hashes','geometry','goal','smoke'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.output).free > 31*2**30
    started=time.monotonic()
    result=globals()[args.stage](args.output)
    result.update(measured_at=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,
        git_commit=START_GIT_COMMIT,code_sha256=START_CODE_SHA256,
        host=platform.node(),python=sys.version,executable=sys.executable,numpy=np.__version__,
        manifest_sha256=sha256(MANIFEST),training_updates=0)
    write(args.output/(args.stage+'.json'),result)
    progress(stage=args.stage,status=result['status'],elapsed_s=result['elapsed_s'])


if __name__ == '__main__':
    main()
