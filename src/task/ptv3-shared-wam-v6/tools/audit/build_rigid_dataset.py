#!/usr/bin/env python3
"""Bounded rigid data closeout: real admission, provenance, dedup and coverage.

--plan computes the exact serialized write set without writing. --execute
publishes into a fresh output directory; source data and old outputs are read
only. No model/physics/optimizer calls, downloads or nonrigid schemas.
"""
import argparse
from collections import Counter, defaultdict
import csv
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
from scipy.spatial import cKDTree

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/ptv3-shared-wam-v5/src'), str(TASK / 'tools/audit')]
from ptv3_shared_wam_v5.cache_contract import metadata_identity, array_identity
from ptv3_shared_wam_v6.rigid_dataset import (
    admission_reasons, numerical_pose_valid, pca_shape, geometry_class, duplicate_of,
)
from curate_tau_training_set import phase_for, near_duplicate_fraction

DATA = Path('/home2/wyy/oyx_ws/Ref2Dex/data/processed_data')
PRIOR = TASK / 'docs/audit/brief35-source-role-20261023-r1'
CURATED = ROOT / 'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json'
EXPERIMENT = 'P-20261023-rigid-dataset-admission'
PHASES = ('approach', 'close', 'contact', 'hold', 'unknown')
CLASSES = ('flat', 'elongated', 'compact', 'unknown')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def npy_bytes(value):
    stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def arrays(folder, names):
    return {name: np.load(Path(folder) / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            for name in names if (Path(folder) / (name + '.npy')).exists()}


def artifact(path):
    path = Path(path)
    s = path.stat()
    return dict(path=str(path.resolve()), sha256=sha(path), bytes=s.st_size,
                filesystem_mtime_ns=s.st_mtime_ns, timestamp_scope='filesystem_only_not_reset_proof')


def legacy(payloads):
    cmroot = DATA / 'cm_decoder_v2/dexplore_rl_v1_3_full10135'
    coroot = DATA / 'coupled_geometric_cache_v1_20260912'
    cm = read(cmroot / 'manifest.json'); co = read(coroot / 'index.json')
    peers = {r.get('parent_seq_id', r['id']): r for r in co['sequences']['train']}
    records, summary = [], []
    for name, root, manifest in [('cm_decoder_v2', cmroot, cm), ('coupled', coroot, co)]:
        picks = sorted(manifest['sequences']['train'], key=lambda r: str(r.get('parent_seq_id', r['id'])))[:3]
        for row in picks:
            parent = row.get('parent_seq_id', row['id']); n = int(row['frame_count'])
            peer = peers[parent]; geometry_dir = coroot / peer['path'] / 'geometry'
            # cm row.path is stale. Resolve only explicit array/view_manifest
            # paths; never search by object name or substitute a human parent.
            folder = Path(row['view_manifest']).parent if name == 'cm_decoder_v2' else root / row['path']
            seq = arrays(folder, ('joints', 'joint_valid', 'surface_offsets', 'surface_valid',
                                  'surface_parts', 'object_poses', 'object_pose_valid'))
            contact = arrays(folder / 'contact', ('contact', 'part', 'valid'))
            event = arrays(folder / 'events', ('state', 'event_type', 'event_delta_steps', 'event_valid'))
            raw = arrays(geometry_dir, ('obj_pose_world', 'obj_points_pool_world', 'obj_normals_pool_world',
                                        'frame_time', 'source_frame_id'))
            poses = np.asarray(raw['obj_pose_world'])
            clock = np.asarray(raw['frame_time']); ids = np.asarray(raw['source_frame_id'])
            if poses.shape != (n, 4, 4) or len(clock) != n or len(ids) != n:
                raise ValueError('peer clock/length binding failure: ' + parent)
            clock_ok = (np.allclose(np.diff(clock), 1 / 30, atol=2e-6)
                        and np.all(np.diff(ids) == 4) and ids[0] == row['source_frame_first']
                        and ids[-1] == row['source_frame_last'])
            if not clock_ok:
                raise ValueError('not aligned observed30Hz source: ' + parent)
            q_binding = None
            if name == 'cm_decoder_v2':
                wrist = np.load(row['wrist_pose_world'], allow_pickle=False)
                local = np.load(row['wrist_pose_object_t'], allow_pickle=False)
                q_binding = float(np.max(np.abs(poses @ local - wrist)))
                if q_binding > 1e-4:
                    raise ValueError('cm/coupled pose-frame mismatch: ' + parent)
            numeric_valid = numerical_pose_valid(poses)
            world = np.asarray(raw['obj_points_pool_world'])
            normals = np.asarray(raw['obj_normals_pool_world'])
            canonical = np.einsum('tpi,tij->tpj', world - poses[:, None, :3, 3], poses[:, :3, :3])
            canonical_normals = np.einsum('tpi,tij->tpj', normals, poses[:, :3, :3])
            drift = float(np.max(np.abs(canonical - canonical[0])))
            ndrift = float(np.max(np.abs(canonical_normals - canonical_normals[0])))
            rigid = bool(numeric_valid.all() and drift <= 1e-4 and ndrift <= 1e-4)
            if not rigid:
                raise ValueError('rigid registry transport failed: ' + parent)
            # Fixed existing point IDs preserve correspondence; no resampling,
            # object identity or learned geometry enters this operation.
            chosen = np.arange(0, 4096, 2)
            geo = dict(surface=canonical[0, chosen].astype(np.float32),
                       normals=canonical_normals[0, chosen].astype(np.float32),
                       query=canonical[0, chosen[::64]].astype(np.float32))
            key = parent.replace('/', '_')
            prefix = 'legacy-bindings/' + key
            for field, value in geo.items():
                payloads[prefix + '/' + field + '.npy'] = npy_bytes(value)
            payloads[prefix + '/numerical_pose_valid.npy'] = npy_bytes(numeric_valid)
            payloads[prefix + '/object_poses.npy'] = npy_bytes(poses[:, None])
            pose_evidence = artifact(geometry_dir / 'obj_pose_world.npy')
            binding = dict(parent_sequence=parent, geometry_source=str(geometry_dir),
                geometry_array_identity=array_identity(geo), selected_pool_ids=chosen.tolist(),
                query_selected_surface_ids=list(range(0, 2048, 64)),
                canonical_frame='inverse(obj_pose_world_t) @ world_pool_t', units='m',
                max_position_drift_m=drift, max_normal_drift=ndrift,
                clock30hz_verified=True, source_frame_stride=4, pose_artifact=pose_evidence,
                pose_validity_scope='Numerical SE3 plus source clock verified; missing annotation-validity evidence is NOT filled with True.',
                geometry_binding_verified=True)
            payloads[prefix + '/binding.json'] = json_bytes(binding)
            # A new registry is usable; pose numerical quality is separately
            # written but does not manufacture an annotation validity mask.
            seq.setdefault('object_poses', poses[:, None])
            hand_meta = read(folder / 'meta.json') if (folder / 'meta.json').exists() else {}
            evidence = dict(hand_meta, frames=n, slots=1, geometry_binding_verified=True)
            reasons = Counter()
            for tick in range(3, n - 24):
                rejected = admission_reasons(seq, geo, contact, event, evidence, tick, 0)
                reasons.update(rejected)
                records.append(dict(root=name, parent_sequence_id=parent, tick=tick, slot=0,
                                    admitted=not rejected, reasons=rejected))
            summary.append(dict(root=name, parent_sequence_id=parent, frames=n,
                source_path=str(folder), stale_manifest_path_resolved=name == 'cm_decoder_v2',
                candidate_windows=n-27, admitted=sum(not r['reasons'] for r in records[-(n-27):]),
                rejected=sum(bool(r['reasons']) for r in records[-(n-27):]),
                reasons=dict(reasons), numerical_valid_frames=int(numeric_valid.sum()),
                cm_peer_pose_max_error=q_binding, registry=prefix + '/binding.json'))
    if len(records) != 1564:
        raise ValueError('frozen1564 pilot changed')
    payloads['legacy-window-verdicts.jsonl'] = b''.join(json_bytes(r) for r in records)
    counts = Counter(x for r in records for x in r['reasons'])
    combinations = Counter('+'.join(r['reasons']) for r in records if r['reasons'])
    return dict(candidate_windows=len(records), admitted=sum(r['admitted'] for r in records),
                rejected=sum(not r['admitted'] for r in records), reasons=dict(counts),
                reason_combinations=dict(combinations), sequences=summary,
                unique_parent_sequences=len({r['parent_sequence_id'] for r in records}),
                selection='unchanged first3 TRAIN rows per root;6 views are3 parents')


def feature(seq, contact, geo, slot, tick, phase):
    w = slice(tick - 3, tick + 25)
    p = np.asarray(seq['object_poses'][w, slot], np.float64)
    anchor = np.linalg.inv(p[3]); rel = anchor @ p
    j = np.asarray(seq['joints'][w], np.float64)
    jv = np.asarray(seq['joint_valid'][w], bool)
    hand = np.where(jv[..., None], j @ anchor[:3, :3].T + anchor[:3, 3], 0)
    # Every point in the frozen query32, every H4/K24 tick: captures both
    # translation and rotation rather than just endpoint/range statistics.
    query = np.einsum('tij,pj->tpi', rel[:, :3, :3], geo['query']) + rel[:, None, :3, 3]
    cv = np.asarray(contact['valid'][w, slot], bool)
    confidence = np.asarray(contact['contact'][w, slot])
    part = np.asarray(contact['part'][w, slot])
    fractions = np.zeros((28, 16), np.float32)
    for k in range(16):
        fractions[:, k] = ((part == k) & (confidence >= .25)).mean(-1) * cv
    shape = pca_shape(geo['surface'])
    trajectory = np.concatenate([hand.reshape(-1), query.reshape(-1)]).astype(np.float32)
    return dict(shape=shape, phase=phase, trajectory=trajectory,
                mask=np.concatenate([jv.reshape(-1), cv, np.asarray(seq['object_pose_valid'][w, slot], bool)]),
                contact=fractions.reshape(-1), rotations=rel[:, :3, :3].astype(np.float32),
                coarse=np.concatenate([np.log(np.maximum(shape, 1e-9)) / np.log(1.05),
                                       hand[3].reshape(-1)[:6] / .002]))


def coverage(rows):
    count = Counter((r['phase'], r['geometry_class'], int(r['source_id'])) for r in rows)
    sources = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    matrix = [dict(phase=p, geometry_class=g, source_id=s, windows=count[p, g, s])
              for p in PHASES for g in CLASSES for s in sources]
    return dict(windows=len(rows), phases=dict(Counter(r['phase'] for r in rows)),
                sources=dict(Counter(str(r['source_id']) for r in rows)),
                sequences=len({r['sequence_key'] for r in rows}),
                object_instances=len({(r['source_id'], r['object']) for r in rows}),
                matrix=matrix, zero_requested_cells=sum(r['windows'] == 0 for r in matrix
                    if r['phase'] != 'unknown' and r['geometry_class'] != 'unknown'))


def curated(payloads, path, deadline):
    cur = read(path); root = Path(cur['data_root']); parent = read(root / 'recipe/manifest.json')
    rows = cur['selection']['train_window_rows']; seqrows = parent['sequences']
    cert = list(csv.DictReader((PRIOR / 'window-certification.csv').open()))
    keys = lambda rows: [(int(r['source_id']), int(r['sequence_id']), int(r['slot']), int(r['tick'])) for r in rows]
    if keys(rows) != keys(cert) or len(rows) != 6000:
        raise ValueError('certification must join exact frozen6000 ordered tuples')
    mothers = {r['sequence_id']: r for r in read(PRIOR / 'mother-provenance.json')}
    erows = {r['key']: r for r in read(root / 'events/manifest.json')['sequences']}
    gpath = Path(parent['geometry_manifest']); gmeta = read(gpath)
    geometries = {}
    for r in gmeta['objects']:
        with np.load(gpath.parent / r['path'], allow_pickle=False) as z:
            geometries[r['key']] = {k: z[k] for k in z.files}
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[int(r['sequence_id'])].append(i)
    features = [None] * len(rows); verdicts = [None] * len(rows); certification = [None] * len(rows)
    for number, (sid, ordinals) in enumerate(sorted(grouped.items()), 1):
        if time.monotonic() > deadline:
            raise TimeoutError('bounded20min audit deadline')
        row = seqrows[sid]; folder = Path(row['path']); cf = Path(row['contact_path'])
        seq = arrays(folder, ('joints', 'joint_valid', 'surface_offsets', 'surface_valid',
                             'surface_parts', 'object_poses', 'object_pose_valid'))
        contact = arrays(cf, ('contact', 'part', 'valid')); event_row = erows.get(row['key'])
        event = arrays(event_row['path'], ('state', 'event_type', 'event_delta_steps', 'event_valid')) if event_row else {}
        hm = read(folder / 'meta.json'); cm = read(cf / 'meta.json')
        hand_bound = metadata_identity(folder / 'meta.json') == row['hand_meta_identity']
        contact_bound = metadata_identity(cf / 'meta.json') == row['contact_meta_identity']
        event_bound = bool(event_row and event_row['contact_meta_identity'] == row['contact_meta_identity'])
        current_pose = artifact(folder / 'object_poses.npy'); previous = mothers[sid]
        if current_pose['sha256'] != previous['current_label_sha256']:
            raise ValueError('prior provenance stale: target pose changed')
        for ordinal in ordinals:
            r = dict(rows[ordinal]); slot = int(r['slot']); tick = int(r['tick'])
            name = row['objects'][slot]; key = 's%d:%s' % (row['source_id'], name) if row['source_id'] < 3 else name
            geo = geometries.get(key, {})
            gb = bool(geo) and cm.get('geometry_identities', {}).get(name) == array_identity(geo)
            evidence = dict(hm, label_semantics=cm.get('label_semantics'),
                binding_verified=hand_bound and contact_bound and cm.get('hand_identity') == row['hand_meta_identity'],
                event_binding_verified=event_bound, geometry_binding_verified=gb)
            reasons = admission_reasons(seq, geo, contact, event, evidence, tick, slot)
            r['phase'] = phase_for(event, slot, tick) if event else 'unknown'
            r['geometry_class'] = geometry_class(pca_shape(geo['surface'])) if geo else 'unknown'
            r['current_admission_reasons'] = reasons; r['current_admitted'] = not reasons
            verdicts[ordinal] = r
            if not reasons:
                features[ordinal] = feature(seq, contact, geo, slot, tick, r['phase'])
            old = cert[ordinal]
            # Reuse checked previous evidence, not a new historical census.
            # UNKNOWN stays UNKNOWN, even though label trajectory is known.
            if old['waypoint_status'] != 'UNCERTAIN_MISSING_RECORDS':
                raise ValueError('new provenance evidence requires explicit inspection')
            certification[ordinal] = dict(ordinal=ordinal, source_id=r['source_id'], sequence_id=sid,
                slot=slot, tick=tick, source_role='UNKNOWN', available_before_reset=None,
                certification='UNCERTIFIABLE_MISSING_RECORDS', reset_timestamp_ns=None,
                source_available_timestamp_ns=None, source_artifact_sha256=None,
                label_source_role='recorded_target_rollout', label_artifact=current_pose,
                label_endpoint_frame=tick+24, goal_uses_target_future=True,
                missing=old['missing'].split(';'), evidence_scope='prior strict certification + fresh ordered tuple/pose hash join')
        if number % 250 == 0:
            print(json.dumps(dict(stage='curated', sequences=number, total=len(grouped))), flush=True)
    valid_ordinals = [i for i, f in enumerate(features) if f is not None]
    # KD tree provides a necessary (not sufficient) content filter. Radius1
    # in infinity norm cannot miss a pair satisfying the exact predicate.
    tree = cKDTree(np.stack([features[i]['coarse'] for i in valid_ordinals])) if valid_ordinals else None
    retained = set(); pairs = []; comparisons = 0
    for local, ordinal in enumerate(valid_ordinals):
        if time.monotonic() > deadline:
            raise TimeoutError('bounded20min dedup deadline')
        f = features[ordinal]
        candidates = tree.query_ball_point(f['coarse'], r=1.000001, p=np.inf)
        representative = None
        for candidate in sorted(valid_ordinals[j] for j in candidates if valid_ordinals[j] in retained):
            comparisons += 1
            if duplicate_of(f, features[candidate]):
                representative = candidate; break
        if representative is None:
            retained.add(ordinal)
        else:
            pairs.append(dict(removed_ordinal=ordinal, retained_ordinal=representative,
                same_sequence=rows[ordinal]['sequence_id'] == rows[representative]['sequence_id'],
                tick_distance=abs(rows[ordinal]['tick'] - rows[representative]['tick'])))
    kept = [verdicts[i] for i in sorted(retained)]
    admitted = [r for r in verdicts if r['current_admitted']]
    payloads['curated-window-verdicts.jsonl'] = b''.join(json_bytes(r) for r in verdicts)
    payloads['window-provenance.jsonl'] = b''.join(json_bytes(r) for r in certification)
    payloads['dedup-pairs.json'] = json_bytes(pairs)
    payloads['deduplicated-train-tuples.npy'] = npy_bytes(np.asarray(
        [[r['sequence_id'], r['slot'], r['tick']] for r in kept], np.int64).reshape(-1, 3))
    payloads['deduplicated-ordinals.npy'] = npy_bytes(np.asarray(sorted(retained), np.int64))
    payloads['legal-goal-tuples.npy'] = npy_bytes(np.empty((0, 3), np.int64))
    before = coverage(admitted); after = coverage(kept)
    for name, data in [('before', before), ('after', after)]:
        payloads['coverage-' + name + '.csv'] = csv_bytes(data['matrix'], ['phase', 'geometry_class', 'source_id', 'windows'])
    payloads['fixed-panel.json'] = json_bytes(dict(anchor_id=cur['selection']['anchor_id'],
        anchor_rows=cur['selection']['anchor_rows'], source=str(path), unchanged=True))
    return dict(input_curated_windows=len(rows), admitted=len(admitted), rejected=len(rows)-len(admitted),
        rejection_reasons=dict(Counter(code for r in verdicts for code in r['current_admission_reasons'])),
        provenance=dict(certified_legal=0, uncertifiable=len(certification),
            source_role='UNKNOWN', available_before_reset=None,
            missing_window_counts=dict(Counter(x for r in certification for x in r['missing'])),
            prior_receipt=artifact(PRIOR/'receipt.json'), ordered_join='PASS'),
        dedup=dict(before=len(admitted), after=len(kept), removed=len(pairs), exact_comparisons=comparisons,
            cross_sequence_removed=sum(not p['same_sequence'] for p in pairs),
            temporal_near_repeat_percent_before=near_duplicate_fraction(admitted),
            temporal_near_repeat_percent_after=near_duplicate_fraction(kept),
            rule='shape PCA extents5%; full H4/K24 anchored joint21+query32 coordinates L_inf2mm; object rotation1degree; identical validity+phase; each28x16 contact fraction2%; earliest frozen ordinal representative, no chaining',
            limitations='Conservative native object-coordinate comparison. Differently oriented canonical CADs may escape detection. No general cross-CAD shape equivalence claim.'),
        coverage_before=before, coverage_after=after,
        curated_artifact=artifact(path), fixed_panel_id=cur['selection']['anchor_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--curated', type=Path, default=CURATED)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true'); mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('fresh output required; preserve all previous outputs')
    if output.parent != (ROOT/'outputs/ptv3-shared-wam-v6').resolve():
        raise ValueError('task-owned output root required')
    payloads = {}
    legacy_result = legacy(payloads)
    print(json.dumps(dict(stage='legacy', admitted=legacy_result['admitted'], rejected=legacy_result['rejected'])), flush=True)
    result = curated(payloads, args.curated, started+1200)
    if legacy_result['admitted']:
        # No implicit expansion, mixing or training when a previously
        # unsupported native semantic adapter unexpectedly becomes present.
        raise ValueError('legacy now has admitted data: bind provenance and append via reviewed immutable manifest before expansion')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    report = dict(schema='ref2dex.rigid-dataset-closeout.v1', task='ptv3-shared-wam-v6',
        experiment_id=EXPERIMENT, run_id=output.name, work_version='rigid-data-audit.v1',
        git_commit=commit, branch=branch, measured_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
            scipy=__import__('scipy').__version__, host=platform.node(), device='CPU cached data/IO; no model or inference', gpu_seconds=0),
        legacy=legacy_result, curated=result, training_updates=0, training_allowed=False,
        decision='OWNER_DECISION_NEW_COLLECTION_OR_EXPLICIT_NATIVE_SEMANTIC_ANNOTATION',
        source_read_only=True, historical_outputs_deleted=0, expanded_data_sources=False,
        elapsed_s=round(time.monotonic()-started, 3))
    payloads['report.json'] = json_bytes(report)
    # Exact serialized bytes and exact file-size allocation rounding, before
    # any conversion/artifact write. Directory/journal usage is FS-dependent.
    free = shutil.disk_usage(output.parent).free
    block = __import__('os').statvfs(output.parent).f_frsize
    exact = sum(len(v) for v in payloads.values())
    allocation = sum(math.ceil(len(v)/block)*block for v in payloads.values())
    preflight = dict(files=len(payloads), exact_serialized_bytes=exact,
        allocated_file_bytes_upper=allocation, metadata_reserve_bytes=2**20,
        disk_free_before_bytes=free, free_after_reserved_bytes=free-allocation-2**20,
        storage_floor_bytes=20*2**30, mode='EXECUTE' if args.execute else 'PLAN_NO_WRITES')
    preflight_bytes = json_bytes(preflight)
    # Include the preflight file itself in the recorded overall bound.
    preflight['exact_serialized_bytes'] += len(preflight_bytes)
    preflight['allocated_file_bytes_upper'] += block
    preflight['free_after_reserved_bytes'] -= block
    preflight['files'] += 1
    for _ in range(4):
        preflight_bytes = json_bytes(preflight)
        preflight['exact_serialized_bytes'] = exact + len(preflight_bytes)
    payloads['storage-preflight.json'] = json_bytes(preflight)
    assert sum(map(len, payloads.values())) == preflight['exact_serialized_bytes']
    print(json.dumps(dict(storage_preflight=preflight, legacy=legacy_result['admitted'],
                         curated_admitted=result['admitted'], dedup=result['dedup'],
                         coverage_zeros_before=result['coverage_before']['zero_requested_cells'],
                         coverage_zeros_after=result['coverage_after']['zero_requested_cells'])), flush=True)
    if preflight['free_after_reserved_bytes'] < 20*2**30:
        raise RuntimeError('storage floor violated before writing')
    if args.execute:
        output.mkdir(exist_ok=False)
        for relative, data in payloads.items():
            if shutil.disk_usage(output).free < 20*2**30:
                raise RuntimeError('storage floor: preserve partial evidence')
            dest = output/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(data)
        actual = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        assert actual == preflight['exact_serialized_bytes']
        print(json.dumps(dict(status='COMPLETED', output=str(output), actual_bytes=actual)), flush=True)


if __name__ == '__main__':
    main()
