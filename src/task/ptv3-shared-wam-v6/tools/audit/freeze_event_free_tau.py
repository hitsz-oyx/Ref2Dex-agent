#!/usr/bin/env python3
"""Freeze an event-free policy and fail-closed launch preflight; never train.

The adapters below are auditable prototypes. They are NOT installed into the
production reader or runtime. A passing prototype must not release training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6.events import event_packet, event_features
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v6.online_runtime import online, GOAL_SEMANTICS

POLICY = 'v6.event-free.unknown-invalid.frame30.v1'
EVENT_DTYPES = dict(event_type=np.int64, event_delta_steps=np.float32,
                    event_valid=np.bool_, event_hz=np.int64)


def unknown_packet():
    """One existing canonical packet constructor, including the validity bit."""
    packet = event_packet('NONE', 0., False)
    return {key: EVENT_DTYPES[key](value) for key, value in packet.items()}


def event_free_sample(sample):
    """Change only four event inputs; never touch supervision or goal tensors."""
    return dict(sample, **unknown_packet())


def event_free_collate(samples):
    return collate([event_free_sample(sample) for sample in samples])


def event_free_predict(runtime, objects, hands, goals, events, object_codes,
                       *, goal_semantics=GOAL_SEMANTICS):
    """Prototype deployment adapter; supplied event contents are never read."""
    if len(events) != len(objects):
        raise ValueError('one event slot per object required')
    return runtime.predict(objects, hands, goals,
                           [unknown_packet() for _ in objects], object_codes,
                           goal_semantics=goal_semantics)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return sha(Path(path).read_bytes())


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def tuple_rows(manifest):
    rows = manifest['selection']['train_window_rows']
    result = []
    for row in rows:
        if row['split'] != 'train' or not row['admitted']:
            raise ValueError('only admitted TRAIN rows may enter the frozen list')
        result.append([int(row['source_id']),
                       [int(row['sequence_id']), int(row['slot']), int(row['tick'])],
                       int(row['tick'])])
    keys = [canonical_bytes(row) for row in result]
    if len(result) != 6000 or len(set(keys)) != 6000:
        raise ValueError('exactly 6000 distinct curated windows required')
    return result


def measure_resources():
    gpu = subprocess.run([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    processes = subprocess.run([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if gpu.returncode or processes.returncode:
        return dict(status='FAILED_GPU_QUERY', error=gpu.stderr+processes.stderr,
                    filesystem_free_bytes=shutil.disk_usage('/home2').free)
    active_uuids = {line.split(',')[0].strip()
                    for line in processes.stdout.splitlines() if line.strip()}
    rows = []
    for line in gpu.stdout.splitlines():
        index, uuid, used, total, util = [part.strip() for part in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, memory_used_mib=int(used),
                         memory_total_mib=int(total), utilization_percent=int(util),
                         has_compute_process=uuid in active_uuids,
                         owner_reserved=int(index) in (4, 5, 6, 7)))
    available = [r['index'] for r in rows if not r['owner_reserved']
                 and not r['has_compute_process'] and r['memory_used_mib'] < 256
                 and r['utilization_percent'] == 0]
    return dict(status='MEASURED', measured_at=datetime.now(timezone.utc).isoformat(),
                filesystem_free_bytes=shutil.disk_usage('/home2').free,
                gpus=rows, compute_processes_csv=processes.stdout,
                available_unreserved_gpus=available,
                required_free_gpus_after_launch=2)


def boundary_probe(recipe_path, curated):
    """Existing production reader + collate; no optimizer/model/GPU execution."""
    dataset = V6Windows(recipe_path, 'train', geometry_gt_object=True)
    sequence_ids = {r['key']: i for i, r in enumerate(dataset.meta['sequences'])}
    chosen = {}
    for row in curated['selection']['train_window_rows']:
        chosen.setdefault(int(row['source_id']), row)
    samples, evidence = [], []
    for sid, row in sorted(chosen.items()):
        sequence_id = sequence_ids[row['sequence_key']]
        index = np.flatnonzero((dataset.index[:, 0] == sequence_id)
                               & (dataset.index[:, 1] == row['slot'])
                               & (dataset.index[:, 2] == row['tick']))
        if len(index) != 1:
            raise ValueError('curated tuple is not uniquely bound in existing reader')
        sample = dataset[int(index[0])]
        samples.append(sample)
        evidence.append(dict(source_id=sid, curated_sample_id=[row['sequence_id'],
                             row['slot'], row['tick']], reader_index=int(index[0]),
                             reader_sample_id=sample['sample_id'].tolist()))
    raw = collate(samples)
    masked = event_free_collate(samples)
    non_event = [key for key in raw if key not in EVENT_DTYPES]
    unchanged = all(torch.equal(raw[key], masked[key]) for key in non_event)
    if not unchanged:
        raise AssertionError('event-free prototype changed supervision/geometry/goal')
    features, valid = event_features(masked)
    # Use the actual deployment sample builder and the SAME packet constructor.
    geo = dict(surface=np.zeros((2048, 3), np.float32),
               normals=np.zeros((2048, 3), np.float32),
               query=np.zeros((32, 3), np.float32))
    deployment = collate([online(geo, np.tile(np.eye(4), (4, 1, 1)),
        np.zeros((4, 2, 21, 3), np.float32), np.zeros((4, 2, 778, 3), np.float32),
        np.ones((2, 21), bool), np.zeros((2, 778), bool),
        np.zeros((2, 778), np.int64), np.eye(4), unknown_packet()) for _ in samples])
    packet_equal = all(torch.equal(masked[k], deployment[k]) for k in EVENT_DTYPES)
    for i, record in enumerate(evidence):
        record.update(raw_event={key: raw[key][i].item() for key in EVENT_DTYPES},
                      frozen_event={key: masked[key][i].item() for key in EVENT_DTYPES},
                      reader_goal_equals_realized_endpoint=bool(torch.equal(
                          raw['goal_object_delta'][i], raw['scene_target'][i, 0, -1])))
    return dict(status='PROTOTYPE_ONLY_NOT_PRODUCTION_WIRING',
                reader_windows=len(dataset), examples=evidence,
                event_shapes={key: list(masked[key].shape) for key in EVENT_DTYPES},
                event_dtypes={key: str(masked[key].dtype) for key in EVENT_DTYPES},
                model_event_features_shape=list(features.shape),
                nonzero_event_features=int(torch.count_nonzero(features)),
                valid_count=int(valid.sum()), known_none_count=int(((masked['event_type']==0)&valid).sum()),
                non_event_fields_unchanged=unchanged, unchanged_fields=non_event,
                deployment_packet_bitwise_equal=packet_equal,
                production_training_wired=False, production_runtime_policy_enforced=False)


def launch_gates(probe, scheduler_receipt, resources, expected_bytes, world_size=3):
    gates = dict(
        shared_event_policy_installed_in_production=bool(
            probe['production_training_wired'] and probe['production_runtime_policy_enforced']),
        curated_training_reader_connected=probe['reader_windows'] == 6000,
        scheduler_goal_cache_available=scheduler_receipt['counts']['scheduler_eligible_windows'] == 6000,
        gpu_allocation_available=(resources.get('status') == 'MEASURED' and
            len(resources.get('available_unreserved_gpus', [])) >= world_size + 2),
        disk_available=resources['filesystem_free_bytes'] - expected_bytes >= 20 * 2**30)
    return dict(status='PREFLIGHT_PASS' if all(gates.values()) else 'BLOCKED_PREFLIGHT',
                gates=gates, training_updates=0, launch=None)


def write_new(path, value):
    # Never overwrite prior evidence, even for a failed preflight.
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode()+b'\n')


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output directory required')
    curated_path = ROOT/'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json'
    reader_recipe = ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'
    old_config = ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/config.json'
    checkpoint = old_config.parent/'step011000.pt'
    prior_receipt_path = ROOT/'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r9/scheduler-cache-identity-receipt.json'
    recipe_path = TASK/'configs/tau_event_free_v1.json'
    legacy_plan = TASK/'configs/tau_curated_data_only_v1.json'
    recipe = json.loads(recipe_path.read_text())
    if (recipe.get('policy') != POLICY or recipe.get('windows') != 6000
            or recipe.get('goal_semantics') != GOAL_SEMANTICS):
        raise ValueError('frozen full-curated event-free waypoint policy required')
    curated = json.loads(curated_path.read_text())
    tuples = tuple_rows(curated)
    tuples_payload = canonical_bytes(tuples)
    resources = measure_resources()
    prior = json.loads(prior_receipt_path.read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Bound the small audit BEFORE writing. No checkpoints are created here.
    estimate = 2 * 2**20
    if resources['filesystem_free_bytes'] - estimate < 20*2**30:
        raise ValueError('insufficient free space even for bounded preflight evidence')
    probe = boundary_probe(reader_recipe, curated)
    observed = []
    for path in sorted(checkpoint.parent.glob('step*.pt')):
        stat = path.stat()
        observed.append(dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size,
                             mtime_s=stat.st_mtime))
    maximum_checkpoint = max(r['bytes'] for r in observed)
    # Conservative: union of scheduled saves, every validation a new best,
    # and one emergency save. Rounded-up 25% payload growth + 128MiB logs.
    budget = recipe['budget_ceiling']
    last = budget['optimizer_updates']
    saves = set([1, budget['validation_initial_step'], budget['validation_probe_step'], last]
                + list(range(budget['validation_interval'], last+1, budget['validation_interval']))
                + list(range(budget['checkpoint_interval'], last+1, budget['checkpoint_interval'])))
    save_count = len(saves) + 1
    expected_bytes = (maximum_checkpoint*5+3)//4 * save_count + 128*2**20
    span = max(r['mtime_s'] for r in observed)-min(r['mtime_s'] for r in observed)
    historical_growth = (sum(r['bytes'] for r in observed[1:])/span*3600
                         if span > 0 and len(observed) > 1 else None)
    decision = launch_gates(probe, prior, resources, expected_bytes, budget['world_size'])
    receipt = dict(schema='ref2dex.tau-event-free-preflight.v1',
        experiment_id='P-20261023-tau-event-free', run_id='tau-event-free-20261023-r1',
        policy=POLICY, frozen_code_head=head,
        identities=dict(curated_manifest=dict(path=str(curated_path.relative_to(ROOT)), sha256=file_hash(curated_path)),
            ordered_window_tuples=dict(schema=['source_id', 'sample_id=[sequence_id,slot,tick]', 'window_start_frame=tick'],
                                       windows=6000, sha256=sha(tuples_payload)),
            recipe=dict(path=str(recipe_path.relative_to(ROOT)), sha256=file_hash(recipe_path)),
            reader_recipe=dict(path=str(reader_recipe.relative_to(ROOT)), sha256=file_hash(reader_recipe)),
            inherited_training_config=dict(path=str(old_config.relative_to(ROOT)), sha256=file_hash(old_config)),
            untouched_legacy_plan=dict(path=str(legacy_plan.relative_to(ROOT)), sha256=file_hash(legacy_plan),
                status=json.loads(legacy_plan.read_text())['status']),
            inherited_scheduler_receipt=dict(path=str(prior_receipt_path.relative_to(ROOT)), sha256=file_hash(prior_receipt_path))),
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda),
        audit_write_estimate_bytes=estimate, resources=resources, boundary_probe=probe,
        storage=dict(checkpoints_observed=observed,
            historical_retained_file_growth_bytes_per_hour=historical_growth,
            historical_rate_limit='retained checkpoint mtimes only; not measured new-run throughput',
            new_run_measured_checkpoint_growth_bytes_per_hour=None,
            projected_checkpoint_saves_max=save_count,
            projected_training_output_bytes=expected_bytes,
            artifact_alarm_bytes=32*2**30, disk_warning_free_bytes=30*2**30,
            disk_stop_free_bytes=20*2**30,
            projected_bytes_formula='ceil(max_observed_checkpoint_bytes*1.25) * max_saves + 128MiB',
            gpu_measurement_requires_before_launch=True),
        inherited_scheduler_eligible_windows=prior['counts']['scheduler_eligible_windows'],
        scheduler_receipt_replayed=False, config_status_modified=False,
        **decision)
    args.output.mkdir(parents=True)
    with (args.output/'window-tuples.json').open('xb') as stream:
        stream.write(tuples_payload)
    write_new(args.output/'preflight.json', receipt)
    written=sum(p.stat().st_size for p in args.output.iterdir())
    if written > estimate:
        raise ValueError('audit exceeded predeclared write estimate')
    print(json.dumps(dict(status=receipt['status'], gates=receipt['gates'],
                          output_bytes=written, training_updates=0), indent=2))


if __name__ == '__main__':
    main()
