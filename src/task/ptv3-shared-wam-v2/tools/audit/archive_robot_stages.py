"""Protect immutable saved stages from source-run checkpoint rotation.

CPU metadata/file handling only. Hard links survive unlink in the source run;
this never changes training code, configuration, files or processes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time


ROOT = Path(__file__).resolve().parents[5]
NAME = re.compile(r'step([0-9]+)\.pt$')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def capture(sources, output, records, budget):
    """Only final atomic-renamed step*.pt files; retain prior archive entries."""
    indexed = {row['step']: row for row in records}
    for source in sources:
        curve = {row['step']: row for row in json.loads((source / 'curve.json').read_text())}
        for path in source.glob('step*.pt'):
            match = NAME.fullmatch(path.name)
            if not match or path.is_symlink():
                continue
            try:
                source_stat = path.stat()
            except FileNotFoundError:
                continue
            step = int(match[1])
            destination = output / ('finetune-' + path.name)
            if destination.exists():
                saved = destination.stat()
                if (saved.st_dev, saved.st_ino) != (source_stat.st_dev, source_stat.st_ino):
                    raise ValueError('Archive identity collision: ' + str(destination))
            else:
                size = source_stat.st_size
                if sum(row['bytes'] for row in indexed.values()) + size > budget:
                    raise RuntimeError('Archive logical-byte budget exceeded')
                try:
                    os.link(path, destination)
                except FileNotFoundError:
                    # Concurrent source rotation can unlink an obsolete file.
                    # Never count an absent link as protected.
                    continue
                saved = destination.stat()
                if (saved.st_dev, saved.st_ino) != (source_stat.st_dev, source_stat.st_ino):
                    raise RuntimeError('Hard-link verification failed')
            indexed[step] = dict(step=step, path=destination.name, source=str(path),
                bytes=destination.stat().st_size, dev=curve.get(step),
                contents='full robot model (WAM+R), normalizers, optimizer, per-rank RNG')
    for row in indexed.values():
        if not (output / row['path']).is_file():
            raise FileNotFoundError('Previously protected archive entry is missing')
    if sum(row['bytes'] for row in indexed.values()) > budget:
        raise RuntimeError('Archive logical-byte budget exceeded')
    return sorted(indexed.values(), key=lambda row: row['step'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', type=Path, required=True)
    parser.add_argument('--active-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=14400)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--budget-bytes', type=int, default=16_000_000_000)
    args = parser.parse_args()
    output = args.output.resolve()
    sources = [path.resolve() for path in args.source]
    active = args.active_run.resolve()
    allowed = ROOT / 'outputs/ptv3-shared-wam-v2'
    if (active not in sources or any(allowed not in path.parents for path in sources + [output])
            or any(output == path or path in output.parents or output in path.parents for path in sources)
            or not 0 < args.seconds <= 14400 or not 1 <= args.poll_seconds <= 60
            or not 0 < args.budget_bytes <= 16_000_000_000):
        raise ValueError('Separate local archive, known active source and bounded operation required')
    manifests = [json.loads((path / 'manifest.json').read_text()) for path in sources]
    reference = manifests[0]
    for manifest in manifests:
        if (manifest.get('schema') != 'ref2dex.ptv3-shared-wam.v2'
                or manifest.get('stage') != 'robot' or manifest['config'] != reference['config']
                or manifest['provenance']['experiment_id'] != reference['provenance']['experiment_id']
                or manifest['provenance']['source'] != reference['provenance']['source']
                or manifest['provenance']['parent_checkpoint'] != reference['provenance']['parent_checkpoint']):
            raise ValueError('Sources must belong to the same frozen fine-tuning trajectory')
    output.mkdir(exist_ok=True)
    state_path = output / 'archive-manifest.json'
    records = json.loads(state_path.read_text())['records'] if state_path.exists() else []
    deadline = time.monotonic() + args.seconds
    try:
        while True:
            if shutil.disk_usage(output).free < 20 * 1024**3:
                raise RuntimeError('Filesystem headroom below20GiB')
            records = capture(sources, output, records, args.budget_bytes)
            for row in records:
                if row['dev'] is not None:
                    dev = row['dev']
                    dev['forecast_residual'] = (dev['objective']
                        - reference['config']['action_weight'] * dev['action']
                        - reference['config']['anchor_weight'] * dev['anchor'])
            manifest = json.loads((active / 'manifest.json').read_text())
            process = json.loads((active / 'process.json').read_text())
            status = 'RUNNING'
            if manifest.get('status') == 'COMPLETED':
                status = 'COMPLETED'
                if manifest['config']['updates'] not in {row['step'] for row in records}:
                    raise RuntimeError('Source completed without the final checkpoint archived')
            elif process.get('status') in ('FAILED', 'STOPPED', 'CANCELLED') or manifest.get('status') == 'FAILED':
                status = 'SOURCE_STOPPED'
            elif time.monotonic() >= deadline:
                status = 'DEADLINE_REACHED'
            state = dict(schema='ref2dex.checkpoint-stage-archive.v1', status=status,
                updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                sources=[str(path) for path in sources], active_run=str(active),
                method='hard links to immutable final checkpoints; source runs remain unchanged',
                budget_bytes=args.budget_bytes, logical_bytes=sum(row['bytes'] for row in records),
                records=records, pretrain_parent=reference['provenance']['parent_checkpoint'],
                lowest_recorded_loss_among_archived={key: min(
                    (row for row in records if row['dev'] is not None),
                    key=lambda row: row['dev'][key])['step']
                    for key in ('action', 'forecast_residual', 'objective')}
                    if any(row['dev'] is not None for row in records) else {},
                scope='saved stages only; cannot reconstruct checkpoints already rotated before archiving')
            atomic_json(state_path, state)
            print(json.dumps(dict(status=status, steps=[row['step'] for row in records])), flush=True)
            if status != 'RUNNING':
                break
            time.sleep(args.poll_seconds)
    except Exception as error:
        atomic_json(output / 'archive-error.json', dict(status='FAILED', error=repr(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
