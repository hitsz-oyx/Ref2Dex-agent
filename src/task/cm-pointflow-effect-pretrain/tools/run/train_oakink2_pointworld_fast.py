#!/usr/bin/env python3
"""Reference temporal trainer with process-local, integer-exact Hilbert fusion.

All reference CLI options and loss/microbatch semantics are retained. Importing
a reference checkpoint requires --resume and --import-reference-checkpoint.
Native fast checkpoints use ordinary --resume with strict extended source checks.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
spec = importlib.util.spec_from_file_location('reference_train', TASK/'tools/run/train_oakink2_pointworld_temporal.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
from oakink_wm.pointworld_performance import install_fused_hilbert, restore_hilbert


def extend_reference_checkpoint(state, reference_sources, fast_sources):
    """Explicit backend migration only; all reference identity must match."""
    if state.get('identity', {}).get('implementation_sources') != reference_sources:
        raise ValueError('reference checkpoint source mismatch; use native resume for fast checkpoints')
    result = dict(state)
    result['identity'] = dict(state['identity'], implementation_sources=fast_sources)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--import-reference-checkpoint', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    own, remaining = parser.parse_known_args()
    if own.import_reference_checkpoint and own.resume is None:
        parser.error('--import-reference-checkpoint requires --resume')
    if own.output is None and not any(x in remaining for x in ('-h', '--help')):
        parser.error('--output is required')
    if own.output is not None and (own.output/'input_manifest.json').exists():
        if own.resume is None or own.import_reference_checkpoint:
            parser.error('reference import requires a fresh output; existing fast run requires native --resume')
    if own.resume: remaining.extend(('--resume', str(own.resume)))
    if own.output: remaining.extend(('--output', str(own.output)))
    reference_sources = train.implementation_sources()
    extra_sources = {str(p.relative_to(TASK)):train.digest(p) for p in
                     (Path(__file__).resolve(), TASK/'src/oakink_wm/pointworld_performance.py')}
    fast_sources = dict(reference_sources, **extra_sources)
    train.implementation_sources = lambda: dict(fast_sources)
    original_load = torch.load
    def load(path, *args, **kwargs):
        state = original_load(path, *args, **kwargs)
        if own.import_reference_checkpoint and isinstance(path, (str, Path)) and Path(path).resolve() == own.resume.resolve():
            return extend_reference_checkpoint(state, reference_sources, fast_sources)
        return state
    torch.load = load
    original_batch = train.device_batch
    calls = [0]
    def guarded_batch(batch):
        if calls[0] % 160 == 0:
            if any(train.digest(TASK/key) != value for key, value in fast_sources.items()):
                raise RuntimeError('frozen implementation source drift')
        calls[0] += 1
        return original_batch(batch)
    train.device_batch = guarded_batch
    original_hilbert = install_fused_hilbert()
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__))] + remaining
        if own.output is not None:
            own.output.mkdir(parents=True, exist_ok=True)
            manifest_path = own.output/'performance_manifest.json'
            performance = dict(
                backend='integer-exact-triton-hilbert', implementation_sources=fast_sources,
                reference_checkpoint=str(own.resume.resolve()) if own.import_reference_checkpoint else None,
                reference_checkpoint_sha256=train.digest(own.resume) if own.import_reference_checkpoint else None,
                preserves_microbatch_and_objective=True, torch_version=torch.__version__)
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                if own.resume is None or previous['implementation_sources'] != fast_sources:
                    raise ValueError('existing performance identity mismatch')
            else:
                train.atomic_json(manifest_path, performance)
        train.main()
    finally:
        torch.load = original_load
        restore_hilbert(original_hilbert)
        sys.argv = original_argv


if __name__ == '__main__':
    main()
