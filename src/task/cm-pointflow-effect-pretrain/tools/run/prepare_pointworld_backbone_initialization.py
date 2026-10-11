#!/usr/bin/env python3
"""Convert released spatial weights for the existing native --init-weights path."""
import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.pointworld_temporal import model_from_config, VENDOR
from oakink_wm.multisource import MixedWindows

REVISION = 'b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba'
CHECKSUM = 'ccb9ed93dff5eea976010c57dd0cb5634db61c68b732c4437cbf54c8da9de8fe'
PREFIX = 'dynamics_predictor.predictor_model.'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_hash(state, *, exclude_backbone=False):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if exclude_backbone and key.startswith('backbone.core.'):
            continue
        h.update((key + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def import_backbone(core, source):
    """All source backbone keys must match; no partial or suffix-based loading."""
    selected = {k[len(PREFIX):]: v for k, v in source.items() if k.startswith(PREFIX)}
    target = core.state_dict()
    if not selected or set(selected) != set(target):
        raise ValueError('backbone keys differ: missing=%s extra=%s' %
                         (sorted(set(target)-set(selected)), sorted(set(selected)-set(target))))
    for key, value in selected.items():
        if value.shape != target[key].shape or value.dtype != target[key].dtype:
            raise ValueError('backbone shape/dtype differs: ' + key)
        if not torch.isfinite(value).all().item():
            raise ValueError('nonfinite released backbone: ' + key)
    core.load_state_dict(selected, strict=True)
    return dict(source_prefix=PREFIX, target_prefix='backbone.core.', tensors=len(selected),
                numel=sum(v.numel() for v in selected.values()), partial_loading=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('official', 'data', 'stats', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='ref2dex.native-video-initializers.v1', status='PREPARING',
                  official_revision=REVISION, official_sha256=CHECKSUM,
                  generator_sha256=digest(Path(__file__).resolve()), no_training=True)
    try:
        if a.official.stat().st_size != 1826853514 or digest(a.official) != CHECKSUM:
            raise ValueError('official asset size/hash differs')
        spec = importlib.util.spec_from_file_location('native_temporal_initializer',
                    TASK / 'tools/run/train_oakink2_pointworld_temporal.py')
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        config, stats = json.loads(a.config.read_text()), json.loads(a.stats.read_text())
        train = MixedWindows(a.data, 'train')
        if (stats['split'] != 'train' or
                stats['input_manifest_sha256'] != train.meta['normalization_source_manifest_sha256'] or
                digest(a.stats) != train.meta['normalization_stats_sha256']):
            raise ValueError('native normalization identity differs')
        random.seed(config['seed']); np.random.seed(config['seed'])
        torch.manual_seed(config['seed']); base.configure_numerics()
        model = model_from_config(stats, config)
        original = state_hash(model.state_dict())
        outside = state_hash(model.state_dict(), exclude_backbone=True)
        dataset_hash = digest(a.data / 'processed/manifest.json')
        identity = dict(arm='action', mixed_data=True, source_manifest=train.meta,
            dataset_hash=dataset_hash, stats_sha256=digest(a.stats), stats=stats,
            normalization_source_manifest_sha256=train.meta['normalization_source_manifest_sha256'],
            implementation_sources=base.implementation_sources(),
            vendor_sources={str(f.relative_to(VENDOR)): digest(f) for f in
                            (VENDOR / 'ptv3').rglob('*') if f.suffix in ('.py', '.yaml')},
            initializer_generator_sha256=report['generator_sha256'],
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            native_training_updates=0, seed=config['seed'])
        report.update(dataset_hash=dataset_hash, config_sha256=digest(a.config),
                      stats_sha256=digest(a.stats), random_state_sha256=original,
                      nonbackbone_state_sha256=outside, files={})
        for arm in ('random', 'video'):
            if arm == 'video':
                released = torch.load(a.official, map_location='cpu', mmap=True, weights_only=False)
                report['import'] = import_backbone(model.backbone.core, released['model'])
                del released
            current_outside = state_hash(model.state_dict(), exclude_backbone=True)
            if current_outside != outside:
                raise ValueError('initialization changed native features/actions/head/statistics')
            state = dict(checkpoint_kind='pointworld-native-initialization.v1', step=0,
                model=model.state_dict(), config=config, dataset_hash=dataset_hash,
                identity=dict(identity, initializer_kind=arm,
                    backbone_import=(dict(report['import'], checkpoint_sha256=CHECKSUM,
                        revision=REVISION) if arm == 'video' else None)))
            path = a.output / (arm + '.pt')
            torch.save(state, path)
            report['files'][arm] = dict(file=path.name, sha256=digest(path), bytes=path.stat().st_size,
                model_state_sha256=state_hash(model.state_dict()),
                nonbackbone_state_sha256=current_outside)
        if sum(v['bytes'] for v in report['files'].values()) > 1024**3:
            raise ValueError('initializer artifacts exceed1GiB')
        report['status'] = 'QUALIFIED_INITIALIZATION_ONLY'
    except Exception as exc:
        report.update(status='FAILED', error=str(exc))
        raise
    finally:
        (a.output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
