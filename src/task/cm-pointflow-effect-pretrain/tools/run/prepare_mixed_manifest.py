"""Freeze four source indices without copying canonical clouds or trajectories."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.multisource import (MixedWindows, SOURCE_NAMES, SOURCE_WEIGHTS,
                                  MAIN_SOURCE_NAMES, MAIN_SOURCE_WEIGHTS, sha,
                                  normalization_source_identity)


def prepare(roots, output, stats, main_dynamics=False):
    if output.exists(): raise FileExistsError('fresh mixed manifest output required')
    try:
        output.resolve().relative_to((ROOT/'outputs/cm-pointflow-effect-pretrain').resolve())
    except ValueError:
        raise ValueError('mixed manifest must stay in owned Task outputs')
    processed = output/'processed'; processed.mkdir(parents=True)
    descriptors = []
    names = MAIN_SOURCE_NAMES if main_dynamics else SOURCE_NAMES
    weights = MAIN_SOURCE_WEIGHTS if main_dynamics else SOURCE_WEIGHTS
    if len(roots) != len(names): raise ValueError('one root per declared source required')
    for name, root, weight in zip(names, roots, weights):
        root = root.resolve(); manifest = root/'processed/manifest.json'
        meta = json.loads(manifest.read_text())
        if meta['status'] != 'COMPLETED': raise ValueError(name+' source incomplete')
        descriptor = dict(name=name, root=str(root), weight=weight,
                          kind='oakink2' if name == 'oakink2' else 'native',
                          manifest_sha256=sha(manifest), indices={})
        records = meta.get('records', [])
        for split in ('train', 'val', 'test'):
            rows = np.load(root/'processed'/('index_'+split+'.npy'))
            if name != 'oakink2':
                keep = np.array([r.get('source') == name for r in records])
                if len(keep) != len(meta['sequences']): raise ValueError('record/sequence count mismatch')
                rows = rows[keep[rows[:, 0]]]
                # Require the source's deliberate official/subject split. Reject
                # the old pilot whose subject rule puts official ARCTIC val in fit.
                if name in ('grab', 'arctic') and meta.get('split_protocol') != 'GRAB author object holdouts; ARCTIC protocol_p1':
                    raise ValueError('native GRAB/ARCTIC needs official reindexing')
                if any(records[int(i)]['split'] != split for i in np.unique(rows[:, 0])):
                    raise ValueError('source split metadata mismatch')
            target = processed/(name+'_'+split+'.npy'); np.save(target, rows)
            descriptor['indices'][split] = dict(path=str(target.resolve()), sha256=sha(target), windows=len(rows))
        descriptors.append(descriptor)
    norm = json.loads(stats.read_text())
    shared = norm.get('normalization_mode') == 'shared_multisource_train_union'
    descriptor_meta = dict(sources=descriptors)
    expected_norm_source = (normalization_source_identity(descriptor_meta)
                            if shared else descriptors[0]['manifest_sha256'])
    if norm['split'] != 'train' or norm['input_manifest_sha256'] != expected_norm_source:
        raise ValueError('normalization is not bound to the declared training source contract')
    meta = dict(schema='pointworld-multisource.wm30.v1', status='COMPLETED', fps=30, history=4, horizon=24,
                sources=descriptors, normalization_stats_sha256=sha(stats),
                normalization_source_manifest_sha256=expected_norm_source,
                sampling='source '+('/'.join(str(w) for w in weights))+'; source-internal .6/.2/.2 renormalized over real strata',
                validation='64 fixed balanced windows/source; no TEST access for tuning',
                supervision='measured_dynamic_hand_main' if main_dynamics else 'mixed_dynamic_and_rigid_transport',
                excluded_auxiliary_sources=['contactpose'] if main_dynamics else [],
                normalization_role=('forward_input_output_shared_multisource_train_union'
                                    if shared else 'forward_input_output_warmstart'),
                category_mapping='OakInk unchanged; native moving1->0/near-static2->1; no program fabrication')
    if shared:
        meta['normalization_mode'] = 'shared_multisource_train_union'
    path = processed/'manifest.json'; path.write_text(json.dumps(meta, indent=2)+'\n')
    try:
        for split in ('train', 'val'):
            dataset = MixedWindows(output, split)
            # Exercise every source through the actual tensor contract.
            for i, source in enumerate(dataset.sources):
                sample = dataset[int(dataset.offsets[i]+source.groups[0][0])]
                if sample['action'].shape != (24, 22, 9) or not np.isfinite(sample['effect']).all():
                    raise ValueError('source sample contract failure')
    except BaseException as error:
        meta.update(status='FAILED', error=repr(error)); path.write_text(json.dumps(meta, indent=2)+'\n'); raise
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in MAIN_SOURCE_NAMES: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--contactpose', type=Path)
    p.add_argument('--main-dynamics', action='store_true', help='Exclude ContactPose rigid transport from main dynamics supervision')
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.main_dynamics and a.contactpose: p.error('main dynamics excludes ContactPose')
    if not a.main_dynamics and not a.contactpose: p.error('four-source corpus requires --contactpose')
    names = MAIN_SOURCE_NAMES if a.main_dynamics else SOURCE_NAMES
    prepare([getattr(a, name) for name in names], a.output, a.stats, a.main_dynamics)
    print(json.dumps(dict(status='COMPLETED', output=str(a.output))))


if __name__ == '__main__': main()
