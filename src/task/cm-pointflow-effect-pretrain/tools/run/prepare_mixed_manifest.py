"""Freeze four source indices without copying canonical clouds or trajectories."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.multisource import MixedWindows, SOURCE_NAMES, SOURCE_WEIGHTS, sha


def prepare(roots, output, stats):
    if output.exists(): raise FileExistsError('fresh mixed manifest output required')
    try:
        output.resolve().relative_to((ROOT/'outputs/cm-pointflow-effect-pretrain').resolve())
    except ValueError:
        raise ValueError('mixed manifest must stay in owned Task outputs')
    processed = output/'processed'; processed.mkdir(parents=True)
    descriptors = []
    for name, root, weight in zip(SOURCE_NAMES, roots, SOURCE_WEIGHTS):
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
    if norm['split'] != 'train' or norm['input_manifest_sha256'] != descriptors[0]['manifest_sha256']:
        raise ValueError('pretrained normalization is not from this OakInk training corpus')
    meta = dict(schema='pointworld-multisource.wm30.v1', status='COMPLETED', fps=30, history=4, horizon=24,
                sources=descriptors, normalization_stats_sha256=sha(stats),
                normalization_source_manifest_sha256=descriptors[0]['manifest_sha256'],
                sampling='source .5/.2/.2/.1; source-internal .6/.2/.2 renormalized over real strata',
                validation='64 fixed balanced windows/source; no TEST access for tuning',
                category_mapping='OakInk unchanged; native moving1->0/near-static2->1; no program fabrication')
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
    for name in SOURCE_NAMES: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare([getattr(a, name) for name in SOURCE_NAMES], a.output, a.stats)
    print(json.dumps(dict(status='COMPLETED', output=str(a.output))))


if __name__ == '__main__': main()
