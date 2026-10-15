"""Reconcile frozen indices and independently replay sampled metric endpoints."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]
for path in (TASK/'src', TASK.parent/'ptv3-shared-wam/src'):
    sys.path.insert(0, str(path))
from pose_mesh_data.clean_data import CleanPoseMeshWindows, clean_collate


def audit(root, samples_per_source=8):
    root = root.resolve()
    output = root/'reader-audit.json'
    if output.exists() or not 1 <= samples_per_source <= 32:
        raise ValueError('fresh audit and bounded sample count required')
    meta = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'cleaning-summary.json').read_text())
    with np.load(root/'windows.npz', allow_pickle=False) as z:
        actual = {k:len(z[k]) for k in z.files}
    if actual != summary['windows']:
        raise ValueError('summary and actual window index counts disagree')
    for row in meta['objects']:
        with np.load(root/row['path'], allow_pickle=False) as clean, np.load(row['source_path'], allow_pickle=False) as original:
            query = original['query'] if 'query' in original else original['surface'][np.linspace(0,len(original['surface'])-1,32,dtype=int)]
            if not np.array_equal(clean['query'], query):
                raise ValueError('query identity changed during cleaning')
            if not np.allclose(np.linalg.norm(clean['normals'], axis=1), 1, atol=1e-5):
                raise ValueError('nonunit cleaned normals')
    records, visual = [], {}
    for split in ('train', 'dev'):
        data = CleanPoseMeshWindows(root, split, engineering=True, verified_clock_only=False)
        by_source = {}
        for i, (row_id, tick) in enumerate(data.index):
            by_source.setdefault(data.meta['sequences'][row_id]['source_id'], []).append(i)
        for sid, candidates in sorted(by_source.items()):
            choices = np.unique(np.linspace(0, len(candidates)-1, min(samples_per_source,len(candidates)), dtype=int))
            for choice in choices:
                window = data[candidates[int(choice)]]
                sample = window.model_inputs
                row_id, tick = sample['sample_id']
                row = data.meta['sequences'][row_id]
                sequence = data._sequence(int(row_id))
                target = row['target_slot']
                anchor = np.linalg.inv(sequence['object_poses'][tick, target])
                keys = row['objects']
                active = np.flatnonzero(sequence['object_pose_valid'][tick-3:tick+1].all(0))
                slot = list(active).index(target)
                queries = data.geometry[keys[target]]['query'].astype(np.float64)
                # Direct world endpoints, independent of the forecasting label helper.
                world = queries @ sequence['object_poses'][tick+24,target,:3,:3].T + sequence['object_poses'][tick+24,target,:3,3]
                endpoint = world @ anchor[:3,:3].T + anchor[:3,3]
                reconstructed = sample['scene_points'][slot] + sample['scene_target'][slot,-1]
                error = float(abs(endpoint-reconstructed).max())
                if error > 2e-5:
                    raise ValueError('independent endpoint replay mismatch')
                if not sequence['object_pose_valid'][tick-3:tick+25,target].all():
                    raise ValueError('sample has an invalid target horizon')
                for key in ('xyz','features','start_hand','start_hand_history','scene_target','trajectory_target','goal_object_points'):
                    if not np.isfinite(sample[key]).all():
                        raise ValueError('nonfinite cleaned model field: '+key)
                batch = clean_collate([window])
                if batch['provenance'][0] != window.provenance or batch['training_allowed'] is not False:
                    raise ValueError('provenance lost during collate')
                record = dict(**asdict(window.provenance), endpoint_max_abs_m=error)
                records.append(record)
                visual.setdefault((sid,split), sample)
        del data
    result = dict(status='ENGINEERING_PASS', training_allowed=False,
        windows=actual, geometry_checked=len(meta['objects']), sampled_windows=len(records),
        sampled_source_splits=dict(Counter(str(r['source_id'])+':'+r['split'] for r in records)),
        endpoint_max_abs_m=max(r['endpoint_max_abs_m'] for r in records), records=records,
        limits=['Sampled semantic replay, not full window-by-window evaluation or efficacy validation.',
                'No model training or automatic training admission.'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    plot(root, visual)
    return {k:v for k,v in result.items() if k != 'records'}


def plot(root, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sources = sorted({sid for sid,split in samples})
    fig = plt.figure(figsize=(12, 3.4*len(sources)))
    for row_id,sid in enumerate(sources):
        for col,split in enumerate(('train','dev')):
            ax = fig.add_subplot(len(sources),2,2*row_id+col+1,projection='3d')
            sample = samples.get((sid,split))
            ax.set_title('source%d / %s' % (sid,split))
            if sample is None:
                ax.text2D(.1,.5,'No source window in this split',transform=ax.transAxes)
                continue
            ax.scatter(*sample['xyz'].T,s=1,color='gray',alpha=.3)
            ax.scatter(*sample['start_object_points'].T,s=8,color='orange',label='object now')
            ax.scatter(*sample['goal_object_points'].T,s=8,color='red',label='object +24')
            visible = [sample['xyz'],sample['goal_object_points']]
            for side,color in ((0,'blue'),(1,'purple')):
                valid = sample['start_hand_valid'][side]
                hand = sample['start_hand'][side][valid]
                ax.scatter(*hand.T,s=9,color=color)
                visible.append(hand)
                mask = sample['trajectory_valid'][-1,side]
                future = (sample['start_hand'][side]+sample['trajectory_target'][-1,side])[mask]
                ax.scatter(*future.T,s=9,color=color,marker='x')
                visible.append(future)
            values = np.concatenate(visible)
            center = (values.max(0)+values.min(0))/2
            radius = max(float(np.ptp(values,axis=0).max())/2,.08)
            for setter,v in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
                setter(v-radius,v+radius)
            ax.set_box_aspect((1,1,1))
            ax.set_xlabel('x/m'); ax.set_ylabel('y/m'); ax.set_zlabel('z/m')
            ax.legend(fontsize=6)
    fig.suptitle('Cleaned native geometry: current anchor, fixed query identities, masked hands')
    fig.tight_layout()
    fig.savefig(root/'cleaned-geometry-review.png',dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--samples-per-source',type=int,default=8)
    a = p.parse_args()
    print(json.dumps(audit(a.root,a.samples_per_source),indent=2),flush=True)
