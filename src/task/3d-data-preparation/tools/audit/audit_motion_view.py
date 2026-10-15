"""Verify motion exclusions and reconstruct removed TRAIN moments through the reader."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
TASK=Path(__file__).resolve().parents[2]
for name in ('ptv3-shared-wam','3d-data-preparation'):
    sys.path.insert(0,str(TASK.parent/name/'src'))
from pose_mesh_data.clean_data import CleanPoseMeshWindows
from pose_mesh_data.statistics import Moments


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if (root/'motion-reader-audit.json').exists():raise ValueError('fresh audit required')
    meta=read(root/'manifest.json');parent=Path(meta['input_snapshots'][0]).parent
    old=read(parent/'manifest.json');stats=read(root/'source-moments.json');oldstats=read(parent/'source-moments.json')
    before=dict(np.load(parent/'windows.npz',allow_pickle=False));after=dict(np.load(root/'windows.npz',allow_pickle=False))
    removed=dict(np.load(root/'excluded-parent-windows.npz',allow_pickle=False))
    sid=np.array([r['source_id'] for r in meta['sequences']])
    decisions=read(root/'quarantine.json')['sequences'] if (root/'quarantine.json').exists() else meta['motion_admission']['manual_decisions']['rows']
    onsets={r['row']:r['first_suspect_frame'] for r in decisions}
    for key,index in before.items():
        bad=np.array([int(r) in onsets and int(t)+24>=onsets[int(r)] for r,t in index])
        assert np.array_equal(index[bad],removed[key]) and np.array_equal(index[~bad],after[key])
        assert np.array_equal(index[sid[index[:,0]]!=4],after[key][sid[after[key][:,0]]!=4])
    for sid_key in ('5','6','8','9','10'):
        assert stats['source_moments'][sid_key]==oldstats['source_moments'][sid_key]
    reader=CleanPoseMeshWindows(parent,'train',engineering=True)
    lookup={tuple(v):i for i,v in enumerate(reader.index)}
    accum={name:Moments() for name in ('hand','scene')}
    for row,tick in removed['verified_train']:
        sample=reader[lookup[(row,tick)]].model_inputs
        accum['hand'].update(sample['trajectory_target'],sample['trajectory_valid'] & sample['start_hand_valid'][None])
        accum['scene'].update(sample['scene_target'].transpose(1,0,2,3),sample['scene_valid'].transpose(1,0,2))
    maximum_error=0.
    for name in ('hand','scene'):
        recombined=Moments();recombined.combine(stats['source_moments']['4'][name]);recombined.combine(accum[name].result())
        original=oldstats['source_moments']['4'][name]
        for field in ('count','mean','m2'):
            actual=np.asarray(recombined.result()[field]);expected=np.asarray(original[field])
            assert np.allclose(actual,expected,rtol=2e-6,atol=2e-6),(name,field,float(abs(actual-expected).max()))
            maximum_error=max(maximum_error,float(abs(actual-expected).max()))
    recipe=read(a.recipe/'recipe.json');common=read(a.recipe/'horizon-stats.json')
    oldrecipe=Path(read(a.recipe/'manifest.json')['parent_recipe'])
    old9moments=read(oldrecipe/'source-moments.json');new9moments=read(a.recipe/'source-moments.json')
    assert recipe==common['recipe'] and Path(recipe['native_root'])==root
    for s in ('0','1','2'):assert old9moments[s]==new9moments[s]
    for name in ('hand','scene'):
        for field in ('mean','raw_std','std'):
            values=np.asarray(common[name][field]);assert values.shape==(24,3) and np.isfinite(values).all()
        assert np.min(common[name]['std'])>=.001
    kept=after['verified_train'];bad_ticks=kept[kept[:,0]==14909,1].tolist()
    assert bad_ticks==[3,11,19,27,35,43,51],bad_ticks
    counts=Counter(str(sid[r]) for r,t in kept);assert counts=={s:n for s,n in recipe['source_counts'].items() if s not in ('0','1','2')}
    result=dict(status='PASS',removed_train_windows_replayed=len(removed['verified_train']),
        moment_reconstruction_max_abs=maximum_error,other_eight_source_windows_unchanged=True,
        original_three_moment_samples_unchanged=True,known_outlier_retained_ticks=bad_ticks,
        known_outlier_excluded_ticks=removed['verified_train'][removed['verified_train'][:,0]==14909,1].tolist(),
        policy=meta['motion_admission']['policy'],training_launched=False)
    (root/'motion-reader-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
