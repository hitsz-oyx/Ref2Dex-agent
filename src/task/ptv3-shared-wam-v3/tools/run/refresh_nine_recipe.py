"""Rebind an existing nine-source recipe to a derived native quality view.

Reuse the original three-source TRAIN moment sample exactly; replace native
sufficient statistics and recompute all mixture weights and 24-horizon scales.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v3.paths import STATS_SCHEMA, SOURCE_IDS
from ptv3_shared_wam_v3.data import NineWindows
from ptv3_shared_wam_v3.checkpoint import atomic_json, git_commit
from pose_mesh_data.native_recipe import recipe_for_snapshot, source_probabilities
from pose_mesh_data.statistics import mix_moments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-recipe',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();parent=a.parent_recipe.resolve();native=a.native.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('fresh output required')
    old=json.loads((parent/'recipe.json').read_text())
    oldstats=json.loads((parent/'horizon-stats.json').read_text())
    if oldstats['recipe']!=old or not old.get('training_allowed'):
        raise ValueError('matching admitted parent recipe/stats required')
    native_recipe,_=recipe_for_snapshot(native,dict(schema='ref2dex.native-sampling-policy.v1',
        permitted_use='engineering_only',temperature=.5,probability_caps={'8':.01},
        std_floor_m=.001,interpolated_targets='include_when_valid'))
    dataset=NineWindows(Path(old['main3_root']),native,'train')
    if dataset.main.base.meta.get('normalization_source_manifest_sha256')!=old['main3_identity']:
        raise ValueError('original three-source identity changed')
    if any(dataset.counts[s]!=old['source_counts'][s] for s in ('0','1','2')):
        raise ValueError('original three-source counts changed')
    native_meta=json.loads((native/'manifest.json').read_text())
    if native_meta.get('input_snapshots') != [str(Path(old['native_root'])/'manifest.json')]:
        raise ValueError('quality view must derive directly from parent recipe native input')
    weights=source_probabilities(dataset.counts,.5,{'8':.01})
    recipe={**old,'source_counts':dataset.counts,'source_probabilities':weights,
        'native_root':str(native),'native_index_stamp':list(dataset.native._index_stamp),
        'native_engineering_recipe':native_recipe,
        'admission':{**old['admission'],'excludes':old['admission']['excludes']+['UniDex versioned motion suspect tails'],
            'motion_admission':native_meta['motion_admission']}}
    moments=json.loads((parent/'source-moments.json').read_text())
    updated=json.loads((native/'source-moments.json').read_text())['source_moments']
    for sid,value in updated.items():moments[sid]=value
    if set(map(int,moments))!=set(SOURCE_IDS):raise ValueError('nine source moments required')
    combined=mix_moments(moments,weights,.001)
    stats={**oldstats,'schema':STATS_SCHEMA,'source_probabilities':weights,'recipe':recipe,**combined}
    out.mkdir(parents=True)
    shutil.copy2(parent/'main3-train-stat-indices.npz',out/'main3-train-stat-indices.npz')
    atomic_json(out/'recipe.json',recipe);atomic_json(out/'horizon-stats.json',stats)
    atomic_json(out/'source-moments.json',moments)
    atomic_json(out/'manifest.json',dict(status='COMPLETED',git_commit=git_commit(),
        parent_recipe=str(parent),native_root=str(native),reused_main3_moments=True,
        device='CPU: sufficient-statistic subtraction and combination, no neural computation'))
    print(json.dumps(dict(status='COMPLETED',counts=dataset.counts,weights=weights)),flush=True)


if __name__=='__main__':main()
