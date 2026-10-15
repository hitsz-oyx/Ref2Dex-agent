"""Admit nine fixed sources and fit TRAIN-only sampler-matched moments."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v3.paths import STATS_SCHEMA, SOURCE_IDS
from ptv3_shared_wam_v3.data import NineWindows
from ptv3_shared_wam_v3.checkpoint import atomic_json, git_commit
from pose_mesh_data.native_recipe import recipe_for_snapshot, source_probabilities
from pose_mesh_data.statistics import Moments, mix_moments
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main3', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-main-source', type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists() or not 512 <= args.samples_per_main_source <= 16384:
        raise ValueError('fresh output and bounded TRAIN moments sample required')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    # Retain the stricter encoder-view/moment admission checks from data preparation.
    native_recipe, _ = recipe_for_snapshot(args.native, dict(schema='ref2dex.native-sampling-policy.v1',
        permitted_use='engineering_only', temperature=.5, probability_caps={'8':.01},
        std_floor_m=.001, interpolated_targets='include_when_valid'))
    dataset = NineWindows(args.main3, args.native, 'train')
    weights = source_probabilities(dataset.counts, .5, {'8':.01})
    recipe = dict(schema='ref2dex.nine-source-recipe.v1', permitted_use='offline_pretraining_probe',
        training_allowed=True, source_counts=dataset.counts, source_probabilities=weights,
        main3_root=str(args.main3.resolve()), native_root=str(args.native.resolve()),
        native_index_stamp=list(dataset.native._index_stamp),
        main3_identity=dataset.main.base.meta.get('normalization_source_manifest_sha256'),
        native_engineering_recipe=native_recipe,
        sampling=dict(source='sqrt(TRAIN windows), DexYCB <=1%', main3='existing .6/.2/.2 actual motion strata',
            native='uniform frozen verified TRAIN windows'),
        admission=dict(user_request='2026-10-15: first round jointly train all nine sources',
            excludes=['HOcap clock-unverified', 'H2O held-out TEST', 'native observed-grid rejected windows'],
            native_snapshot_unchanged=True, interpolation='HOI4D valid 15->30Hz, preserve observed masks',
            interpretation='training permission for this recipe only; no scientific validation claim'))
    moments = json.loads((args.native/'source-moments.json').read_text())['source_moments']
    selected = {}
    for source in dataset.main.base.sources:
        source.sequence = lru_cache(maxsize=32)(source.sequence)
    for sid in ('0','1','2'):
        rng = np.random.default_rng(np.random.SeedSequence([19,int(sid)]))
        indices = dataset.draw_source(sid, args.samples_per_main_source, rng)
        selected[sid] = indices
        hand, scene = Moments(), Moments()
        for n,index in enumerate(indices):
            sample = dataset[int(index)][0]
            hand.update(sample['trajectory_target'], sample['trajectory_valid'] & sample['start_hand_valid'][None])
            scene.update(sample['scene_target'].transpose(1,0,2,3), sample['scene_valid'].transpose(1,0,2))
            if (n+1)%512 == 0:
                print(json.dumps(dict(source=sid,samples=n+1,elapsed_s=time.monotonic()-started)),flush=True)
        moments[sid] = dict(windows=len(indices), hand=hand.result(), scene=scene.result(),
            estimation='4096 replacement TRAIN draws under existing source motion-strata sampler')
    combined = mix_moments(moments, weights, .001)
    stats = dict(schema=STATS_SCHEMA, split='train', units='m', fps=30, history=4, horizon=24,
        source_ids=list(SOURCE_IDS), source_probabilities=weights,
        recipe=recipe, std_floor_m=.001, **combined,
        estimation='main3 per-source sampled moments; native six exact full TRAIN moments; mixture point-weighted',
        main3_samples_per_source=args.samples_per_main_source, main3_moment_seed=19)
    np.savez(args.output/'main3-train-stat-indices.npz', **selected)
    atomic_json(args.output/'recipe.json',recipe)
    atomic_json(args.output/'horizon-stats.json',stats)
    atomic_json(args.output/'source-moments.json',moments)
    atomic_json(args.output/'manifest.json',dict(status='COMPLETED',git_commit=git_commit(),
        elapsed_s=time.monotonic()-started,device='CPU: label reads and moments, no neural model computation'))
    print(json.dumps(dict(status='COMPLETED',weights=weights,counts=dataset.counts,elapsed_s=time.monotonic()-started)),flush=True)


if __name__ == '__main__':
    main()
