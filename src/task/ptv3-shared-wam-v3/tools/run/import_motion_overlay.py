"""Import new motion-quality metadata while reusing an audited native payload store.

Run on the destination with uploaded native/recipe metadata directories. This
never changes the parent import, copies no payloads, and launches no training.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-metadata',type=Path,required=True)
    p.add_argument('--recipe-metadata',type=Path,required=True)
    p.add_argument('--parent-import-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-git-commit',default='4739a0e2',
        help='Base commit; this importer is explicitly recorded as an uncommitted worktree addition.')
    a=p.parse_args()
    started=time.monotonic()
    parent=a.parent_import_root.resolve();out=a.output.resolve()
    source=a.native_metadata.resolve();recipe_source=a.recipe_metadata.resolve()
    if out.exists() or parent==out or parent in out.parents:
        raise ValueError('fresh independent overlay outside the parent import required')
    if read(parent/'import-progress.json')['status']!='COMPLETED' or read(parent/'portable-reader-audit.json')['status']!='CPU_READER_PASS':
        raise ValueError('parent import must finish its CPU reader audit first')
    meta=read(source/'manifest.json');progress=read(source/'progress.json')
    if meta.get('status')!='COMPLETED' or progress.get('status')!='COMPLETED' or meta.get('training_allowed') is not False:
        raise ValueError('completed immutable engineering motion view required')
    if meta.get('motion_admission',{}).get('schema')!='simulator-impulse-prefix.v1':
        raise ValueError('explicit simulator impulse motion admission required')
    old_native=parent/'native';old_meta=read(old_native/'manifest.json')
    old_recipe=read(parent/'recipe/recipe.json')
    recipe=read(recipe_source/'recipe.json');origin_recipe=copy.deepcopy(recipe)
    stats=read(recipe_source/'horizon-stats.json')
    moments=read(source/'source-moments.json')
    if stats.get('recipe')!=recipe:
        raise ValueError('new horizon statistics must bind the exact new recipe')
    if Path(recipe['native_root'])/'manifest.json'!=Path(moments['input_manifest']):
        raise ValueError('new recipe and native moments origin do not match')
    expected=(meta['window_index_size_bytes'],meta['window_index_mtime_ns'])
    if recipe['native_index_stamp']!=list(expected):
        raise ValueError('new recipe frozen window index differs from the new view')
    native_recipe=recipe['native_engineering_recipe']
    if (native_recipe['data_manifest']!=moments['input_manifest']
            or (native_recipe['window_index_size_bytes'],native_recipe['window_index_mtime_ns'])!=expected):
        raise ValueError('embedded native recipe binding mismatch')
    if recipe['main3_identity']!=old_recipe['main3_identity'] or any(recipe['source_counts'][sid]!=old_recipe['source_counts'][sid] for sid in ('0','1','2')):
        raise ValueError('overlay must retain the exact existing main3 identity and counts')
    if len(meta['sequences'])!=len(old_meta['sequences']):
        raise ValueError('motion overlay must retain original native row identities')
    # Source4 source_sequence strings repeat across replay batches. Match row
    # first, then cross-check sequence/source/group/target/profile/split together.
    identity=('source_sequence','source_id','group_id','objects','target_slot','profile','split')
    for row,previous in zip(meta['sequences'],old_meta['sequences']):
        if any(row.get(k)!=previous.get(k) for k in identity):
            raise ValueError('native row identity changed; positional reuse is unsafe')
        path=(old_native/previous['path']).resolve();stamp=path.stat()
        if ((row['payload_size_bytes'],row['payload_mtime_ns'])!=(previous['payload_size_bytes'],previous['payload_mtime_ns'])
                or (stamp.st_size,stamp.st_mtime_ns)!=(row['payload_size_bytes'],row['payload_mtime_ns'])):
            raise ValueError('overlay payload differs from its audited parent')
        row['path']=str(path)
    geometries={g['key']:g for g in old_meta['objects']}
    if len(geometries)!=len(old_meta['objects']) or set(geometries)!={g['key'] for g in meta['objects']}:
        raise ValueError('overlay must reuse the same unique geometry keys')
    for geometry in meta['objects']:
        path=(old_native/geometries[geometry['key']]['path']).resolve()
        if not path.is_file():raise ValueError('missing parent CAD '+geometry['key'])
        geometry['path']=str(path)
    index_name=Path(meta['window_index'])
    if index_name.is_absolute() or '..' in index_name.parts:
        raise ValueError('relative frozen index path required')
    index=source/index_name
    if index.stat().st_size!=expected[0]:
        raise ValueError('uploaded window index is incomplete')
    native_counts=Counter()
    with np.load(index,allow_pickle=False) as arrays:
        for split in ('train','dev'):
            accepted=arrays['verified_'+split]
            if accepted.ndim!=2 or accepted.shape[1]!=2 or accepted.dtype.kind not in 'iu':
                raise ValueError('invalid frozen window array')
            for row,tick in accepted:
                if not 0<=int(row)<len(meta['sequences']):raise ValueError('index row out of bounds')
                record=meta['sequences'][int(row)]
                if record['split']!=split or record.get('clock_verified') is not True or tick<3:
                    raise ValueError('frozen split/clock/window identity changed')
                if split=='train':native_counts[str(record['source_id'])]+=1
    if (moments.get('training_allowed') is not False or moments.get('split')!='train'
            or moments['windows']!=sum(native_counts.values())
            or dict(native_counts)!=native_recipe['source_counts']
            or any(recipe['source_counts'].get(sid)!=count for sid,count in native_counts.items())
            or any(moments['source_moments'][sid]['windows']!=count for sid,count in native_counts.items())):
        raise ValueError('new source counts, native moments and recipe disagree')
    if shutil.disk_usage(out.parent).free<20*2**30+500_000_000:
        raise ValueError('overlay would breach 20GiB free-space floor')
    out.mkdir();native=out/'native';native.mkdir();recipe_dir=out/'recipe';recipe_dir.mkdir()
    write(out/'import-progress.json',dict(status='COPYING_METADATA',training_allowed=False))
    try:
        # Only metadata/index artifacts may be transported in this directory.
        # A .npz other than these two would be an accidental payload copy.
        npz_names={str(index_name),'excluded-parent-windows.npz'}
        for path in source.iterdir():
            if not path.is_file():continue
            if path.suffix=='.npz' and path.name not in npz_names:
                raise ValueError('unexpected payload in metadata-only overlay')
            if path.suffix not in ('.json','.npz'):continue
            shutil.copy2(path,native/path.name)
        shutil.copy2(source/'manifest.json',native/'origin-manifest.json')
        for path in recipe_source.iterdir():
            if path.is_file() and path.suffix in ('.json','.npz'):
                shutil.copy2(path,recipe_dir/path.name)
        os.utime(native/index_name,ns=(expected[1],expected[1]))
        bound=(native/index_name).stat()
        if (bound.st_size,bound.st_mtime_ns)!=expected:
            raise ValueError('destination cannot preserve frozen index stamp')
        origin_snapshots=meta.get('input_snapshots',[])
        meta.update(input_snapshots=[str(old_native/'manifest.json')],
            portable_origin_input_snapshots=origin_snapshots,
            portable_parent_native=str(old_native),portable_mode='metadata-only-motion-overlay.v1')
        moments.update(portable_origin_input_manifest=moments['input_manifest'],
            input_manifest=str(native/'manifest.json'),parent_moments=str(old_native/'source-moments.json'))
        recipe.update(main3_root=old_recipe['main3_root'],native_root=str(native))
        recipe['native_engineering_recipe']['data_manifest']=str(native/'manifest.json')
        numeric={k:stats[k] for k in ('hand','scene')}
        stats['recipe']=recipe
        write(native/'manifest.json',meta);write(native/'source-moments.json',moments)
        write(recipe_dir/'recipe.json',recipe);write(recipe_dir/'horizon-stats.json',stats)
        write(recipe_dir/'origin-horizon-numeric.json',numeric)
        # Progress and summary are provenance, not admission overrides.
        progress['portable_origin_input_manifest']=progress.get('input_manifest')
        progress['input_manifest']=str(old_native/'manifest.json')
        write(native/'progress.json',progress)
        summary=read(native/'cleaning-summary.json')
        summary.update(portable_origin_parent=summary.get('parent'),parent=str(old_native))
        write(native/'cleaning-summary.json',summary)
        receipt=dict(schema='ref2dex.portable-nine-motion-overlay.v1',status='IMPORTED_PENDING_READER_AUDIT',
            native_root=str(native),main3_root=recipe['main3_root'],recipe_root=str(recipe_dir),
            source_counts=recipe['source_counts'],original_recipe=origin_recipe,
            numeric_statistics_preserved=True,native_payloads_preserved=True,index_stamp_preserved=True,
            parent_import_root=str(parent),payloads_copied=0,geometries_copied=0,
            native_training_allowed=meta['training_allowed'],recipe_training_allowed=recipe.get('training_allowed'),
            remote_training_started=False,execution_gate='User requested screening/cleaning only; no training.',
            code_identity=dict(git_commit=a.source_git_commit,worktree_modified=True,
                worktree_changes=['src/task/ptv3-shared-wam-v3/tools/run/import_motion_overlay.py']),
            source_path_annotations='source_manifest/source_manifest_snapshot/source_path are original provenance annotations, not remote runtime dependencies; only rebound payload/CAD paths are read.',
            elapsed_s=time.monotonic()-started)
        write(out/'import-receipt.json',receipt)
        write(out/'import-progress.json',dict(status='IMPORTED_PENDING_READER_AUDIT',training_allowed=False))
        print(json.dumps({k:v for k,v in receipt.items() if k!='original_recipe'}),flush=True)
    except Exception as error:
        write(out/'import-progress.json',dict(status='FAILED',training_allowed=False,error=repr(error)))
        raise


if __name__=='__main__':main()
