"""Import into a new directory, explicitly rebind locations, preserve numeric data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--main3', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    descriptor = read(a.export/'portable.json')
    if descriptor['status'] != 'EXPORTED' or a.output.exists():
        raise ValueError('complete export and fresh import directory required')
    main3 = a.main3.resolve()
    mainmeta = read(main3/'processed/manifest.json')
    if mainmeta['normalization_source_manifest_sha256'] != descriptor['main3_identity']:
        raise ValueError('existing main3 identity does not match')
    # Preserve the original source-manifest/index hashes rather than inventing identities.
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for part in iter(lambda:f.read(2**20),b''): h.update(part)
        return h.hexdigest()
    for source in mainmeta['sources']:
        if sha(Path(source['root'])/'processed/manifest.json') != source['manifest_sha256']:
            raise ValueError('main3 source manifest changed')
        for split in ('train','val','test'):
            if sha(source['indices'][split]['path']) != source['indices'][split]['sha256']:
                raise ValueError('main3 '+split+' index changed')
    if shutil.disk_usage(a.output.parent).free < descriptor['archive_bytes'] + 20*2**30:
        raise ValueError('import would breach 20GiB free floor')
    a.output.mkdir()
    write(a.output/'import-progress.json', dict(status='EXTRACTING',training_allowed=False))
    with tarfile.open(a.export/'native-and-recipe.tar') as tar:
        # Export archive contains only regular files below native/ or recipe/.
        for number, member in enumerate(tar):
            target = a.output/member.name
            if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():
                raise ValueError('unsafe archive member')
            if not member.name.startswith(('native/','recipe/')):
                raise ValueError('unknown archive member')
            if number % 256 == 0 and shutil.disk_usage(a.output).free < 20*2**30:
                raise ValueError('20GiB free floor reached; partial import preserved')
            tar.extract(member, a.output)
    for entry in descriptor['files']:
        path = a.output/entry['path']
        if path.stat().st_size != entry['size']:
            raise ValueError('incomplete payload')
        # PAX times can be rounded by Python/NAS; restore the exact already recorded identity.
        os.utime(path, ns=(entry['mtime_ns'],entry['mtime_ns']))
        if path.stat().st_mtime_ns != entry['mtime_ns']:
            raise ValueError('filesystem cannot preserve nanosecond payload stamp')
    native = (a.output/'native').resolve()
    meta = read(native/'manifest.json')
    index = native/meta['window_index']
    os.utime(index, ns=(meta['window_index_mtime_ns'],meta['window_index_mtime_ns']))
    if (index.stat().st_size,index.stat().st_mtime_ns) != (meta['window_index_size_bytes'],meta['window_index_mtime_ns']):
        raise ValueError('index stamp not preserved')
    moments = read(native/'source-moments.json')
    old_manifest = moments['input_manifest']
    moments['input_manifest'] = str(native/'manifest.json')
    moments['portable_origin_input_manifest'] = old_manifest
    write(native/'source-moments.json', moments)
    recipe_dir = a.output/'recipe'
    recipe = read(recipe_dir/'recipe.json')
    old_recipe = read(recipe_dir/'recipe.json')
    recipe.update(main3_root=str(main3), native_root=str(native))
    recipe['native_engineering_recipe']['data_manifest'] = str(native/'manifest.json')
    stats = read(recipe_dir/'horizon-stats.json')
    if stats['recipe'] != old_recipe:
        raise ValueError('export stats/recipe binding mismatch')
    stats['recipe'] = recipe
    write(recipe_dir/'recipe.json', recipe)
    write(recipe_dir/'horizon-stats.json', stats)
    receipt = dict(schema='ref2dex.portable-nine-import.v1', status='IMPORTED_PENDING_READER_AUDIT',
        native_root=str(native),main3_root=str(main3),recipe_root=str(recipe_dir.resolve()),
        source_counts=recipe['source_counts'],original_recipe=old_recipe,
        numeric_statistics_preserved=True,native_payloads_preserved=True,index_stamp_preserved=True,
        native_training_allowed=meta['training_allowed'], recipe_training_allowed=recipe['training_allowed'],
        source_path_annotations='Original source_manifest/source_path references record provenance only; use native/origin-manifest.json for originals.')
    write(a.output/'import-receipt.json',receipt)
    write(a.output/'import-progress.json',dict(status='IMPORTED_PENDING_READER_AUDIT',training_allowed=False))
    print(json.dumps({k:v for k,v in receipt.items() if k!='original_recipe'}),flush=True)


if __name__ == '__main__':
    main()
