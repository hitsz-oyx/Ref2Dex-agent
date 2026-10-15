"""Export the native payload closure and frozen nine-source recipe (main3 reused)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('fresh export directory required')
    root = a.native.resolve()
    meta = json.loads((root/'manifest.json').read_text())
    recipe = json.loads((a.recipe/'recipe.json').read_text())
    if recipe['native_root'] != str(root) or not recipe['training_allowed']:
        raise ValueError('explicit admitted recipe must bind this snapshot')
    stamp = (root/meta['window_index']).stat()
    expected = (meta['window_index_size_bytes'],meta['window_index_mtime_ns'])
    if (stamp.st_size,stamp.st_mtime_ns) != expected or recipe['native_index_stamp'] != list(expected):
        raise ValueError('frozen native window index changed before export')
    if json.loads((a.recipe/'horizon-stats.json').read_text())['recipe'] != recipe:
        raise ValueError('statistics do not bind the exported recipe')
    paths = sorted({str((root/r['path']).resolve()) for r in meta['sequences']+meta['objects']})
    sizes = {path: Path(path).stat().st_size for path in paths}
    needed = sum(sizes.values()) + 500_000_000
    if shutil.disk_usage(a.output.parent).free < needed + 20*2**30:
        raise ValueError('export would breach 20GiB free floor')
    a.output.mkdir()
    mapping = {path: 'native/payloads/%05d.npz'%i for i,path in enumerate(paths)}
    entries = []
    for i,path in enumerate(paths):
        entries.append(dict(origin=path, path=mapping[path], size=sizes[path], mtime_ns=Path(path).stat().st_mtime_ns))
    for row in meta['sequences']:
        actual = (root/row['path']).resolve().stat()
        if (actual.st_size, actual.st_mtime_ns) != (row['payload_size_bytes'], row['payload_mtime_ns']):
            raise ValueError('source payload changed after cleaning: '+row['path'])
        row['path'] = str(Path(mapping[str((root/row['path']).resolve())]).relative_to('native'))
    for row in meta['objects']:
        row['path'] = str(Path(mapping[str((root/row['path']).resolve())]).relative_to('native'))
    descriptor = dict(schema='ref2dex.portable-nine-source.v1', status='EXPORTING',
        origin_native=str(root), origin_recipe=str(a.recipe.resolve()),
        main3_identity=recipe['main3_identity'], source_counts=recipe['source_counts'],
        payload_count=len(entries), payload_bytes=sum(sizes.values()), files=entries,
        provenance_scope='Original provenance paths are recorded origins, not runtime dependencies. Raw datasets are not included.',
        main3_mode='Reuse an already deployed main3 corpus with the identical frozen identity and indices; never rewrite its manifests.')
    (a.output/'portable.json').write_text(json.dumps(descriptor, indent=2))
    started = time.monotonic()
    with tarfile.open(a.output/'native-and-recipe.tar.part', 'w', format=tarfile.PAX_FORMAT) as tar:
        def add_json(name, value):
            import io
            data = json.dumps(value, indent=2).encode()
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add_json('native/manifest.json', meta)
        tar.add(root/'manifest.json', 'native/origin-manifest.json', recursive=False)
        for name in ('windows.npz', 'progress.json', 'source-moments.json', 'excluded-parent-windows.npz', 'cleaning-summary.json'):
            tar.add(root/name, 'native/'+name, recursive=False)
        for path in a.recipe.iterdir():
            if path.is_file():
                tar.add(path, 'recipe/'+path.name, recursive=False)
        for i,entry in enumerate(entries):
            if i % 512 == 0:
                if shutil.disk_usage(a.output).free < 20*2**30:
                    raise ValueError('20GiB free floor reached; partial export preserved')
                if (a.output/'STOP').exists():
                    raise RuntimeError('requested STOP; partial export preserved')
                print(json.dumps(dict(files=i,total=len(entries),elapsed_s=time.monotonic()-started)), flush=True)
            actual = Path(entry['origin']).stat()
            if (actual.st_size,actual.st_mtime_ns) != (entry['size'],entry['mtime_ns']):
                raise ValueError('source file changed during export')
            tar.add(entry['origin'], entry['path'], recursive=False)
    os.rename(a.output/'native-and-recipe.tar.part', a.output/'native-and-recipe.tar')
    descriptor.update(status='EXPORTED', archive_bytes=(a.output/'native-and-recipe.tar').stat().st_size,
        elapsed_s=time.monotonic()-started)
    (a.output/'portable.json').write_text(json.dumps(descriptor, indent=2))
    print(json.dumps({k:v for k,v in descriptor.items() if k!='files'}), flush=True)


if __name__ == '__main__':
    main()
