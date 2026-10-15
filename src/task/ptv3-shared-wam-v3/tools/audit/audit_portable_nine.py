"""CPU reader check of every relocated dependency and nine-source sample contract."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.paths import SOURCE_IDS
from ptv3_shared_wam_v3.data import NineWindows, collate
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--import-root', type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    root = a.import_root.resolve()
    recipe = json.loads((root/'recipe/recipe.json').read_text())
    stats = json.loads((root/'recipe/horizon-stats.json').read_text())
    receipt = json.loads((root/'import-receipt.json').read_text())
    if stats['recipe'] != recipe or not receipt['numeric_statistics_preserved']:
        raise ValueError('import recipe/statistics binding mismatch')
    native_root = Path(recipe['native_root'])
    native_meta = json.loads((native_root/'manifest.json').read_text())
    moments = json.loads((native_root/'source-moments.json').read_text())
    if moments['input_manifest'] != str(native_root/'manifest.json'):
        raise ValueError('native moments were not rebound')
    for row in native_meta['sequences']:
        stamp = (native_root/row['path']).stat()
        if (stamp.st_size,stamp.st_mtime_ns) != (row['payload_size_bytes'],row['payload_mtime_ns']):
            raise ValueError('relocated payload stamp mismatch')
    for row in native_meta['objects']:
        if not (native_root/row['path']).is_file():
            raise ValueError('missing relocated CAD geometry')
    train = NineWindows(recipe['main3_root'],recipe['native_root'],'train',recipe)
    dev = NineWindows(recipe['main3_root'],recipe['native_root'],'dev')
    if train.counts != recipe['source_counts'] or set(map(int,train.counts)) != set(SOURCE_IDS):
        raise ValueError('nine frozen source counts changed')
    rng = np.random.default_rng(17)
    checked = {}
    for split,dataset in [('train',train),('dev',dev)]:
        samples=[]
        for sid in sorted(dataset.counts,key=int):
            for index in dataset.draw_source(sid,4,rng):
                item = dataset[int(index)]
                if int(item[0]['source_id']) != int(sid):
                    raise ValueError('source identity mismatch')
                samples.append(item)
        batch = collate(samples)
        tensors=batch['model_inputs']
        for key,value in tensors.items():
            if hasattr(value,'is_floating_point') and value.is_floating_point() and not value.isfinite().all():
                raise ValueError('non-finite tensor '+key)
        checked[split]=dict(windows=dataset.counts,samples=len(samples),
            source_ids=sorted({int(p['source_id']) for p in batch['provenance']}))
    original = json.loads((root/'recipe/origin-horizon-numeric.json').read_text())
    if any(original[k]!=stats[k] for k in ('hand','scene')):
        raise ValueError('numeric stats drift during reader check')
    result=dict(status='CPU_READER_PASS',elapsed_s=time.monotonic()-start,
        dependency_payloads=len(native_meta['sequences']),geometry=len(native_meta['objects']),
        recipe_root=str(root/'recipe'),splits=checked,
        heldout_test_accessed=False,model_or_gpu_used=False,
        excludes=['HOcap unverified clock','H2O official held-out TEST','encoder-rejected windows'],
        limitation='Confirms relocation and CPU input contracts; no remote CUDA/full-training check.')
    (root/'portable-reader-audit.json').write_text(json.dumps(result,indent=2))
    (root/'import-progress.json').write_text(json.dumps(dict(status='COMPLETED',training_allowed=recipe['training_allowed']),indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
