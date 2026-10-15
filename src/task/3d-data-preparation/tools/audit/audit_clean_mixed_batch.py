"""Exercise heterogeneous cleaned batches, including provenance and synthetic flags."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parents[2]
for path in (TASK/'src',TASK.parent/'ptv3-shared-wam/src'):
    sys.path.insert(0,str(path))
from pose_mesh_data.clean_data import CleanPoseMeshWindows,clean_collate


def audit(root):
    output=root/'mixed-reader-audit.json'
    if output.exists():
        raise ValueError('fresh mixed audit required')
    selected={}
    for split in ('train','dev'):
        data=CleanPoseMeshWindows(root,split,engineering=True,verified_clock_only=False)
        for i,(row_id,tick) in enumerate(data.index):
            sid=data.meta['sequences'][row_id]['source_id']
            if sid not in selected:
                selected[sid]=data[i]
    windows=[selected[sid] for sid in sorted(selected)]
    for ordered in (windows,list(reversed(windows))):
        batch=clean_collate(ordered)
        if (batch['model_inputs']['source_id'].tolist()!=[w.provenance.source_id for w in ordered]
                or batch['provenance']!=tuple(w.provenance for w in ordered)
                or batch['training_allowed'] is not False):
            raise ValueError('heterogeneous batch identity mismatch')
        for value in batch['model_inputs'].values():
            if hasattr(value,'is_floating_point') and value.is_floating_point() and not value.isfinite().all():
                raise ValueError('nonfinite mixed batch')
        for record in batch['provenance']:
            if record.source_fps==15 and (sum(record.observed_history)!=2 or sum(record.observed_future)!=12):
                raise ValueError('HOI4D observation provenance lost')
    result=dict(status='ENGINEERING_PASS',training_allowed=False,
        scope='Forward/reverse mixed engineering batch across source/split boundaries; not a training batch.',
        sources=sorted(selected),batch_size=len(windows),
        provenance=[asdict(w.provenance) for w in windows],
        shapes={k:list(v.shape) for k,v in batch['model_inputs'].items() if hasattr(v,'shape')})
    with output.open('x') as f:
        json.dump(result,f,indent=2)
    return {k:v for k,v in result.items() if k not in ('provenance','shapes')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.root.resolve()),indent=2),flush=True)
