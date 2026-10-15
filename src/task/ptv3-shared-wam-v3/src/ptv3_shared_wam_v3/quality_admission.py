"""Frozen global-window exclusions over an unchanged nine-source base index."""
from pathlib import Path
import numpy as np


def excluded_indices(recipe,split,total):
    d=recipe.get('quality_admission') if recipe else None
    if not d:return np.empty(0,np.int64)
    if (d.get('schema')!='ref2dex.strict-motion-window-admission.v1'
            or d['base_native_index_stamp']!=recipe['native_index_stamp']
            or d['base_main3_identity']!=recipe['main3_identity']):
        raise ValueError('quality admission base identity mismatch')
    p=Path(d['index_path']);s=p.stat()
    if [s.st_size,s.st_mtime_ns]!=d['index_stamp']:raise ValueError('quality exclusion index changed')
    with np.load(p,allow_pickle=False) as z:indices=z[split].copy()
    if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or
            (len(indices) and (indices[0]<0 or indices[-1]>=total or (np.diff(indices)<=0).any()))):
        raise ValueError('sorted unique in-range quality exclusions required')
    if len(indices)!=d['excluded_windows'][split]:raise ValueError('quality exclusion count mismatch')
    return indices
