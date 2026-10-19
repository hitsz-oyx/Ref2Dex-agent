"""Bind owned immutable cache runs to reconstruction and direct upstream inputs.

No per-batch payload hashing. New sequence metadata records array digests once
at publication; legacy human inheritance is explicit and never covers Shadow.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .storage import read_json


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def metadata_identity(path):
    return identity(read_json(path))


def array_identity(arrays):
    return identity({name:dict(shape=list(value.shape),dtype=str(value.dtype),
        sha256=hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest())
        for name,value in arrays.items()})


def bind_cache(root,kind,inputs):
    root=Path(root);path=root/'cache-contract.json'
    contract=dict(kind=kind,inputs=inputs);contract['identity']=identity(contract)
    if path.exists():
        if read_json(path)!=contract:raise ValueError('cache upstream/protocol changed; use a fresh output directory')
    else:
        if any(root.iterdir()):raise ValueError('unbound existing cache; use a fresh output directory')
        pending=path.with_suffix('.json.part');pending.write_text(json.dumps(contract,indent=2)+'\n')
        pending.replace(path);path.chmod(0o664)
    return contract


def inherited_human(folder,root):
    if not folder.is_symlink() or folder.name.startswith('s4-') or root is None:return False
    return folder.resolve().parent==Path(root).resolve()


def validate_binding(folder,meta,contract,legacy_root=None):
    if inherited_human(folder,legacy_root):return
    if meta.get('cache_contract')!=contract['identity']:
        raise ValueError('sequence cache identity differs from bound upstream run: '+str(folder))
