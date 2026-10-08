"""Verify recorded collection code against its Git commit, artifacts in place."""
import hashlib
from pathlib import Path
import re
import subprocess

from .data import sha


def verify_collection_sources(recorded, collection_commit, root):
    """Do not mistake a later tracked-code update for original artifact drift.

    Changed Python sources must match a blob in the exact recorded collection
    commit. Changed/untracked/outside-repo data, configs, weights and assets
    have no fallback. Return live hashes to freeze current execution separately
    from the verified historical collection-code hashes.
    """
    root = Path(root).resolve()
    if not re.fullmatch(r'[0-9a-f]{7,40}', collection_commit):
        raise ValueError('concrete recorded collection commit required')
    live, historical = {}, {}
    for name, expected in recorded.items():
        path = Path(name)
        actual = sha(path)
        live[name] = actual
        if actual == expected:
            continue
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            raise ValueError('external collection artifact drift: ' + name)
        if path.suffix != '.py':
            raise ValueError('collection data/config/weight/asset drift: ' + name)
        try:
            content = subprocess.check_output(['git', 'cat-file', 'blob',
                collection_commit + ':' + relative.as_posix()], cwd=root, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise ValueError('changed collection code has no recorded Git blob: ' + name)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('historical collection-code hash mismatch: ' + name)
        historical[name] = dict(collection_commit=collection_commit,
            recorded_sha256=expected, current_execution_sha256=actual,
            verification='exact recorded Git blob')
    return live, historical
