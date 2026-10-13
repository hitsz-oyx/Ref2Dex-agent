"""Small atomic metadata writes and confined artifact paths."""
import json
import os
from pathlib import Path
import uuid


def atomic_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with part.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(part, path)  # Atomic no-clobber publication on the same filesystem.
        else:
            os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def confined_path(root, name):
    root = Path(root).resolve()
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be relative and confined")
    result = (root / path).resolve()
    if root not in result.parents:
        raise ValueError("artifact escapes dataset root")
    return result
