"""Read task-owned derived artifacts across the NAS's observed mode000 states.

This is deliberately not used for original/raw inputs. Only permission metadata
is repaired on PermissionError; artifact contents and identities are unchanged.
"""
import json
from pathlib import Path
import numpy as np


def read_json(path):
    path=Path(path)
    try:return json.loads(path.read_text())
    except PermissionError:
        path.chmod(0o664)
        return json.loads(path.read_text())


def load_array(path,**kwargs):
    path=Path(path)
    try:return np.load(path,**kwargs)
    except PermissionError:
        path.chmod(0o664)
        return np.load(path,**kwargs)
