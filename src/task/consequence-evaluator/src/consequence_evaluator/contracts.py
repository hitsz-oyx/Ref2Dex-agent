"""Lightweight contracts: safe before Isaac Gym initializes Torch."""
from pathlib import Path

K = 24
K_EXEC = 8
SCHEMA = 'ref2dex.consequence-evaluator.windows.v1'


def is_within(path, root):
    """Path ownership check compatible with the native Python3.8 environment."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
