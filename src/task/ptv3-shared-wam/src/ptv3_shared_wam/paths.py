"""Explicit repository-local dependencies; no changes to sibling Tasks."""
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def add_dependencies():
    for name in ("cm-pointflow-effect-pretrain", "goal-point-wam", "trajectory-policy"):
        path = str(TASK.parent / name / "src")
        if path not in sys.path:
            sys.path.insert(0, path)
