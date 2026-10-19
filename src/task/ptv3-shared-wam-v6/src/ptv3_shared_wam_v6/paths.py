"""Reuse v5 geometry/data primitives from the same frozen code snapshot."""
import os
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
V5_ROOT = Path(os.environ.get('REF2DEX_V6_V5_CODE_ROOT', ROOT)).resolve()
sys.path.insert(0, str(V5_ROOT/'src/task/ptv3-shared-wam-v5/src'))
from ptv3_shared_wam_v5 import paths as base_paths
