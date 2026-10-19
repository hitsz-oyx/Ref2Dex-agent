"""Reuse the audited PTv3 primitives without changing v4."""
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
for name in ('ptv3-shared-wam-v4', 'ptv3-shared-wam-v3', 'ptv3-shared-wam-v2',
             'ptv3-shared-wam', '3d-data-preparation'):
    path = TASK.parent/name/'src'
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from ptv3_shared_wam.paths import add_dependencies
add_dependencies()
