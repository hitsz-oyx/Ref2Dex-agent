"""Independent nine-source training ownership; sibling code remains read-only."""
from pathlib import Path
import sys
TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
for name in ('ptv3-shared-wam-v2', 'ptv3-shared-wam', '3d-data-preparation'):
    sys.path.insert(0, str(TASK.parent/name/'src'))
from ptv3_shared_wam.paths import add_dependencies
add_dependencies()
SCHEMA = 'ref2dex.ptv3-shared-wam.v3'
STATS_SCHEMA = 'ref2dex.nine-source-horizon-stats.v1'
SOURCE_IDS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
