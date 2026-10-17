"""Independent v4 package paths and semantic identities."""
from pathlib import Path
import os
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()
for name in ("ptv3-shared-wam-v3", "ptv3-shared-wam-v2", "ptv3-shared-wam", "3d-data-preparation"):
    path = TASK.parent / name / "src"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from ptv3_shared_wam.paths import add_dependencies
add_dependencies()

SCHEMA = "ref2dex.ptv3-shared-wam.v4"
STATS_SCHEMA = "ref2dex.nine-source-horizon-stats.v1"
SOURCE_IDS = (0, 1, 2, 4, 5, 6, 8, 9, 10)
HORIZON = 24
INTERACTION_ENCODING = "object-centric-contact-approach.v1"
