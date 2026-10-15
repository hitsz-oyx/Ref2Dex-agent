"""Reuse frozen data and runtime utilities without mutating the original Task."""
from pathlib import Path
import sys
import os

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()
LEGACY = TASK.parent / "ptv3-shared-wam"
sys.path.insert(0, str(LEGACY / "src"))
from ptv3_shared_wam.paths import add_dependencies
add_dependencies()
DEFAULT_DATA = ROOT / "outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree"
SCHEMA = "ref2dex.ptv3-shared-wam.v2"
STATS_SCHEMA = "ref2dex.horizon-displacement-stats.v1"

DEFAULT_ROBOT_SOURCE = ROOT / "outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1"
