from pathlib import Path
import os
import sys

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
for task in ('ptv3-shared-wam-v2','video-data-preparation'):
    sys.path.insert(0,str(TASK.parent/task/'src'))
from ptv3_shared_wam_v2 import paths as _v2_paths
SCHEMA='ref2dex.shadow-native-policy.v1'
PARENT=ROOT/'outputs/ptv3-shared-wam-v2/remote-parent-20261015-r1/step80000.pt'
VENDOR=ROOT/'third_party/UniDexGrasp2'
SOURCE=ROOT/'outputs/3d-data-preparation/unidexgrasp-expansion-20261014-r1'
PILOT=ROOT/'outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2'
