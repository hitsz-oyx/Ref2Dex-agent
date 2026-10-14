from pathlib import Path
import sys

TASK=Path(__file__).resolve().parents[1]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
sys.path.insert(0,str(TASK.parent/'ptv3-shared-wam/src'))


def pytest_configure(config):
    if config.option.basetemp is None:config.option.basetemp=str(ROOT/'tmp/pytest-3d-data')
