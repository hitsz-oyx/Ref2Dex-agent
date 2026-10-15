from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(ROOT/'src/task/ptv3-shared-wam-v2/tests'))


def pytest_configure(config):
    if config.option.basetemp is None:config.option.basetemp=str(ROOT/'tmp/pytest-shadow-finetune')
