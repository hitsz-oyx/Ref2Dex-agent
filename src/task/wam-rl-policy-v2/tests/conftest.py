from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp=str(Path(__file__).resolve().parents[4]/'tmp/pytest-wam-rl-policy-v2')
