from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parents[4]
        config.option.basetemp = str(root / "tmp/pytest-ptv3-shared-wam")
