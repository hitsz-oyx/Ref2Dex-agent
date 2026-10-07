"""Reuse the audited native training launcher without editing its shared code."""
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[5]
    launcher = root/'src/task/CmResidual/tools/run_multitrajectory_baseline_probe.py'
    loaded = runpy.run_path(str(launcher))
    loaded['main'].__globals__['BOOTSTRAP'] = Path(__file__).with_name('rebuild_rank_bootstrap.py')
    loaded['main']()


if __name__ == '__main__':
    main()
