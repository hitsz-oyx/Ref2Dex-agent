"""Reuse the existing DExplore teacher collector with same-execution force fields."""
from pathlib import Path
import runpy
import shutil
import sys

ROOT=Path(__file__).resolve().parents[5]

if __name__=='__main__':
    if shutil.disk_usage(ROOT).free<24*2**30:raise RuntimeError('24GiB startup storage reserve')
    for flag in ('--capture-current-observation','--capture-dof-force'):
        if flag not in sys.argv:sys.argv.append(flag)
    runpy.run_path(str(ROOT/'src/task/ptv3-shared-wam/tools/run/collect_rl660_batch.py'),run_name='__main__')
