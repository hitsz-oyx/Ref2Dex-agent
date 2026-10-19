"""Add local task fields to every existing command; no neural cache rebuild."""
import argparse
from pathlib import Path
import signal
import sys
TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v2.inspire_task_cache import reuse_full_cache

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('fresh task cache required')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('task-cache30min limit')));signal.alarm(1800)
    reuse_full_cache(a.source,a.output)
