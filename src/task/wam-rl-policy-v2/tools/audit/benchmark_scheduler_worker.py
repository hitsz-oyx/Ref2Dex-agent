"""Bounded native A/B worker; optional frozen baseline scheduler implementation."""
import argparse,importlib.util,runpy,sys
from pathlib import Path
TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
sys.path.insert(0,str(TASK/'tools/run'))
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--baseline-scheduler',type=Path)
a,remaining=p.parse_known_args()
if a.baseline_scheduler:
    from wam_rl_policy_v2.goal_scheduler import OrderedGoalScheduler
    spec=importlib.util.spec_from_file_location('wam_rl_policy_v2._baseline_scheduler',a.baseline_scheduler)
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    OrderedGoalScheduler._bounded=baseline.OrderedGoalScheduler._bounded
sys.argv=[str(TASK/'tools/run/closed_loop_scheduled_shadow.py')]+remaining
runpy.run_path(sys.argv[0],run_name='__main__')
