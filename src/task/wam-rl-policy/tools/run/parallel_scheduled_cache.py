"""Compatibility entry point; use wam-rl-policy-v2 for new runs."""
from pathlib import Path
import runpy,sys
_target=Path(__file__).resolve().parents[3]/"wam-rl-policy-v2/tools/run/parallel_scheduled_cache.py"
sys.path.insert(0,str(_target.parent))
globals().update(runpy.run_path(str(_target),run_name="__main__" if __name__=="__main__" else __name__))
