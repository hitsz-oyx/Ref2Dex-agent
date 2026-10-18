"""Compatibility import; implementation lives in wam-rl-policy-v2."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"wam-rl-policy-v2/src"))
from wam_rl_policy_v2 import control_pipeline as _implementation
sys.modules[__name__] = _implementation
