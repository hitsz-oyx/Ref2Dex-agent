"""Limit accounting catches deliberate over-speed and rejected-target errors."""
import importlib.util
from pathlib import Path


def test_rate_monitor_catches_injected_error_and_distinguishes_boundary():
    p=Path(__file__).resolve().parents[1]/'tools/run/summarize_coverage40_details.py';s=importlib.util.spec_from_file_location('coverage_details',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    pose=dict(linear_step_m=.025,linear_rate_limit_mps=.75,angular_step_rad=.01,angular_rate_limit_radps=2.4)
    good=m.limits([dict(native_target_admitted=True,relative_pose_servo=pose)])
    assert good['linear_rate_boundary']==1 and good['linear_rate_violations']==0
    bad=m.limits([dict(native_target_admitted=False,relative_pose_servo=dict(pose,linear_step_m=.04,angular_step_rad=.10))])
    assert bad['native_target_rejections']==1 and bad['linear_rate_violations']==1 and bad['angular_rate_violations']==1
