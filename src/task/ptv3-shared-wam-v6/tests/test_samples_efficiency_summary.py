import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'tools/audit/summarize_samples_efficiency.py'
spec=importlib.util.spec_from_file_location('efficiency_summary',path)
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


def test_threshold_requires_two_points_and_full_auc_requires_full_budget():
    def row(x,success):return dict(environment_steps=x,task_successes=success,n=10,rows=[])
    points=[row(0,0),row(262144,6),row(524288,4),row(786432,6),row(1048576,7)]
    result=summary.metrics(points)
    assert result['threshold_steps']==786432
    assert np.isclose(result['normalized_auc'],.4875)
    partial=summary.metrics(points[:3])
    assert partial['threshold_steps'] is None
    assert partial['normalized_auc'] is None
    assert partial['threshold_censored_at']==524288


def test_common_states_require_exact_membership_and_coordinates(tmp_path):
    left=tmp_path/'left.npz';right=tmp_path/'right.npz'
    np.savez(left,q=np.array([[1.,2.]]),sequences=np.array(['mug']))
    np.savez(right,q=np.array([[1.,2.]]),sequences=np.array(['mug']))
    assert summary.compare_initials(left,right)
    np.savez(right,q=np.array([[1.,2.001]]),sequences=np.array(['mug']))
    assert not summary.compare_initials(left,right)
