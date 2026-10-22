import numpy as np
import pytest
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.terminal_support_contract import terminal_support_compatibility


def geometry(x=0,z=.01,offset=0):
    path=np.repeat(np.eye(4)[None],2,0);path[:,0,3]=x;path[:,2,3]=z+offset
    spec=TaskSpec.from_reference(path,source_fps=30,terminal_mode='RELEASE')
    points=np.array([[-.01,-.01,-.01],[.01,.01,-.01],[0,0,.01]])
    table=np.array([0.,0.,offset,0.,0.,0.,1.]);bounds=np.array([[-.5,-.5,-.01],[.5,.5,0.]])
    return spec,points,table,bounds


@pytest.mark.parametrize('x,z',[(0,-1),(2,.01),(0,.5)])
def test_impossible_release_support_is_disproved_with_tolerance_bound(x,z):
    result=terminal_support_compatibility(*geometry(x,z))
    assert result['geometrically_disproved']
    assert result['gap_sets_disjoint'] or result['footprint_sets_disjoint']


def test_compatible_geometry_is_not_called_qualified_and_is_translation_invariant():
    first=terminal_support_compatibility(*geometry());shifted=terminal_support_compatibility(*geometry(offset=10))
    assert not first['geometrically_disproved'] and not shifted['geometrically_disproved']
    np.testing.assert_allclose(first['goal_gap_m'],shifted['goal_gap_m'],atol=1e-12)
    assert 'certificate' not in first
