"""A physically similar state with another condition is not a duplicate input."""
import copy
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/audit'))
from build_full_prepared_catalog import same_input


def test_near_duplicate_requires_matching_goal_validity_and_content():
    a=dict(shape=np.array([.1,.08,.05]),phase=3,trajectory=np.zeros(10),contact=np.zeros(16),
           mask=np.ones(10,bool),goal_valid=False,goal_delta=None)
    b=copy.deepcopy(a)
    assert same_input(a,b)
    b.update(goal_valid=True,goal_delta=np.zeros((32,3)))
    assert not same_input(a,b)
    a.update(goal_valid=True,goal_delta=np.zeros((32,3)))
    assert same_input(a,b)
    b['goal_delta'][0,0]=.003
    assert not same_input(a,b)
    b['goal_delta'][0,0]=0;b['trajectory'][0]=.02
    assert not same_input(a,b)
