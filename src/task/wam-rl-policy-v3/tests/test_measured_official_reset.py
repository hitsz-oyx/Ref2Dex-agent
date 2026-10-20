import copy
import numpy as np
import pytest
from wam_rl_policy_v3.measured_official_env import check_indexed_reset


def test_indexed_reset_preserves_other_live_episode_and_clock():
 before=dict(episodes=[8,12,3],tick=[91,14,7]);after=dict(episodes=[9,12,3],tick=[0,14,7])
 check_indexed_reset(before,after,[0])
 # A silent physics tick in a live environment corrupts the policy transition.
 wrong=copy.deepcopy(after);wrong['tick'][1]+=1
 with pytest.raises(ValueError):check_indexed_reset(before,wrong,[0])
 wrong=copy.deepcopy(after);wrong['episodes'][0]=8
 with pytest.raises(ValueError):check_indexed_reset(before,wrong,[0])
 wrong=copy.deepcopy(after);wrong['episodes'][1]+=1
 with pytest.raises(ValueError):check_indexed_reset(before,wrong,[0])
