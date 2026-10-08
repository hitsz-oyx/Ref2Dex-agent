"""Capacity diagnostics must distinguish H confounds and episode diversity."""
from pathlib import Path
import runpy

import numpy as np

capacity = runpy.run_path(str(Path(__file__).resolve().parents[1]/
                            'tools/audit/continuous_pair_capacity.py'))['capacity']


def test_capacity_counts_all_episode_pairs_without_promoting_history_confounds():
    good = dict(episode='good',tick=8,event='maintained_hold',split='train',
                task='airplane',expert='airplane_base',motion='s3',phase='hold',
                initial_relative_height=.04,history=np.zeros(6),
                object_pose=np.eye(4),hand_keypoints=np.zeros((11,3)))
    bad = dict(good,episode='bad',event='unrecovered_drop')
    confounded = dict(bad,episode='confounded',history=np.full(6,10.))
    distant = dict(bad,episode='distant',hand_keypoints=np.full((11,3),.03))
    wrong_motion = dict(bad,episode='wrong-motion',motion='s9')
    repeated = [dict(good,tick=i) for i in range(10)] + [dict(bad,tick=i) for i in range(10)]
    result = capacity([*repeated,confounded,distant,wrong_motion])
    assert result['totals']['event'] == dict(train=3,val=0,test=0)
    assert result['totals']['physical'] == dict(train=2,val=0,test=0)
    assert result['totals']['history'] == dict(train=1,val=0,test=0)


def test_capacity_does_not_compare_events_within_an_episode_or_across_splits():
    good = dict(episode='one',tick=8,event='lift_achieved',split='train',
                task='airplane',expert='airplane_base',motion='s3',phase='grasp',
                initial_relative_height=0.,history=np.zeros(6),
                object_pose=np.eye(4),hand_keypoints=np.zeros((11,3)))
    result = capacity([good,dict(good,tick=40,event='grasp_lost'),
                       dict(good,episode='validation',split='val',event='grasp_lost')])
    assert all(count == 0 for groups in result['totals'].values() for count in groups.values())
