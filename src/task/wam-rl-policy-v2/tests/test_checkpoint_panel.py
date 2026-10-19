import importlib.util
import json
from pathlib import Path
import numpy as np


spec=importlib.util.spec_from_file_location('checkpoint_panel',Path(__file__).resolve().parents[1]/'tools/audit/summarize_checkpoint_panel.py')
panel=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel)


def test_cluster_uncertainty_keeps_perfectly_correlated_object_outcomes_together():
    values=np.r_[np.ones(8),-np.ones(8)]
    assert panel.cluster_interval(values,['a']*8+['b']*8,np.random.default_rng(246))==[-100.,100.]
    assert panel.cluster_interval(np.zeros(16),['a']*8+['b']*8,np.random.default_rng(246))==[0.,0.]


def test_equal_observations_do_not_hide_unobserved_simulator_state_difference(tmp_path):
    left=tmp_path/'left';right=tmp_path/'right';left.mkdir();right.mkdir()
    fields=dict(observation=np.zeros((2,565)),pose=np.repeat(np.eye(4)[None],2,0),
        hand=np.zeros((2,11,3)),goal=np.zeros((2,3)),root=np.zeros((8,13)))
    for path in (left,right):
        np.savez(path/'initial-state.npz',**fields)
        (path/'env-objects.json').write_text(json.dumps([dict(object_code='object-a',scale=.1)]*2))
    assert panel.initial_difference(left,right)['matched']
    fields['root'][3,7]=.01;np.savez(right/'initial-state.npz',**fields)
    result=panel.initial_difference(left,right)
    assert not result['matched'] and result['fields']['observation']['equal']
    assert result['fields']['root']['max_abs_difference']==.01
    assert result['matched_envs']==[1]
