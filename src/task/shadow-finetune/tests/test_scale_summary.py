import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'tools/audit/summarize_scale.py'
spec=importlib.util.spec_from_file_location('shadow_scale_summary',SCRIPT)
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


def report_fixture(root):
    controls=dict(settings=dict(seed=214,updates=1000,batch_size=64,parent='parent'),
                  training_caches=dict(small='small-cache',full='full-cache'),validation_ids=[0,2],native_reset_seed=216)
    (root/'controls.json').write_text(json.dumps(controls))
    for arm,mae,success in [('small',.1,0),('full',.075,2)]:
        train=root/('train-'+arm);train.mkdir()
        m=dict(status='COMPLETED',comparison=controls,data=controls['training_caches'][arm],
               seed=214,updates=1000,batch_size=64,parent='parent',parent_step=80000,config={},stats={},
               parameters=100,git_commit='fixed',code_root='fixed-code')
        (train/'manifest.json').write_text(json.dumps(m));np.save(train/'validation.npy',[0,2])
        np.savez(train/'initial-actor.npz',weight=np.zeros((2,2)))
        (train/'curve.json').write_text(json.dumps([dict(step=1000,action=mae,command_first_mae=mae)]))
        native=root/('native-final-'+arm);native.mkdir()
        roles=dict(teacher=dict(controller_audit=dict(student=False,invalid_envs=[])),
                   student=dict(controller_audit=dict(student=True,invalid_envs=[]),by_split=dict(dev=dict(official_success=success))))
        (native/'comparison.json').write_text(json.dumps(dict(status='COMPLETED',seed=216,roles=roles,initial_state_max_error=dict(q=0))))
        for role in ('teacher','student'):
            folder=native/role;folder.mkdir();(folder/'environment-objects.json').write_text(json.dumps([dict(object_code='test',split='dev')]))
        np.savez(native/'teacher/episode-00.npz',**{key:np.zeros((2,1,2)) for key in summary.FIELDS})


def test_summary_checks_actual_actor_initialization_and_native_pairing(tmp_path):
    report_fixture(tmp_path)
    result=summary.summarize(tmp_path)
    assert result['signal']=='PROMISING' and result['dev_success']==dict(small=0,full=2)
    assert result['initial_actor_exact_equal']
    np.savez(tmp_path/'train-full/initial-actor.npz',weight=np.ones((2,2)))
    with pytest.raises(ValueError,match='initial actor'):summary.summarize(tmp_path)
    np.savez(tmp_path/'train-full/initial-actor.npz',weight=np.zeros((2,2)))
    file=tmp_path/'native-final-full/teacher/episode-00.npz'
    np.savez(file,**{key:np.ones((2,1,2)) for key in summary.FIELDS})
    with pytest.raises(ValueError,match='cross-arm native initial'):summary.summarize(tmp_path)
