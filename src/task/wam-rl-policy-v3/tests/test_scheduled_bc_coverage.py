import importlib.util
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
import train_measured_scheduled_bc as trainer


def test_real_startup_labels_stay_in_training_with_explicit_invalid_zero_tau():
    n=6
    data=dict(measured=np.zeros((n,347),np.float32),task=np.zeros((n,10),np.float32),
        action=np.arange(n*12,dtype=np.float32).reshape(n,12),object_id=np.arange(n,dtype=np.int64)%2,
        points=np.zeros((2,32,3),np.float32))
    valid=np.array([False,False,False,True,True,False])
    tau=np.ones((n,264),np.float32)
    batch=trainer.scheduled_tensors(data,tau,valid,'cpu')
    assert len(batch['action'])==n
    np.testing.assert_array_equal(batch['action'].numpy(),data['action'])
    np.testing.assert_array_equal(batch['valid'].numpy(),valid)
    assert not batch['tau'][~batch['valid']].any()
    assert batch['tau'][batch['valid']].all()


def test_parallel_replay_preserves_serial_rows_and_sequence_order(tmp_path):
    import json
    from test_teacher_goals import build_teacher_fixture
    manifest,*_=build_teacher_fixture(tmp_path)
    data=json.loads(manifest.read_text()); row=data['records'][0]
    data['records']=[dict(row,sequence='sequence_'+str(i)) for i in range(3)]
    manifest.write_text(json.dumps(data))
    serial=trainer.load(manifest,'train',0)
    parallel=trainer.load(manifest,'train',2)
    assert serial['sequences']==parallel['sequences']==['sequence_0','sequence_1','sequence_2']
    for key in ('measured','task','action','object_id','points'):
        np.testing.assert_array_equal(serial[key],parallel[key])
    for a,b in zip(serial['active_masks'],parallel['active_masks']):
        np.testing.assert_array_equal(a,b)
