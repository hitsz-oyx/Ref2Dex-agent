import numpy as np
import pytest
import torch
from types import SimpleNamespace
from wam_rl_policy_v2.mixed_objects import partition_objects,bind_environment_objects


def test_train_partitions_disjoint_and_measured_identity_binding():
    records=[dict(object_code=str(i),scale=.08,split='train' if i<12 else 'dev') for i in range(20)]
    parts=partition_objects(records,3,4)
    assert len({r['object_code'] for p in parts for r in p})==12
    assert all(r['split']=='train' for p in parts for r in p)
    assert parts==partition_objects(records,3,4)
    rows=parts[0]
    task=SimpleNamespace(object_code_list=[r['object_code'] for r in rows],
        object_id_buf=torch.tensor([3,2,1,0]),object_scale_buf={i:.08 for i in range(4)})
    bound=bind_environment_objects(task,rows)
    assert [r['object_code'] for r in bound]==task.object_code_list[::-1]
    task.object_scale_buf[0]=.1
    with pytest.raises(ValueError,match='identity'):bind_environment_objects(task,rows)
