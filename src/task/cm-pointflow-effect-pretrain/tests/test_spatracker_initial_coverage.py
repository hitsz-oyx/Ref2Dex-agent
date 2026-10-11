import importlib.util
from pathlib import Path

import numpy as np

script=Path(__file__).resolve().parents[1]/'tools/audit/audit_spatracker_initial_coverage.py'
spec=importlib.util.spec_from_file_location('initial_coverage',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_columns_do_not_inflate_initial_object_support():
    coords=np.asarray([[1.,1,1],[1,1,1],[3,3,1],[5,5,1]])
    mask=np.zeros((6,6),bool)
    mask[1,1]=True
    mask[3,3]=True
    value=module.initial_queries(coords,np.eye(4),np.eye(3),mask)
    assert value['original_queries']==4
    assert value['unique_initial_queries']==3
    assert value['object_query_columns']==3
    assert value['unique_initial_object_queries']==2
    assert value['unique_object_query_indices']==[0,2]


def test_outside_query_does_not_clip_to_a_border_object():
    mask=np.ones((4,4),bool)
    value=module.initial_queries(np.asarray([[-1.,0,1],[2,2,1]]),np.eye(4),np.eye(3),mask)
    assert value['projected_outside_image']==1
    assert value['unique_initial_object_queries']==1
