"""Keep the full-census accelerator identical to the rigid scalar contract."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from test_rigid_dataset import bundle

spec=importlib.util.spec_from_file_location('full_prepare',Path(__file__).parents[1]/'tools/audit/prepare_full_rigid_data.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)


def test_vector_gate_agrees_on_masked_unknown_and_corrupt_frames():
    rng=np.random.default_rng(170)
    for mutation in range(12):
        seq,geo,contact,event,evidence=bundle()
        if mutation<4:
            seq['joint_valid'][rng.integers(0,32,5)]=False
            seq['object_pose_valid'][rng.integers(0,32,1)]=False
        elif mutation==4:seq['joints'][12,0,0]=np.nan
        elif mutation==5:contact['part'][16,0,5]=22
        elif mutation==6:contact['contact'][9,0,9]=np.inf
        elif mutation==7:event['event_delta_steps'][5,0]=-1
        elif mutation==8:event['state'][5,0]=5
        elif mutation==9:seq['object_poses'][10,0,0,0]=2
        elif mutation==10:
            contact['valid'][:]=False;event['event_valid'][:]=False;event['state'][:]=-1
        else:evidence['geometry_binding_verified']=False
        ticks,_,_=full.admitted_ticks(seq,geo,contact,event,evidence,0)
        expected=[t for t in range(3,8) if not full.admission_reasons(seq,geo,contact,event,evidence,t,0)]
        assert list(ticks)==expected


def test_fixed_conversion_formula_matches_actual_npy_serialization():
    n,s=32,2
    shapes=[((n,2,21,3),'f4'),((n,2,21),'?'),((n,2,778,3),'f2'),((n,2),'?'),((2,778),'u1'),
            ((n,s,4,4),'f4'),((n,s),'?'),((n,),'?'),((n,2),'?'),((n,s,2048),'f2'),((n,s,2048),'u1'),
            ((n,s),'?'),((n,s),'i1'),((n,s),'u1'),((n,s),'i4'),((n,s),'?'),((n,s),'u2'),((n,s),'u2'),((n,s),'?')]
    import io
    actual=3*full.META_BYTES
    for shape,dtype in shapes:
        f=io.BytesIO();np.save(f,np.zeros(shape,dtype));actual+=f.tell()
    assert actual==full.fixed_conversion_sizes(n,s)


def test_floor_checks_projected_bytes_before_writing(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(full.shutil,'disk_usage',lambda _:SimpleNamespace(free=full.FLOOR+99))
    assert full.guard('.',99)==full.FLOOR+99
    with pytest.raises(InterruptedError,match='30GiB'):full.guard('.',100)
