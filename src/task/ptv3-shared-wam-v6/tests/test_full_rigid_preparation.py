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


def test_resume_normalizes_published_metadata_before_receipt(tmp_path,monkeypatch):
    """A kill after atomic stage publish but before padding must not reject data."""
    from types import SimpleNamespace
    from ptv3_shared_wam_v5.cache_contract import metadata_identity,array_identity
    seq,geo,contact,event,evidence=bundle();seq['object_poses']=seq['object_poses'].astype(np.float32)
    seq.update(observed_frame=np.ones(32,bool),expected_hand_presence=np.ones((32,2),bool))
    event.update(enter_count=np.zeros((32,1),np.uint16),leave_count=np.zeros((32,1),np.uint16),raw_valid=contact['valid'].copy())
    out=tmp_path/'run';key='s4-resume';folder=out/'hands/sequences'/key;cf=out/'contacts'/key;ef=out/'events'/key
    for p in [folder,cf,ef,out/'windows',out/'receipts']:p.mkdir(parents=True,exist_ok=True)
    for values,target in [(seq,folder),(contact,cf),(event,ef)]:
        for name,value in values.items():np.save(target/(name+'.npy'),value)
    registry=tmp_path/'registry';registry.mkdir();np.savez(registry/'g.npz',**geo)
    monkeypatch.setattr(full,'REGISTRY',registry);monkeypatch.setattr(full,'_GEOMETRY_ROWS',{'example':dict(path='g.npz')})
    monkeypatch.setattr(full,'_PLAN_SHA','fixture');monkeypatch.setattr(full,'_TOOLS',(None,None,'cpu'));full._GEOS.clear()
    full.write(out/'contract.json',dict(identity='fixture'))
    hm=dict(frames=32,key=key,joint_semantics=evidence['joint_semantics'],part_semantics=evidence['part_semantics'])
    full.write(folder/'meta.json',hm)
    cm=dict(label_semantics=full.LABEL_SEMANTICS,hand_identity=metadata_identity(folder/'meta.json'),geometry_identities=dict(example=array_identity(geo)))
    full.write(cf/'meta.json',cm)
    full.write(ef/'meta.json',dict(contact_meta_identity=metadata_identity(cf/'meta.json')) ,fixed=True)
    row=dict(sequence_id=0,key=key,source_id=4,source_sequence='fixture',split='train',status='CONVERT',conversion_bytes=full.fixed_conversion_sizes(32,1))
    job=dict(key=key,source_id=4,objects=['example'],source_row=dict(source_path=str(tmp_path/'original.npz'),target_slot=0))
    result=full.process_one((str(out),row,job,None,None))
    assert result['status']=='ADMITTED',result
    assert result['admitted_windows']==5
    assert result['conversion_actual_bytes']==row['conversion_bytes']
    assert (folder/'meta.json').stat().st_size==full.META_BYTES
    assert (cf/'meta.json').stat().st_size==full.META_BYTES
    assert metadata_identity(folder/'meta.json')==cm['hand_identity']
    full._GEOS.clear()
