"""The real queue once killed two clients on an ancestor manifest write race."""
import importlib.util
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def load_runner():
    path=Path(__file__).resolve().parents[1]/'tools/run/run_source_ceiling40.py'
    spec=importlib.util.spec_from_file_location('source_ceiling_queue',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_empty_or_truncated_ancestor_is_pending_not_release_or_exception(tmp_path):
    runner=load_runner();(tmp_path/'g6').mkdir()
    (tmp_path/'registration.json').write_text(json.dumps(dict(predecessor=None)))
    path=tmp_path/'g6/manifest.json'
    assert not runner.ancestors_released(tmp_path,6)
    for content in ('','{"status":','{"status":"RUNNING"}'):
        path.write_text(content)
        assert not runner.ancestors_released(tmp_path,6)
    path.write_text('{"status":"COMPLETED"}')
    assert runner.ancestors_released(tmp_path,6)


def test_queue_regression_rejects_fabricating_completed_from_unreadable(tmp_path):
    runner=load_runner();(tmp_path/'g6').mkdir()
    (tmp_path/'registration.json').write_text('{"predecessor":null}')
    (tmp_path/'g6/manifest.json').write_text('')
    def check():assert not runner.ancestors_released(tmp_path,6)
    check()
    with patch.object(runner,'read_manifest',return_value={'status':'COMPLETED'}):
        with pytest.raises(AssertionError):check()


def test_interrupted_atomic_publish_preserves_old_manifest_and_kills_direct_writer(tmp_path):
    path=Path(__file__).resolve().parents[1]/'tools/run/run_source_ceiling_worker.py'
    spec=importlib.util.spec_from_file_location('source_ceiling_worker',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    target=tmp_path/'manifest.json'
    def check():
        target.write_text('{"status":"RUNNING"}')
        with patch.object(Path,'replace',side_effect=OSError('injected publish interruption')):
            try:worker.write(target,dict(status='COMPLETED'))
            except OSError:pass
        assert json.loads(target.read_text())['status']=='RUNNING'
    check()
    with patch.object(worker,'write',lambda p,v:p.write_text(json.dumps(v))):
        with pytest.raises(AssertionError):check()


def test_source_reference_waits_for_motor_admission_and_freezes_while_held():
    path=Path(__file__).resolve().parents[1]/'tools/run/run_source_ceiling_worker.py'
    spec=importlib.util.spec_from_file_location('source_reference_clock',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    ready=dict(unloading=False,finger_bandwidth_limited=False,
        root_bandwidth_or_compliance_limited=False,bandwidth_projection_override=False)
    assert worker.advance_source_reference(7,100,True,ready,False)==8
    assert worker.advance_source_reference(7,100,False,ready,True)==7
    assert worker.advance_source_reference(99,100,True,ready,True)==99
    for reason in ('unloading','finger_bandwidth_limited','bandwidth_projection_override'):
        limited=dict(ready,**{reason:True})
        assert worker.advance_source_reference(7,100,True,limited,True)==7
    root_limited=dict(ready,root_bandwidth_or_compliance_limited=True)
    def check():assert worker.advance_source_reference(7,100,True,root_limited,False)==7
    check()
    assert worker.advance_source_reference(7,100,True,root_limited,True)==8
    with patch.object(worker,'advance_source_reference',lambda cursor,*args:cursor+1):
        with pytest.raises(AssertionError):check()


def test_measured_approach_progress_breaks_precontact_motor_reference_deadlock():
    path=Path(__file__).resolve().parents[1]/'tools/run/run_source_ceiling_worker.py'
    spec=importlib.util.spec_from_file_location('source_approach_clock',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    guard=dict(unloading=False,finger_bandwidth_limited=False,
        root_bandwidth_or_compliance_limited=True,bandwidth_projection_override=False)
    original=worker.advance_source_reference
    def check():
        try:result=worker.advance_source_reference(36,100,True,guard,False,fingertip_gap_m=.04,approach_distance_m=.06)
        except TypeError:result=worker.advance_source_reference(36,100,True,guard,False) # Old production gate ignored approach evidence.
        assert result==37, 'Near the object, deliver the next preshape without inventing contact or grasp success'
    check()
    with patch.object(worker,'advance_source_reference',lambda *args,**kwargs:original(*args)):
        with pytest.raises(AssertionError):check()
