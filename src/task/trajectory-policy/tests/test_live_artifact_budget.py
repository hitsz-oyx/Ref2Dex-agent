import importlib.util
from pathlib import Path
import subprocess
import pytest

path=Path(__file__).resolve().parents[1]/'tools/run/launch_wam_native_bridge.py'
spec=importlib.util.spec_from_file_location('native_bridge_budget',path)
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


def test_cache_rename_retries_without_accepting_partial_count(monkeypatch):
    responses=iter([subprocess.CompletedProcess([],1,'3 cache','du: cache/tmp: No such file or directory'),
                    subprocess.CompletedProcess([],0,'123 cache\n456 outputs','')])
    monkeypatch.setattr(bridge.subprocess,'run',lambda *a,**k:next(responses))
    assert bridge.directory_bytes('cache','outputs')==579


def test_permission_errors_and_persistent_missing_paths_remain_fatal(monkeypatch):
    calls=[]
    def run(*a,**k):
        calls.append(a);return subprocess.CompletedProcess([],1,'3 cache','du: Permission denied')
    monkeypatch.setattr(bridge.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):bridge.directory_bytes('cache')
    assert len(calls)==1
    calls.clear()
    def missing(*a,**k):
        calls.append(a);return subprocess.CompletedProcess([],1,'3 cache','du: No such file or directory')
    monkeypatch.setattr(bridge.subprocess,'run',missing)
    with pytest.raises(subprocess.CalledProcessError):bridge.directory_bytes('cache')
    assert len(calls)==3
