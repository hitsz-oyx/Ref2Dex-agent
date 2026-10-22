"""Reproduce actual resource-sharer listener and spawned tensor handoff, CPU only."""
import importlib.util
import multiprocessing as mp
from multiprocessing.connection import Listener
import os
from pathlib import Path
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader,Dataset
from ptv3_shared_wam_v6.loader_ipc import configure_loader_ipc
TASK=Path(__file__).resolve().parents[1]

class TinyDataset(Dataset):
    def __len__(self):return 2
    def __getitem__(self,i):return torch.arange(6).reshape(2,3)+i

@pytest.fixture
def restore_ipc():
    old=dict(mp.current_process()._config);tmp=tempfile.tempdir;env=os.environ.get('TMPDIR')
    yield
    mp.current_process()._config.clear();mp.current_process()._config.update(old);tempfile.tempdir=tmp
    if env is None:os.environ.pop('TMPDIR',None)
    else:os.environ['TMPDIR']=env

def test_long_absolute_run_root_reproduces_resource_sharer_failure(tmp_path,restore_ipc):
    p=tmp_path/('long-run-name-'*8);p.mkdir();mp.current_process()._config['tempdir']=str(p)
    with pytest.raises(OSError,match='AF_UNIX path too long'):Listener()

def test_production_ipc_supports_actual_listener_and_spawn_tensor_delivery(restore_ipc):
    root=configure_loader_ipc()
    with Listener() as listener:
        assert len(os.fsencode(listener.address))<=107
        assert not os.path.isabs(listener.address)
    loader=DataLoader(TinyDataset(),batch_size=2,num_workers=1,multiprocessing_context='spawn',timeout=15)
    actual=next(iter(loader));assert torch.equal(actual,torch.stack([TinyDataset()[0],TinyDataset()[1]]))

def test_failed_loader_result_is_never_marked_complete_even_with_zero_exit_code():
    s=importlib.util.spec_from_file_location('tau_launcher',TASK/'tools/run/launch_tau_no_goal_no_event.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    assert m.terminal_reason(0,{'status':'FAILED_DATA_LOADER'})=='FAILED_DATA_LOADER_NO_RESTART'
    assert m.terminal_reason(0,{'status':'STOPPED'},'HARD_STOP')=='HARD_STOP'
    assert m.terminal_reason(0,{'status':'COMPLETED'})=='COMPLETED'
