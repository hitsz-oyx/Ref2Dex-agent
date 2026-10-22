"""Actual AF_UNIX boundary and cross-output recipe exclusion, no training."""
import ast
import importlib.util
import multiprocessing as mp
from multiprocessing.connection import Listener
import os
from pathlib import Path
import tempfile
import pytest
TASK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brief38_launcher',TASK/'tools/run/launch_tau_no_goal_no_event.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)

@pytest.fixture
def ipc_state():
    old=dict(mp.current_process()._config);tmp=tempfile.tempdir;env=dict(os.environ)
    yield
    mp.current_process()._config.clear();mp.current_process()._config.update(old);tempfile.tempdir=tmp
    os.environ.clear();os.environ.update(env)

def training_ipc():
    # Execute the production stdlib helper without importing the CUDA model.
    tree=ast.parse((TASK/'tools/run/train_pretrain.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_training_ipc')
    scope=dict(os=os,Path=Path,tempfile=tempfile,multiprocessing=mp,TASK=TASK)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<production IPC helper>','exec'),scope)
    return scope['configure_training_ipc']()

def test_original_deep_absolute_socket_fails(ipc_state,tmp_path):
    deep=tmp_path/('deep-'*24);deep.mkdir();mp.current_process()._config['tempdir']=str(deep)
    with pytest.raises(OSError,match='AF_UNIX path too long'):Listener()

def test_deep_output_converges_to_short_real_listener(ipc_state,tmp_path):
    env=launch.runtime_environment(tmp_path/('deep-'*24))
    os.environ.update(env);socket_dir=training_ipc()
    assert len(os.fsencode(str(socket_dir)+'/listener-12345678'))<108
    assert 'v6i-' in str(socket_dir)
    with Listener() as listener:assert len(os.fsencode(listener.address))<108
    assert launch.ROOT/'tmp' in Path(socket_dir).parents

def test_same_recipe_cannot_run_in_different_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'active_recipe_trainings',lambda recipe:[])
    with launch.recipe_guard('a'*64,tmp_path/'first',tmp_path/'locks'):
        with pytest.raises(RuntimeError,match='recipe already locked'):
            with launch.recipe_guard('a'*64,tmp_path/'second',tmp_path/'locks'):pass
    with launch.recipe_guard('a'*64,tmp_path/'second',tmp_path/'locks'):pass

def test_existing_legacy_recipe_process_blocks_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'active_recipe_trainings',lambda recipe:[12345])
    with pytest.raises(RuntimeError,match='already running'):
        with launch.recipe_guard('b'*64,tmp_path/'other',tmp_path/'locks'):pass

def test_overdeep_checkout_is_explicit_before_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path/('deep-'*24))
    with pytest.raises(ValueError,match='repository path too deep'):
        launch.runtime_environment(tmp_path/'output')
