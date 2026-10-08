"""Adopted monitoring must never signal recycled or unrelated process IDs."""
import importlib.util
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location('pointworld_monitor',
    Path(__file__).resolve().parents[1] / 'tools/run/adopt_pointworld_monitor.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_process_identity_can_pin_the_current_process():
    identity = monitor.process_identity(os.getpid())
    assert identity['uid'] == os.getuid() and identity['start_ticks'] > 0
    assert monitor.same_process(identity)
    assert not monitor.same_process({**identity, 'start_ticks': identity['start_ticks'] + 1})


def test_save_request_rejects_pid_reuse_and_unrelated_workers(monkeypatch):
    parent = dict(pid=100, start_ticks=1, uid=os.getuid(), command=['torchrun'])
    good = dict(pid=101, start_ticks=2, uid=os.getuid(), command=['trainer'])
    recycled = dict(pid=102, start_ticks=2, uid=os.getuid(), command=['trainer'])
    unrelated = dict(pid=103, start_ticks=2, uid=os.getuid(), command=['other'])
    live = {100: parent, 101: good, 102: {**recycled, 'start_ticks': 3}, 103: unrelated}
    monkeypatch.setattr(monitor, 'process_identity', live.get)
    monkeypatch.setattr(monitor.launcher, 'belongs_to', lambda pid, owner: pid == 101 and owner == 100)
    signals = []
    monkeypatch.setattr(monitor.os, 'kill', lambda pid, sig: signals.append(pid))
    assert monitor.request_save([good, recycled, unrelated], parent) == [101]
    assert signals == [101]
    live[100] = {**parent, 'start_ticks': 9}
    assert monitor.request_save([good], parent) == []
    assert signals == [101]
