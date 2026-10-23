"""The disk emergency must publish a durable receipt before signaling writers."""
import importlib.util
import json
from pathlib import Path
import signal

spec=importlib.util.spec_from_file_location('rigid_watch',Path(__file__).parents[1]/'tools/audit/watch_rigid_storage.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)


def test_receipt_and_stop_exist_before_any_writer_signal(tmp_path,monkeypatch):
    calls=[]
    def signal_writer(pid,sig):
        files=list(tmp_path.glob('storage-stop-receipt-*.json'))
        assert len(files)==1
        receipt=json.loads(files[0].read_text())
        assert receipt['resumable'] and not receipt['automatic_restart']
        assert (tmp_path/'STOP').exists()
        assert (tmp_path/'stop-receipt.json').exists()
        assert receipt['free_bytes']==watch.FLOOR-1
        calls.append((pid,sig))
    monkeypatch.setattr(watch.os,'killpg',signal_writer)
    watch.stop(tmp_path,watch.FLOOR-1,[123,456])
    assert calls==[(123,signal.SIGINT),(456,signal.SIGINT)]
