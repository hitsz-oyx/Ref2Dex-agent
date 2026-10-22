import importlib.util
from pathlib import Path

def test_queue_only_admits_idle_allowed_physical_devices():
    path=Path(__file__).resolve().parents[1]/'tools/run/run_grasp_components.py';spec=importlib.util.spec_from_file_location('component_queue',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.idle_gpus('0, 0, 0\n4, 4884, 10\n5, 64, 0\n6, 2, 0\n7, 0, 0')=={5,6}
