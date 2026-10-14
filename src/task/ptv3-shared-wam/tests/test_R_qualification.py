"""Protect the deferred handoff from resolving the wrong repository."""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/run/qualify_R_after_training.py'
spec = importlib.util.spec_from_file_location('qualify_R', SCRIPT)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


def test_handoff_resolves_real_repository_and_training_output():
    assert (qualify.ROOT / 'docs/CAMPAIGN.md').is_file()
    assert qualify.TASK == SCRIPT.parents[2]


def test_quota_counts_own_and_unknown_processes_and_never_takes_occupied_target():
    memory={0:7000,5:2,7:2};pids={0:[10],5:[],7:[]}
    assert qualify.choose_idle_gpu(memory,pids,{10:222},111)==5
    memory[7]=7000;pids[7]=[11]
    assert qualify.choose_idle_gpu(memory,pids,{10:111,11:111},111) is None
    assert qualify.choose_idle_gpu(memory,pids,{10:222,11:111},111)==5
    assert qualify.choose_idle_gpu(memory,pids,{},111) is None
    memory[5]=2;pids[5]=[12]
    assert qualify.choose_idle_gpu(memory,pids,{10:222,11:222,12:222},111) is None


def test_teacher_calibration_and_gt_clipping_control_conclusion():
    def rows(teacher,held=80,clipping=0):
        return [dict(maximum_held_frames=teacher),dict(maximum_held_frames=held,clipping=clipping)]
    assert qualify.assess({'A_all8':rows(100),'B_repair':rows(100)})==(True,True,'PROMISING')
    assert qualify.assess({'A_all8':rows(44),'B_repair':rows(100)})==(False,False,'UNCLEAR')
    assert qualify.assess({'A_all8':rows(100),'B_repair':rows(100,79)})==(True,False,'UNPROMISING')
    assert qualify.assess({'A_all8':rows(100),'B_repair':rows(100,80,.01)})==(True,False,'UNPROMISING')
