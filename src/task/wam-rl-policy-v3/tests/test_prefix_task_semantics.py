"""A safe data boundary never rewrites the declared task or future events."""
import json
from pathlib import Path

import numpy as np
import pytest

from wam_rl_policy_v3 import task_annotations
from wam_rl_policy_v3.task_annotations import task_from_teacher_archive
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
from test_teacher_goals import build_teacher_fixture


def archive():
    n=50;poses=np.repeat(np.eye(4)[None],n,0);poses[:,0,3]=np.arange(n)*.002
    forces=np.zeros((n,5,3));forces[3:40,:2,2]=1
    return dict(obj=poses,body_contact_forces=forces,contact_force_valid=np.arange(n)>0,frame_tick=np.arange(n))


@pytest.mark.parametrize('prefix',[1,10,40,49])
def test_safe_prefix_keeps_full_terminal_endpoint_and_original_event_frames(prefix):
    data=archive();raw=task_from_teacher_archive(data,list(range(5)))
    task,available=task_annotations.task_with_safe_prefix(raw,prefix)
    assert task.packet()==raw.packet() and task.source_length==50
    assert task.terminal_mode.value=='RELEASE' and task.source_frames[-1]==49
    assert [(e.source_frame,e.kind.value) for e in task.events]==[(3,'CONTACT'),(40,'RELEASE')]
    assert available['safe_prefix_commands']==prefix
    assert available['task_censored']==(prefix<49)
    assert available['complete_positive_eligible'] is False
    assert available['event_labels_negative_beyond_prefix'] is False
    assert available['declared_events']==raw.packet()['events']
    # Explicit mutant is the former producer's clean[:prefix+1] re-inference.
    if prefix in (1,10):
        clean={k:v[:prefix+1] for k,v in data.items()}
        wrong=task_from_teacher_archive(clean,list(range(5)))
        assert wrong.packet()!=task.packet()
        assert wrong.terminal_mode.value!=task.terminal_mode.value


def test_reader_accepts_full_task_with_short_prefix_and_keeps_future_declaration(tmp_path):
    manifest,packet,*_=build_teacher_fixture(tmp_path)
    m=json.loads(manifest.read_text());m['records'][0]['training_prefix_commands']=4;manifest.write_text(json.dumps(m))
    side=manifest.parent/m['records'][0]['sidecar'];(side/'raw-task.json').write_text(json.dumps(packet))
    e=TeacherRolloutCorpus(manifest,'train').episode(0)
    assert len(e['action'])==4 and e['task_spec']==packet
    assert e['task_censored'] is True and e['complete_positive_eligible'] is False
    assert e['task_prefix_availability']['declared_endpoint_frame']==8


def test_reader_rejects_legacy_reinferred_prefix_even_with_full_source_length(tmp_path):
    manifest,packet,poses,*_=build_teacher_fixture(tmp_path)
    m=json.loads(manifest.read_text());m['records'][0]['training_prefix_commands']=4;manifest.write_text(json.dumps(m))
    side=manifest.parent/m['records'][0]['sidecar'];(side/'raw-task.json').write_text(json.dumps(packet))
    wrong=TaskSpec.from_reference(poses[:5],source_fps=30,terminal_mode='HOLD',trim_interaction=False).packet()
    wrong['source_length']=packet['source_length'];wrong['source_role']=packet['source_role']
    (side/'task.json').write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='declared|truncated|raw-task'):
        TeacherRolloutCorpus(manifest,'train').episode(0)


def test_beyond_prefix_release_is_censored_not_a_negative_event_label():
    raw=task_from_teacher_archive(archive(),list(range(5)))
    _,available=task_annotations.task_with_safe_prefix(raw,10,unsafe_future=True)
    assert available['declared_events'][-1]==dict(source_frame=40,kind='RELEASE')
    assert available['events_beyond_observed_prefix']==[dict(source_frame=40,kind='RELEASE')]
    assert available['unsafe_future'] is True and available['complete_positive_eligible'] is False
    assert available['terminal_intent_verified'] is False
