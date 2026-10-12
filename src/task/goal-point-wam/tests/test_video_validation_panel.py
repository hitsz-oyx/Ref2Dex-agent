"""One long recording must not crowd all other held sessions off the panel."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'tools/run/video_validation_panel.py'
spec = importlib.util.spec_from_file_location('video_validation_panel',path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def records(session,count):
    return [dict(split='val',session=session,path='%s-%03d'%(session,i),sha256=str(i),source_current_frame=i*15+3) for i in range(count)]


def test_panel_covers_all_sessions_and_full_clock_without_duplicate_short_clips():
    source = records('a-long.mov',80)+records('b-short',2)+records('c',6)+records('d',3)
    panel = module.balanced_video_panel(source)
    assert len(panel) == 13 and len({p['index'] for p in panel}) == 13
    assert {p['session'] for p in panel} == {'a-long','b-short','c','d'}
    assert [p['source_current_frame'] for p in panel if p['session']=='a-long'] == [3,393,783,1188]
    assert module.balanced_video_panel(source[::-1]) == [dict(p,index=len(source)-1-p['index']) for p in panel]


def test_panel_rejects_training_and_does_not_split_session_suffix_aliases():
    source = records('same.mov',2)+records('same',2)
    assert {p['session'] for p in module.balanced_video_panel(source)} == {'same'}
    source[0]['split']='train'
    with pytest.raises(ValueError,match='training'):
        module.balanced_video_panel(source)
