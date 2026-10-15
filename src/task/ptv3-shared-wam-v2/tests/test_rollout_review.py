"""Review must preserve recorded frames, link geometry and immutable sources."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('rollout_review_data',
    Path(__file__).resolve().parents[1] / 'tools/audit/rollout_review_data.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_visual_origin_is_applied_in_link_frame(tmp_path):
    urdf = tmp_path / 'test.urdf'
    urdf.write_text('<robot><link name="finger"><visual>'
        '<origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>'
        '<geometry><box size="2 4 6"/></geometry>'
        '</visual></link></robot>')
    mesh = review.visual_meshes(urdf)['finger']
    np.testing.assert_allclose(mesh.bounds, [[-1, 1, 0], [3, 3, 6]], atol=1e-7)
    np.testing.assert_array_equal(review.xyzw_to_wxyz([.1, .2, .3, .9]), [.9, .1, .2, .3])


def test_annotation_interval_split_and_source_are_preserved(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    frozen = root / 'manifest.json'
    frozen.write_text('original source bytes')
    corpus = SimpleNamespace(root=root, fps=30, by_name={'clip': dict(frames=100, split='val')})
    journal = review.ReviewJournal(tmp_path / 'review', corpus)
    first = journal.save('clip', 9, 23, 'unusable', 'early_drop', 'dropped')
    row = json.loads(first.read_text())
    assert (row['start_frame'], row['end_frame_exclusive'], row['split']) == (9, 24, 'val')
    second = journal.save('clip', 24, 99, 'usable', 'none', '')
    assert first != second and first.exists()
    assert frozen.read_text() == 'original source bytes'
    assert journal.count('clip') == 2
    for start, end in [(5, 4), (-1, 4), (0, 100), (.5, 9)]:
        with pytest.raises(ValueError):
            journal.save('clip', start, end, 'uncertain', 'other', '')
    with pytest.raises(ValueError):
        review.ReviewJournal(root / 'review', corpus)
