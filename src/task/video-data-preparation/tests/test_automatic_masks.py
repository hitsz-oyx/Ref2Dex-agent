import json
import numpy as np
import pytest

from video_data_preparation.automatic_masks import target_phrase, choose_box
from video_data_preparation.pipeline import process_window, ProcessingConfig
from video_data_preparation.shards import ShardWriter, freeze_snapshot, recover_publications
from test_pipeline import fixture_episode, Masks, Depth, Tracker


def test_target_language_does_not_select_destination():
    assert target_phrase("Pick up a white lego block from the table and place it in the blue bowl.") == "white lego toy block"
    assert target_phrase("Pick up a yellow mustard bottle and place it in a Tupperware.") == "yellow mustard bottle"
    with pytest.raises(ValueError, match="MATERIAL"):
        target_phrase("Pick up the blue scrub and place it in a bowl.")
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        target_phrase("Pick up the cup and the bottle.")


def test_separated_current_detections_fail_closed():
    boxes = np.array([[20, 20, 80, 80], [150, 20, 210, 80]], np.float32)
    uv = np.array([[100, 50]], np.float32)
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        choose_box(boxes, [.5, .49], uv, (360, 640))
    chosen = choose_box(boxes, [.7, .4], uv, (360, 640))
    np.testing.assert_array_equal(chosen["box"], boxes[0])


def test_future_hook_and_provisional_shards_never_become_trainable(tmp_path):
    calls = []
    class Provisional(Masks):
        def select_current(self, key, tick, rgb, native):
            calls.append("current")
            assert rgb.shape == (32, 48, 3) and native.frames == 4
            out = super().select_current(key, tick, rgb, native)
            out.update(training_allowed=False, verification="automatic-provisional")
            return out
        def prepare_labels(self, key, ids, rgb):
            calls.append("future")
            assert len(rgb) == 25 and list(ids) == list(range(3, 28))
    native = fixture_episode()
    sample, meta, audit = process_window(native, "ep", 7, 3, np.zeros((28, 32, 48, 3), np.uint8),
        native.intrinsic, Provisional(), Depth(), Tracker(), ProcessingConfig(scene_points=128, erosion=1))
    assert calls == ["current", "future"] and not meta["training_allowed"]
    writer = ShardWriter(tmp_path/"shards/a.h5", "train", native.session, "profile")
    writer.append(sample, meta, audit)
    receipt = writer.publish()
    assert not receipt["training_allowed"]
    with pytest.raises(ValueError, match="unadmitted"):
        freeze_snapshot(tmp_path, tmp_path/"snapshot.json", require_both=False)
    (tmp_path/"shards/a.json").unlink()
    assert recover_publications(tmp_path, "profile") == 1
    assert not json.loads((tmp_path/"shards/a.json").read_text())["training_allowed"]
