import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from video_data_preparation.automatic_masks import target_phrase, choose_box, color_evidence
from video_data_preparation.pipeline import process_window, ProcessingConfig
from video_data_preparation.shards import ShardWriter, freeze_snapshot, recover_publications
from test_pipeline import fixture_episode, Masks, Depth, Tracker


def test_target_language_does_not_select_destination():
    assert target_phrase("Pick up a white lego block from the table and place it in the blue bowl.") == "white lego toy block"
    assert target_phrase("Pick up a yellow mustard bottle and place it in a Tupperware.") == "yellow mustard bottle"
    assert target_phrase("Pick up a yellow rhombus block from the drying rack and place it in a bowl.") == "yellow rhombus block"
    with pytest.raises(ValueError, match="MATERIAL"):
        target_phrase("Pick up the blue scrub and place it in a bowl.")
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        target_phrase("Pick up the cup and the bottle.")
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        target_phrase("Pick up the red sharpie from the orange bowl and place it on the table.")
    with pytest.raises(ValueError, match="MATERIAL"):
        target_phrase("Pick up a clear hand sanitizer bottle from the table and place it on a lid.")


def test_separated_current_detections_fail_closed():
    boxes = np.array([[20, 20, 80, 80], [150, 20, 210, 80]], np.float32)
    uv = np.array([[115, 50]], np.float32)
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        choose_box(boxes, [.5, .49], uv, (360, 640))
    chosen = choose_box(boxes, [.7, .4], uv, (360, 640))
    np.testing.assert_array_equal(chosen["box"], boxes[0])


def test_named_white_rejects_green_but_allows_warm_lighting():
    mask = np.ones((16, 16), bool)
    green = np.broadcast_to(np.array([117, 119, 69], np.uint8), (16, 16, 3))
    with pytest.raises(ValueError, match="COLOR_CONTRADICTION"):
        color_evidence(green, mask, "white lego block")
    white = np.broadcast_to(np.array([210, 186, 158], np.uint8), (16, 16, 3))
    assert color_evidence(white, mask, "white lego block")["contradictory_fraction"] == 0


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
    review = tmp_path/"review.json"
    review.write_text(json.dumps(dict(reviewer="test-visual-review", rows=[dict(shard="a.h5", row=0,
        decision="APPROVED", reason="current target and all four trajectory views checked")])) )
    script = Path(__file__).resolve().parents[1]/"tools/audit/review_candidates.py"
    approved = tmp_path/"approved"
    subprocess.run([sys.executable, str(script), "--corpus", str(tmp_path), "--review", str(review),
                    "--output", str(approved)], check=True, capture_output=True, text=True)
    snapshot = json.loads((approved/"snapshot.json").read_text())
    assert snapshot["counts"] == {"train": 1, "val": 0}
    assert snapshot["shards"][0]["training_allowed"]
    assert not json.loads((tmp_path/"shards/a.json").read_text())["training_allowed"]
