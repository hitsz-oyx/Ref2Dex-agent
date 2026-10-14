import importlib.util
import json
import signal
from pathlib import Path
import numpy as np
import pytest

from video_data_preparation.video import VideoFrames
from video_data_preparation.backends import AnnotatedMasks
from video_data_preparation.job_ledger import JobLedger


def test_index_only_cli_never_initializes_models(tmp_path, capsys):
    script = Path(__file__).resolve().parents[1]/"tools/run/prepare_egodex.py"
    spec = importlib.util.spec_from_file_location("prepare_new_egodex", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ready = tmp_path/"ready.jsonl"
    ready.write_text(json.dumps(dict(key="part2/a/0.hdf5", raw_hdf5="/readonly/0.hdf5",
                                    release_split="official_train"))+"\n")
    output = tmp_path/"consumer"
    module.main(["--ready", str(ready), "--output", str(output), "--index-only"])
    assert json.loads(capsys.readouterr().out)["jobs"] == {"READY": 1}
    assert not (output/"run_config.json").exists()
    assert not (output/"jobs.sqlite").exists()
    module.main(["--ready", str(ready), "--output", str(output), "--index-only"])
    assert json.loads(capsys.readouterr().out)["jobs"] == {"READY": 1}


def test_sequential_video_cache_and_clock(tmp_path):
    import cv2
    video = tmp_path/"clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30., (48, 32))
    if not writer.isOpened():
        pytest.skip("test environment lacks mp4v encoder")
    for i in range(40):
        writer.write(np.full((32, 48, 3), i*5, np.uint8))
    writer.release()
    with VideoFrames(video, 40, width=48, height=32) as frames:
        a = frames.get(range(0, 28))
        b = frames.get(range(8, 36))
        np.testing.assert_array_equal(a[8:], b[:20])
        assert len(frames.cache) <= 28
    with pytest.raises(ValueError, match="clock mismatch"):
        VideoFrames(video, 39)


def test_annotation_backend_checks_current_and_future_identity(tmp_path):
    import cv2
    cv2.imwrite(str(tmp_path/"target.png"), np.ones((32, 48), np.uint8)*255)
    cv2.imwrite(str(tmp_path/"hand.png"), np.zeros((32, 48), np.uint8))
    path = tmp_path/"annotations.json"
    path.write_text(json.dumps(dict(episodes={"ep": dict(target_name="drawer", current_verified=True,
        frames={"3": dict(target="target.png", hand="hand.png")})})))
    backend = AnnotatedMasks(path)
    mask = backend.select_current("ep", 3, np.zeros((32, 48, 3), np.uint8), None)
    assert mask["target"].all() and not mask["hand"].any()
    assert backend.label_masks("ep", [3], (32, 48)).all()
    assert backend.label_masks("ep", [3, 4], (32, 48)) is None


def test_indexed_resume_rejects_duplicate_published_windows(tmp_path):
    ledger = JobLedger(tmp_path/"jobs.sqlite")
    assert ledger.ingest(tmp_path/"not-yet-created.jsonl", "p") == 0
    receipt = dict(status="PUBLISHED", episode_keys=["ep"], profile_id="p", path="a.h5", ticks=[3, 11])
    ledger.register_shard(receipt)
    ledger.register_shard(receipt)
    assert ledger.published_ticks("p", "ep") == {3, 11}
    with pytest.raises(ValueError, match="already published"):
        ledger.register_shard(dict(receipt, path="b.h5"))
    ledger.close()


def test_stop_during_initialization_never_reports_queue_drained(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/"tools/run/prepare_egodex.py"
    spec = importlib.util.spec_from_file_location("prepare_stopped_egodex", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handlers = {}
    monkeypatch.setattr(module.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    monkeypatch.setattr(module, "gpu_state", lambda gpu: dict(gpu=gpu, pids=[]))
    def backend(spec, config):
        handlers[signal.SIGTERM](None, None)
        return object()
    monkeypatch.setattr(module, "load_backend", backend)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ready = tmp_path/"ready.jsonl"
    ready.write_text(json.dumps(dict(key="episode", raw_hdf5="/readonly/native.hdf5",
                                    release_split="official_train"))+"\n")
    config = tmp_path/"config.json"
    config.write_text(json.dumps(dict(profile="geometry-only", mask_backend="test:factory",
        depth_backend="test:factory", tracker_backend="test:factory")))
    output = tmp_path/"consumer"
    module.main(["--ready", str(ready), "--config", str(config), "--output", str(output), "--once"])
    progress = json.loads((output/"progress.json").read_text())
    assert progress["status"] == "STOPPED" and progress["jobs"] == {"READY": 1}
