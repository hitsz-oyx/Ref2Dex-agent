import json
import pickle
import pytest
import numpy as np

from video_data_preparation.job_ledger import JobLedger, stable_split
from video_data_preparation.shards import ShardWriter, ShardDataset, freeze_snapshot, recover_publications
from test_pipeline import built, fixture_episode


def write_shard(root, split, session, filename, tick=3):
    sample, meta, audit = built(fixture_episode(session))
    sample["sample_id"][1] = tick
    meta["tick"] = tick
    meta["episode_key"] = session+"/episode"
    writer = ShardWriter(root/"shards"/filename, split, session, "profile-1")
    writer.append(sample, meta, audit)
    writer.publish()
    return sample


def test_ready_cursor_ignores_incomplete_line_and_test(tmp_path):
    ready = tmp_path/"ready.jsonl"
    row = dict(key="part2/task/0.hdf5", raw_hdf5="/readonly/data/0.hdf5", release_split="official_train")
    ready.write_text(json.dumps(dict(row, release_split="official_test"))+"\n"+json.dumps(row))
    ledger = JobLedger(tmp_path/"ledger.sqlite")
    assert ledger.ingest(ready, "p") == 0
    with ready.open("a") as f:
        f.write("\n")
    assert ledger.ingest(ready, "p") == 1
    assert ledger.ingest(ready, "p") == 0
    job = ledger.claim(lease_seconds=1, now=10)
    takeover = ledger.claim(lease_seconds=1, now=12)
    with pytest.raises(RuntimeError, match="stale worker"):
        ledger.finish(job, "PUBLISHED", {})
    ledger.finish(takeover, "PUBLISHED", {})
    assert ledger.claim(now=20) is None
    ledger.close()


def test_shard_snapshot_read_and_process_local_cache(tmp_path):
    expected = write_shard(tmp_path, "train", "train-session", "train.h5")
    write_shard(tmp_path, "val", "dev-session", "dev.h5")
    snapshot = tmp_path/"snapshot.json"
    meta = freeze_snapshot(tmp_path, snapshot)
    assert meta["counts"] == dict(train=1, val=1)
    dataset = ShardDataset(snapshot, "train", cache_shards=1)
    actual = dataset[0]
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert dataset.audit(0)["metadata"]["tick"] == 3
    clone = pickle.loads(pickle.dumps(dataset))
    assert not clone._handles
    np.testing.assert_array_equal(clone[0]["xyz"], expected["xyz"])
    clone.close()
    dataset.close()
    with pytest.raises(FileExistsError):
        freeze_snapshot(tmp_path, snapshot)


def test_orphan_publication_and_partial_are_safe(tmp_path):
    write_shard(tmp_path, "train", "train-session", "train.h5")
    (tmp_path/"shards/train.json").unlink()  # Reproduce rename-before-receipt crash.
    (tmp_path/"shards/bad.h5.partial").write_bytes(b"incomplete")
    assert recover_publications(tmp_path, "profile-1") == 1
    assert recover_publications(tmp_path, "profile-1") == 0
    assert json.loads((tmp_path/"shards/train.json").read_text())["samples"] == 1
    freeze_snapshot(tmp_path, tmp_path/"audit-snapshot.json", require_both=False)
    assert (tmp_path/"shards/bad.h5.partial").exists()


def test_snapshot_rejects_session_leakage_and_duplicate_windows(tmp_path):
    write_shard(tmp_path, "train", "same", "a.h5")
    write_shard(tmp_path, "val", "same", "b.h5", tick=11)
    with pytest.raises(ValueError, match="session leaked"):
        freeze_snapshot(tmp_path, tmp_path/"leaked.json")
    info = json.loads((tmp_path/"shards/b.json").read_text())
    info["split"] = "train"
    info["ticks"] = [3]
    (tmp_path/"shards/b.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="duplicate native"):
        freeze_snapshot(tmp_path, tmp_path/"duplicates.json", require_both=False)


def test_session_registry_and_stable_split():
    assert stable_split("recording.mov") == stable_split("recording")
    assert stable_split("s", {"s": "val"}) == "val"
    assert stable_split("recording.mov", {"recording": "val"}) == "val"
    with pytest.raises(ValueError):
        stable_split("s", {"s": "test"})
    with pytest.raises(ValueError, match="conflicting"):
        stable_split("s.mov", {"s": "train", "s.mov": "val"})
