"""Long-run retention preserves evidence and cannot delete unrelated outputs."""
import pytest

from ptv3_shared_wam.checkpoint import prune_recovery_checkpoints


def test_retention_keeps_milestones_latest_and_unregistered_files(tmp_path):
    registered = ["step1.pt", "step2000.pt", "step20000.pt", "step22000.pt"]
    for name in registered + ["step999.pt", "old-evidence.pt"]:
        (tmp_path / name).write_text("owned fixture")
    config = dict(updates=80000, checkpoint_keep_recent=1, checkpoint_milestone_interval=20000)
    kept = prune_recovery_checkpoints(tmp_path, registered, config)
    assert set(kept) == {"step1.pt", "step20000.pt", "step22000.pt"}
    assert not (tmp_path / "step2000.pt").exists()
    assert (tmp_path / "step999.pt").exists() and (tmp_path / "old-evidence.pt").exists()
    assert prune_recovery_checkpoints(tmp_path, kept, {}) == kept


def test_registry_escape_is_rejected_before_any_deletion(tmp_path):
    (tmp_path / "step2.pt").write_text("preserve")
    config = dict(updates=80000, checkpoint_keep_recent=1, checkpoint_milestone_interval=20000)
    with pytest.raises(ValueError, match="basenames"):
        prune_recovery_checkpoints(tmp_path, ["step2.pt", "../step3.pt"], config)
    assert (tmp_path / "step2.pt").exists()
