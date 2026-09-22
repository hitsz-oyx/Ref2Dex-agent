import hashlib

import pytest

from src.task.CmResidual.scratch_checkpoint import authorize_scratch_restore


def test_verified_scratch_resume_is_authorized(tmp_path):
    checkpoint = tmp_path / "ours.pth"
    checkpoint.write_bytes(b"our random-initialized policy")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    environ = {
        "REF2DEX_SCRATCH_POLICY": "1",
        "REF2DEX_SCRATCH_RESUME_CHECKPOINT": str(checkpoint),
        "REF2DEX_SCRATCH_RESUME_SHA256": digest,
    }
    assert authorize_scratch_restore(str(checkpoint), 1, environ)


@pytest.mark.parametrize("change", ["path", "sha", "missing"])
def test_unverified_scratch_resume_is_rejected(tmp_path, change):
    checkpoint = tmp_path / "ours.pth"
    checkpoint.write_bytes(b"our random-initialized policy")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    allowed = checkpoint
    candidate = checkpoint
    if change == "path":
        candidate = tmp_path / "other.pth"
        candidate.write_bytes(checkpoint.read_bytes())
    elif change == "sha":
        digest = "0" * 64
    elif change == "missing":
        allowed = tmp_path / "missing.pth"
        candidate = allowed
    environ = {
        "REF2DEX_SCRATCH_POLICY": "1",
        "REF2DEX_SCRATCH_RESUME_CHECKPOINT": str(allowed),
        "REF2DEX_SCRATCH_RESUME_SHA256": digest,
    }
    with pytest.raises(ValueError, match="unverified"):
        authorize_scratch_restore(str(candidate), 1, environ)


def test_scratch_run_without_restore_remains_random_initialization():
    assert not authorize_scratch_restore("Base", 0, {"REF2DEX_SCRATCH_POLICY": "1"})
