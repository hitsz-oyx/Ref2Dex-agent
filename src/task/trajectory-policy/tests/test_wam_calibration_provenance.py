import hashlib

import pytest

from trajectory_policy.wam_calibration_provenance import authenticate_calibration_inputs


def fixture(tmp_path, monkeypatch):
    task = tmp_path/'src/task/trajectory-policy'
    path = task/'src/trajectory_policy/pd_retargeter.py'
    path.parent.mkdir(parents=True)
    old, new = b'old calibrated code', b'evolved K24-compatible code'
    path.write_bytes(new)
    old_hash, new_hash = [hashlib.sha256(v).hexdigest() for v in (old, new)]
    manifest = dict(git_commit='old-commit', input_sha256={str(path): old_hash})
    monkeypatch.setattr('trajectory_policy.wam_calibration_provenance.subprocess.check_output', lambda *a, **k: old)
    proof = dict(pass_=True, old_source_sha256=old_hash, current_source_sha256=new_hash)
    return task, manifest, proof


def test_evolved_R_requires_runtime_compatibility_bound_to_both_hashes(tmp_path, monkeypatch):
    task, manifest, proof = fixture(tmp_path, monkeypatch)
    updates = authenticate_calibration_inputs(manifest, tmp_path, task, lambda *a: proof)
    assert updates[0]['compatibility'] == proof


@pytest.mark.parametrize('bad', ['failed', 'wrong_old', 'wrong_current'])
def test_missing_or_forged_compatibility_rejected(tmp_path, monkeypatch, bad):
    task, manifest, proof = fixture(tmp_path, monkeypatch)
    if bad == 'failed':
        proof['pass_'] = False
    else:
        proof['old_source_sha256' if bad == 'wrong_old' else 'current_source_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='compatibility'):
        authenticate_calibration_inputs(manifest, tmp_path, task, lambda *a: proof)


def test_other_calibration_inputs_cannot_drift(tmp_path, monkeypatch):
    task, manifest, proof = fixture(tmp_path, monkeypatch)
    path = tmp_path/'oracle.npz'
    path.write_bytes(b'changed')
    manifest['input_sha256'][str(path)] = hashlib.sha256(b'original').hexdigest()
    with pytest.raises(ValueError, match='native/data/config drift'):
        authenticate_calibration_inputs(manifest, tmp_path, task, lambda *a: proof)
