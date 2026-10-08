"""Historical Python blobs are allowed; changed artifacts never are."""
import hashlib
from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from consequence_evaluator.historical_sources import verify_collection_sources


def repository(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-c', 'user.name=Source Test',
            '-c', 'user.email=source-test@example.invalid', *args], cwd=tmp_path, stderr=subprocess.DEVNULL)
    git('init', '-q')
    code = tmp_path / 'collector.py'; code.write_text('VALUE=1\n')
    asset = tmp_path / 'asset.txt'; asset.write_text('original\n')
    old = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (code, asset)}
    git('add', 'collector.py', 'asset.txt'); git('commit', '-qm', 'initial')
    return code, asset, old, git('rev-parse', 'HEAD').decode().strip()


def test_current_execution_and_original_collection_code_keep_distinct_hashes(tmp_path):
    code, _, old, commit = repository(tmp_path)
    code.write_text('VALUE=2\n')
    live, history = verify_collection_sources(old, commit, tmp_path)
    assert history[str(code)]['recorded_sha256'] == old[str(code)]
    assert live[str(code)] != old[str(code)]


@pytest.mark.parametrize('drift', ['asset', 'wrong_blob', 'unknown_commit'])
def test_nonhistorical_drift_is_rejected(tmp_path, drift):
    code, asset, old, commit = repository(tmp_path)
    code.write_text('VALUE=2\n')
    if drift == 'asset': asset.write_text('changed\n')
    if drift == 'wrong_blob': old[str(code)] = '0' * 64
    if drift == 'unknown_commit': commit = '0' * 40
    with pytest.raises(ValueError):
        verify_collection_sources(old, commit, tmp_path)
