"""Bound AF_UNIX address length independently of artifact/run directory names."""
import multiprocessing
import os
from pathlib import Path
import tempfile


def configure_loader_ipc(pid=None):
    # Spawn preserves cwd and process _config; AF_UNIX accepts relative paths.
    # Use a task-local short address, without /tmp or symlinking other outputs.
    ipc = Path(__file__).resolve().parents[2] / 'tmp' / ('ipc-'+str(pid or os.getpid()))
    ipc.mkdir(parents=True, exist_ok=True, mode=0o700)
    address_root = os.path.relpath(str(ipc), os.getcwd())
    # multiprocessing adds listener-XXXXXXXX (17 bytes including slash).
    if len(os.fsencode(address_root)) + 18 > 107:
        raise ValueError('loader IPC address exceeds AF_UNIX limit; run from repository root')
    os.environ['TMPDIR'] = address_root
    tempfile.tempdir = address_root
    multiprocessing.current_process()._config['tempdir'] = address_root
    return address_root
