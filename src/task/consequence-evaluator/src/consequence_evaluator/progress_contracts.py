"""Independent ref4 geometry-progress contract; never unlocks residual-plan data."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

SCHEMA = 'ref2dex.grab-progress.v1'
HAND_ORDER = ['right', 'left']
SEMANTICS = ['wrist', 'thumb_mcp', 'index_mcp', 'middle_mcp', 'ring_mcp', 'little_mcp',
             'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'little_tip']
ARMS = ('E0', 'Oracle-E', 'Oracle-EI')
TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    part = path.with_suffix(path.suffix+'.partial')
    part.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    part.replace(path)


class RunGuard:
    def __init__(self, output, seconds, max_gib, gpu=None):
        self.output = Path(output).resolve()
        owned = ROOT/'outputs/consequence-evaluator'
        if owned not in self.output.parents or self.output.exists():
            raise ValueError('fresh task-owned output required')
        if gpu is not None:
            occupied = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if occupied:
                raise RuntimeError('GPU occupied: '+occupied)
        self.output.mkdir(parents=True)
        self.started = time.time()
        self.deadline = self.started+seconds
        self.max_gib = max_gib
        self.inputs = {}
        self.signatures = {}

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('input identity mismatch: '+str(path))
        self.inputs[str(path)] = value
        s = path.stat()
        self.signatures[str(path)] = (s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def check(self, full=False):
        if time.time() >= self.deadline:
            raise TimeoutError('original run deadline')
        if shutil.disk_usage(self.output).free < 20*2**30:
            raise RuntimeError('20GiB reserve')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > self.max_gib*2**30:
            raise RuntimeError('output bound')
        for name, signature in self.signatures.items():
            s = Path(name).stat()
            if ((s.st_size, s.st_mtime_ns, s.st_ctime_ns) != signature
                    or (full and sha(name) != self.inputs[name])):
                raise ValueError('frozen input drift: '+name)


def validate_manifest(meta, allow_weak=False):
    if (meta.get('schema') != SCHEMA or meta.get('status') != 'COMPLETED'
            or meta.get('fps') != 30 or meta.get('history') != 4 or meta.get('horizon') != 24
            or meta.get('hand_order') != HAND_ORDER or meta.get('hand_semantics') != SEMANTICS
            or meta.get('action_semantics') != 'measured_hand_trajectory'
            or meta.get('effect_semantics') != 'current_target_inverse_times_future_target'):
        raise ValueError('GRAB progress contract mismatch')
    if meta.get('annotation_review_status') != 'human_confirmed':
        if not allow_weak or meta.get('probe_training_allowed') is not True or meta.get('training_allowed') is not False:
            raise ValueError('unreviewed weak labels require explicit Probe mode')


def check_execution_mask(executed_steps, horizon=24):
    """Future robot plans supervise only the portion actually executed."""
    import numpy as np
    if not 0 <= executed_steps <= horizon:
        raise ValueError('invalid executed plan length')
    return np.arange(horizon) < executed_steps
