"""Recover completed measurements after a summary-only failure, preserving originals."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from collect_competent_pd import summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    original = json.loads((source / 'manifest.json').read_text())
    if (out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents
            or original['status'] != 'FAILED' or original.get('dagger_rollin')
            or 'operands could not be broadcast' not in original.get('error', '')):
        raise ValueError('fresh owned output and known summary-only failure required')
    for path, digest in original['input_sha256'].items():
        if sha(path) == digest:
            continue
        relative = Path(path).relative_to(ROOT)
        pinned = subprocess.check_output(['git', 'show', original['git_commit'] + ':' + str(relative)], cwd=ROOT)
        if hashlib.sha256(pinned).hexdigest() != digest:
            raise ValueError('input cannot be authenticated: ' + path)
    with np.load(source / 'trajectory.npz', allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    if (data['action'].shape != (542, 24, 18)
            or any(len(data[k]) != 543 for k in ('q', 'dq', 'obj', 'hand', 'velocity'))
            or any(not np.isfinite(v).all() for v in data.values())):
        raise ValueError('complete finite captured measurements required')
    with np.load(original['reference'], allow_pickle=False) as f:
        reference = {k: f[k].copy() for k in f.files}
    result = summarize(data, original['roles'], reference, original, True, True)
    out.mkdir()
    (out / 'trajectory.npz').symlink_to(source / 'trajectory.npz')
    (out / 'producer_manifest.json').write_bytes((source / 'manifest.json').read_bytes())
    manifest = dict(original, status='COMPLETED', run_id=out.name,
                    trajectory_sha256=sha(source / 'trajectory.npz'),
                    recovery=dict(source=str(source), original_status='FAILED',
                                  original_error=original['error'],
                                  producer_manifest_sha256=sha(source / 'manifest.json'),
                                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                  tool_sha256=sha(__file__),
                                  summary_tool_sha256=sha(TASK / 'tools/run/collect_competent_pd.py')))
    for name, value in [('manifest', manifest), ('result', result)]:
        (out / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
