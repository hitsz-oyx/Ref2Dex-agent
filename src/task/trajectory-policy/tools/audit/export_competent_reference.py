"""Export authorized planned references from the pinned official raw motion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['TMPDIR'] = str(ROOT / 'tmp/trajectory-policy/competent-pd')
    import numpy as np
    import torch
    from trajectory_policy.reference_actor import reference_from_raw
    manifest = json.loads((args.source / 'manifest.json').read_text())
    motions = [Path(path) for path in manifest['input_sha256'] if path.endswith('/interaction_hand_inspire.pt')]
    if len(motions) != 1:
        raise ValueError('one pinned motion required')
    motion = motions[0]
    digest = hashlib.sha256(motion.read_bytes()).hexdigest()
    if digest != manifest['input_sha256'][str(motion)]:
        raise ValueError('reference source drift')
    raw = torch.load(motion, map_location='cpu', weights_only=False).numpy()
    reference = reference_from_raw(raw)
    if len(reference['obj']) != 543:
        raise ValueError('full native reference length required')
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task output required')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'reference.npz', **reference)
    packet = dict(motion=str(motion), motion_sha256=digest, status='COMPLETED', planned_reference_only=True,
        reference_sha256=hashlib.sha256((out / 'reference.npz').read_bytes()).hexdigest(), offsets=[1, 16],
        fields='human16 point positions and orientations; object pose; no future measured state/robot q/contact',
        cpu_reason='pure pinned-file pose conversion, no model inference',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (out / 'manifest.json').write_text(json.dumps(packet, indent=2) + '\n')
    print(json.dumps(packet))


if __name__ == '__main__':
    main()
