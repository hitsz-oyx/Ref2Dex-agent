#!/usr/bin/env python3
"""Validate reference-free object-relative geometry in recorded native observations."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def quat_matrix(q, torch):
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def run(qualification, output):
    import torch
    torch.set_num_threads(2)
    if output.exists():
        raise ValueError('unique output required')
    begin = time.monotonic()
    b = json.loads(qualification.read_text())
    errors = dict(duplicated_body_position=0., object_velocity_frame=0., rotation_orthogonality=0.,
                  frame_invariance=0., repeated_physical_observation=0.)
    rows = 0
    hashes = {str(qualification.resolve()): sha(qualification), str(Path(__file__).resolve()): sha(Path(__file__))}
    keys = [0, 3, 6, 9, 12, 15]
    for name, expected in b['input_sha256'].items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError('source drift')
        hashes[name] = expected
        record = torch.load(path, map_location='cpu', weights_only=False)
        bucket = torch.tensor([int(hashlib.sha256(f'9851/{int(i)}/{int(j)}'.encode()).hexdigest()[:8], 16) % 100
                               for i, j in zip(record['motion_id'], record['start_frame'])])
        selected = bucket < 70
        if not selected.any():
            continue
        obs = record['native_observation'][selected].flatten(0, 1)
        future = record['future_state'][selected]
        pre = torch.cat((record['state'][selected, None], future[:, :-1]), 1).flatten(0, 1)
        if obs.shape[1] != 1442:
            raise ValueError('native observation contract drift')
        body = obs[:, 406:454].reshape(-1, 16, 3)
        obj = obs[:, 646:649]
        x, z = obs[:, 649:652], obs[:, 652:655]
        rot = torch.stack((x, torch.cross(z, x, dim=-1), z), -1)
        rel = body[:, keys] - obj[:, None]
        geometry = torch.einsum('bij,bkj->bki', rot.transpose(-1, -2), rel)
        world_rotation = quat_matrix(pre[:, 39:43], torch)
        obj_vel_from_native = torch.einsum('bij,bj->bi', rot.transpose(-1, -2), obs[:, 655:658])
        obj_vel_from_state = torch.einsum('bij,bj->bi', world_rotation.transpose(-1, -2), pre[:, 43:46])
        # A fixed global frame change must cancel in object-relative geometry.
        angle = .73
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        frame = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        turned = torch.einsum('ij,bkj->bki', frame, rel)
        rot_turned = frame @ rot
        invariant = torch.einsum('bij,bkj->bki', rot_turned.transpose(-1, -2), turned)
        measured = dict(duplicated_body_position=float((obs[:, 15:60] - body[:, 1:].flatten(1)).abs().max()),
                        object_velocity_frame=float((obj_vel_from_native-obj_vel_from_state).abs().max()),
                        rotation_orthogonality=float((rot.transpose(-1, -2) @ rot-torch.eye(3)).abs().max()),
                        frame_invariance=float((invariant-geometry).abs().max()),
                        repeated_physical_observation=float((obs[:, 15:257]-obs[:, 736:978]).abs().max()))
        for key, value in measured.items():
            errors[key] = max(errors[key], value)
        rows += len(pre)
        if time.monotonic()-begin > 600:
            raise TimeoutError('engineering audit budget')
    passed = rows > 0 and max(errors.values()) <= 2e-5
    result = dict(status='COMPLETED', engineering_passed=passed, rows=rows, max_errors=errors,
                  input_sha256=hashes, elapsed_seconds=time.monotonic()-begin,
                  device_reason='CPU observation/frame algebra; no neural computation',
                  definition='native observed base+five fingertip link origins relative to object in object frame; not identified contact points',
                  scope='engineering representation validation; no predictive or control utility claim')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))
    if not passed:
        raise ValueError('geometry extraction failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.qualification, a.output)
