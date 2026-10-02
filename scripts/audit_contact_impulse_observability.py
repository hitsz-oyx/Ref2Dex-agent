#!/usr/bin/env python3
"""Independent double-precision cal reconstruction and raw scale diagnostics."""
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


def run(result_path, output):
    import torch
    torch.set_num_threads(2)
    begin = time.monotonic()
    if output.exists():
        raise ValueError('unique output required')
    result = json.loads(result_path.read_text())
    deltas, predictions, forces, weights, masses = [], [], [], [], []
    metadata = set()
    for name, expected in result['input_sha256'].items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError('input drift')
        if path.suffix != '.pt':
            continue
        record = torch.load(path, weights_only=False, map_location='cpu')
        mask = torch.tensor([50 <= int(hashlib.sha256(f'9851/{int(i)}/{int(j)}'.encode()).hexdigest()[:8], 16) % 100 < 70
                             for i, j in zip(record['motion_id'], record['start_frame'])])
        if not mask.any():
            continue
        pre = record['state'][mask, -13:].double()
        previous_force = record['initial_object_force'][mask].double()
        mass = record['mass_kg'][mask].double()
        gravity = float(record['gravity_magnitude'])
        for step in range(10):
            post = record['future_state'][mask, step, -13:].double()
            force = record['future_object_force'][mask, step].double()
            actual = post[:, 7:10] - pre[:, 7:10]
            avg = (force + previous_force) * .5
            predicted = avg / mass[:, None] / 30
            predicted[:, 2] -= gravity / 30
            deltas.append(actual); predictions.append(predicted)
            forces.append(force.norm(dim=-1)); weights.append(mass*gravity); masses.append(mass)
            pre = post; previous_force = force
        metadata.add((int(record['contact_collection']), int(record['substeps'])))
    delta, predicted = torch.cat(deltas), torch.cat(predictions)
    mask = delta.norm(dim=-1) > .05
    measured = dict(rows=int(mask.sum()), constant_velocity=float(delta[mask].square().sum(-1).mean().sqrt()),
                    endpoint_average=float((predicted[mask]-delta[mask]).square().sum(-1).mean().sqrt()))
    expected = result['metrics']['cal']['dynamic']['velocity_vector_rmse_m_s']
    error = max(abs(measured[k]-expected[k])/max(1., abs(expected[k])) for k in ['constant_velocity','endpoint_average'])
    def quantiles(value):
        return dict(zip(['p50','p90','p99','max'], [float(v) for v in torch.quantile(value, torch.tensor([.5,.9,.99,1.],dtype=torch.float64))]))
    audit = dict(status='COMPLETED', passed=error < 2e-6 and measured['rows'] == result['metrics']['cal']['dynamic']['rows'],
                 cal_dynamic=measured, relative_reconstruction_error=error,
                 raw_force_N=quantiles(torch.cat(forces)), raw_force_in_weights=quantiles(torch.cat(forces)/torch.cat(weights)),
                 actual_velocity_change_m_s=quantiles(delta.norm(dim=-1)),
                 endpoint_velocity_error_m_s=quantiles((predicted-delta).norm(dim=-1)),
                 mass_kg_range=[float(torch.cat(masses).min()),float(torch.cat(masses).max())],
                 contact_collection_substeps=sorted(metadata), elapsed_seconds=time.monotonic()-begin,
                 result_sha256=sha(result_path), auditor_sha256=sha(Path(__file__)),
                 device_reason='CPU independent sensor/statistical reconstruction; no models',
                 scope='endpoint force is not qualified as control-period average; exact cause remains unknown')
    output.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    if not audit['passed']:
        raise ValueError('independent reconstruction failure')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.result,a.output)
