"""Descriptive post-hoc localization of the fixed s3 drink failure; no training."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    report = dict(scope='Post-hoc phase localization; proximity is not exact contact or causal attribution', runs={})
    for relative in ('s3_mug_drink_1', 'component-attribution/s3_mug_drink_1'):
        folder = a.run / relative
        meta = json.loads((folder / 'manifest.json').read_text())
        data = np.load(folder / 'trajectory.npz')
        source = np.load(meta['native_input']['reference'])
        ref = source['reference_current']
        if ref.shape[1] != 428:
            raise ValueError('Inspire reference layout differs')
        # Native ref = quaternion4 + xyz3 + hand48 + velocities51 + object13
        # + sixteen body positions48 + global contact1 + sixteen part contacts.
        global_contact = ref[:, 167]
        transitions = np.flatnonzero((global_contact[:-1] > .5) & (global_contact[1:] <= .5)) + 1
        result = dict(reference_contact_off_ticks=transitions.tolist(), roles={})
        for row, role in enumerate(meta['roles']):
            errors = np.linalg.norm(data['obj'][:, row, :3, 3] - data['reference_object'][:, row], axis=-1) * 1000
            above = np.flatnonzero(errors > 200)
            result['roles'][role] = dict(first_error_above_200mm_tick=int(above[0]) if len(above) else None,
                terminal_error_mm=float(errors[-1]), events=[])
            for tick in (145, 150, 155, 160, 165, 180, 210):
                result['roles'][role]['events'].append(dict(tick=tick,
                    reference_contact=float(global_contact[tick]),
                    actual_object_xyz=data['obj'][tick, row, :3, 3].tolist(),
                    reference_object_xyz=data['reference_object'][tick, row].tolist(),
                    actual_wrist_xyz=data['wrist'][tick, row].tolist(),
                    object_error_mm=float(errors[tick]),
                    surface_gap_mm=float(data['surface_gap'][tick, row] * 1000),
                    support_gap_mm=float(data['support_gap'][tick, row] * 1000),
                    table_footprint=bool(data['table_footprint'][tick, row])))
        report['runs'][relative] = result
    output = a.run / 'summary/release-transition.json'
    if output.exists():
        raise ValueError('Preserve existing analysis')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
