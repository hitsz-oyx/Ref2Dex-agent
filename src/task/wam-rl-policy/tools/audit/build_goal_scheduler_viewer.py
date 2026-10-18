"""Build a compact 3-D review from existing replay arrays; no model or physics."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

TASK = Path(__file__).resolve().parents[2]


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.run/'replay.json').read_text())
    lookup = {r['object_code']: r for r in json.loads((args.cache/'manifest.json').read_text())['records']}
    packed = []
    for case in cases:
        identity = case['metadata'].get('identity')
        if identity:
            with np.load(lookup[identity['object_code']]['geometry']) as geometry:
                if (str(geometry['object_code'].item()) != identity['object_code']
                        or abs(float(geometry['scale'].item())-identity['scale']) > 1e-7):
                    raise ValueError('display CAD identity mismatch')
                keep = np.linalg.norm(geometry['normals'], axis=-1) > 1e-6
                cloud, normals = geometry['surface'][keep], geometry['normals'][keep]
        else:
            cloud = np.array(case['points'])
            normals = cloud/np.maximum(np.linalg.norm(cloud, axis=-1, keepdims=True), 1e-9)
        rows = []
        for r in case['rows']:
            rows.append(dict(t=r['time_s'], pose=r['object_pose'][:3], goal=r['goal_pose'][:3],
                legacy=r.get('legacy_goal_pose', r['clock_goal_pose'])[:3], hand=r['hand'],
                progress=r['confirmed_progress'], ci=r['confirmed_index'], pi=r['projected_index'],
                gi=r['target_reference_index'], mode=r['mode'], reason=r['reason'],
                onref=r['target_on_reference'], ready=r['execution_ready'], interaction=r['interaction'],
                gap=r['goal_point_m'], angle=r['goal_rotation_rad'], legacygap=r.get('legacy_goal_point_m'),
                jump=r['relative_goal_step_m'], legacyjump=r.get('legacy_relative_goal_step_m'),
                witness=r['arrival_witness']))
        packed.append(dict(name=case['name'], metadata=case['metadata'], summary=case['summary'],
            reference=[p[:3] for p in case['reference']], milestones=case['milestones'],
            surface=cloud.tolist(), normals=normals.tolist(), rows=rows))
    default = next((i for i, c in enumerate(packed) if c['metadata']['kind']=='recorded_geometry_replay'
        and c['metadata']['variant']=='delayed_half_speed'
        and any(r['interaction']['state']=='COUPLED' for r in c['rows'])),
        next(i for i, c in enumerate(packed) if c['metadata']['kind']=='recorded_geometry_replay'))
    chosen = packed[default]
    initial = next((i for i, r in enumerate(chosen['rows']) if r['interaction']['state']=='COUPLED'), len(chosen['rows'])//2)
    payload = rounded(dict(cases=packed, default_case=default, default_frame=min(initial+8,len(chosen['rows'])-1)))
    output = args.run/'viewer-data.js'
    if output.exists():
        raise FileExistsError(output)
    output.write_text('window.GOAL_REPLAY='+json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<','\\u003c')+';\n')
    (args.run/'review-3d.html').write_text((TASK/'tools/audit/goal_scheduler_3d_viewer.html').read_text())
    report = dict(cases=len(packed), source='replay.json', display_rounding_decimals=6,
        hand_layout='palm, five bases, five tips; measured landmarks, no invented joint mesh',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_bytes=output.stat().st_size, default_case=chosen['name'], default_frame=payload['default_frame'])
    (args.run/'viewer-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
