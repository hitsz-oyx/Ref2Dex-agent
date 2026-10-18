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
    parser.add_argument('--output', type=Path, help='Separate display directory; replay inputs stay unchanged')
    args = parser.parse_args()
    destination = args.output or args.run
    destination.mkdir(parents=True, exist_ok=True)
    output = destination/'viewer-data.js'
    if output.exists():
        raise FileExistsError(output)
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
                witness=r['arrival_witness'], witnesses=r['milestone_witnesses'],
                error=r['reference_error_m'], lead=r['lookahead_m'], horizon=r['horizon_s']))
        # Use the scheduler's same-ID query points, rather than denser display CAD,
        # so the position axis reproduces its translation-and-rotation arc metric.
        poses, points = np.asarray(case['reference']), np.asarray(case['points'])
        path_points = np.einsum('tij,nj->tni', poses[:, :3, :3], points)+poses[:, None, :3, 3]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=-1).mean(-1))]
        packed.append(dict(name=case['name'], metadata=case['metadata'], summary=case['summary'],
            reference=[p[:3] for p in case['reference']], milestones=case['milestones'],
            arc=arc.tolist(), config=case['config'],
            surface=cloud.tolist(), normals=normals.tolist(), rows=rows))
    default = next((i for i, c in enumerate(packed) if c['metadata']['kind']=='recorded_geometry_replay'
        and c['metadata']['variant']=='delayed_half_speed'
        and any(r['interaction']['state']=='COUPLED' for r in c['rows'])),
        next(i for i, c in enumerate(packed) if c['metadata']['kind']=='recorded_geometry_replay'))
    chosen = packed[default]
    initial = next((i for i, r in enumerate(chosen['rows']) if r['interaction']['state']=='COUPLED'), len(chosen['rows'])//2)
    payload = rounded(dict(cases=packed, default_case=default, default_frame=min(initial+8,len(chosen['rows'])-1)))
    output.write_text('window.GOAL_REPLAY='+json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<','\\u003c')+';\n')
    template = (TASK/'tools/audit/goal_scheduler_3d_viewer.html').read_text()
    (destination/'review-3d.html').write_text(template)
    (destination/'review.html').write_text(template)
    report = dict(cases=len(packed), source='replay.json', display_rounding_decimals=6,
        hand_layout='palm, five bases, five tips; measured landmarks, no invented joint mesh',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        replay_directory=str(args.run.resolve()),
        position_axis='mean same-ID query-point arc including rotation; not elapsed replay time',
        data_bytes=output.stat().st_size, default_case=chosen['name'], default_frame=payload['default_frame'])
    (destination/'viewer-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
