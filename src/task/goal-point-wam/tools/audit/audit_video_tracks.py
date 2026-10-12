#!/usr/bin/env python3
"""Diagnose teacher drift independently of point-model fit; never select inputs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK.parent/'cm-pointflow-effect-pretrain/src'))
from oakink_wm.pointwam_data import lift_tracks, project_world, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    meta = json.loads((a.data/'manifest.json').read_text())
    records = []
    for record in meta['records']:
        teacher = Path(record['teacher_path'])
        clip = a.data/record['path']
        if sha256(teacher) != record['teacher_sha256'] or sha256(clip) != record['sha256']:
            raise ValueError('data identity drift')
        with np.load(teacher, allow_pickle=False) as f:
            t = {k: f[k].copy() for k in f.files}
        with np.load(clip, allow_pickle=False) as f:
            d = {k: f[k].copy() for k in f.files}
        chosen = d['track_ids']
        uv = t['tracks_uv'][:, chosen]
        world, valid = lift_tracks(uv, t['visibility'][:, chosen], t['depth_m'], t['intrinsic'], t['camera_to_origin'])
        initial = np.broadcast_to(world[0], world.shape)
        projected, _ = project_world(initial[:, None], t['intrinsic'], t['camera_to_origin'])
        residual_px = np.linalg.norm(projected[:, 0]-uv, axis=-1)
        # Future-based strata are AUDIT ONLY, never used in training input sampling.
        candidate = valid.all(0) & np.isfinite(residual_px).all(0) & (np.max(residual_px, 0) <= 2.)
        displacement = np.linalg.norm(world-world[0], axis=-1)*1000
        C = np.linalg.inv(t['camera_to_origin'][0])
        teacher_initial = world[0] @ C[:3, :3].T + C[:3, 3]
        error = np.linalg.norm(teacher_initial-d['query_xyz'], axis=-1)*1000
        current_good = valid[0] & np.isfinite(error)
        records.append(dict(session=record['session'], query_points=len(chosen),
            current_disagreement_percentiles_mm=np.percentile(error[current_good], [25, 50, 75, 90]).tolist(),
            candidate_static_tracks=int(candidate.sum()), candidate_rule='all11 visible/depth valid; camera-compensated pixel residual<=2px at every frame, audit only',
            static_candidate_median_displacement_mm=np.median(displacement[:, candidate], axis=1).tolist() if candidate.any() else None,
            original_measurement_offsets_s=((t['source_frame_ids']-t['source_frame_ids'][0])/30).tolist(),
            initial_tracker_query_residual_px=float(np.median(residual_px[0, current_good])),
            warning='Low image residual is not proof of physical staticness; depth drift and motion along viewing rays remain confounded'))
    a.output.write_text(json.dumps(dict(status='COMPLETED', manifest_sha256=sha256(a.data/'manifest.json'),
                                       scope='no model inference/training; label-quality audit only', records=records), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
