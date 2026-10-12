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
from trajectory_geometry import shared_trajectory_depth


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
        if record.get('teacher_audit_path') and sha256(Path(record['teacher_audit_path'])) != record['teacher_audit_sha256']:
            raise ValueError('teacher calibration identity drift')
        with np.load(teacher, allow_pickle=False) as f:
            t = {k: f[k].copy() for k in f.files}
        with np.load(clip, allow_pickle=False) as f:
            d = {k: f[k].copy() for k in f.files}
        chosen = d['track_ids']
        frame_ids = np.asarray(record['teacher_frame_ids'])
        offsets = np.searchsorted(t['source_frame_ids'], frame_ids)
        if not np.array_equal(t['source_frame_ids'][offsets], frame_ids):
            raise ValueError('label clock differs from recorded teacher')
        depth = t['depth_m'][offsets]
        if record.get('trajectory_depth_path'):
            import h5py
            metadata_path = Path(record['trajectory_geometry_manifest_path'])
            if sha256(metadata_path) != record['trajectory_geometry_manifest_sha256']:
                raise ValueError('shared geometry metadata changed')
            metadata = json.loads(metadata_path.read_text())
            with h5py.File(Path(metadata['video']).with_suffix('.hdf5'), 'r') as raw:
                native_camera = raw['transforms/camera'][:]
            depth, _, provenance = shared_trajectory_depth(Path(record['trajectory_depth_path']),
                frame_ids, t['intrinsic'], native_camera, metadata['video'])
            if provenance['trajectory_depth_sha256'] != record['trajectory_depth_sha256']:
                raise ValueError('shared geometry depth changed')
        if 'label_depth_path' in record:
            depth_path = a.data/record['label_depth_path']
            if sha256(depth_path) != record['label_depth_sha256']:
                raise ValueError('rolling label depth changed')
            with np.load(depth_path, allow_pickle=False) as f:
                if not np.array_equal(f['source_frame_ids'], frame_ids):
                    raise ValueError('rolling depth clock mismatch')
                depth = f['depth_m'].copy()
        camera = t['camera_to_origin'][offsets]
        uv = t['tracks_uv'][offsets][:, chosen]
        world, valid = lift_tracks(uv, t['visibility'][offsets][:, chosen], depth, t['intrinsic'], camera)
        initial = np.broadcast_to(world[0], world.shape)
        projected, _ = project_world(initial[:, None], t['intrinsic'], camera)
        residual_px = np.linalg.norm(projected[:, 0]-uv, axis=-1)
        # Future-based strata are AUDIT ONLY, never used in training input sampling.
        candidate = valid.all(0) & np.isfinite(residual_px).all(0) & (np.max(residual_px, 0) <= 2.)
        displacement = np.linalg.norm(world-world[0], axis=-1)*1000
        C = np.linalg.inv(camera[0])
        teacher_initial = world[0] @ C[:3, :3].T + C[:3, 3]
        error = np.linalg.norm(teacher_initial-d['query_xyz'], axis=-1)*1000
        current_good = valid[0] & np.isfinite(error)
        first = offsets[:1]
        raw_world, raw_valid = lift_tracks(t['tracks_uv'][first][:, chosen], t['visibility'][first][:, chosen],
            t['depth_m'][first], t['intrinsic'], t['camera_to_origin'][first])
        raw_current = raw_world[0] @ C[:3, :3].T + C[:3, 3]
        raw_error = np.linalg.norm(raw_current-d['query_xyz'], axis=-1)*1000
        raw_good = raw_valid[0] & np.isfinite(raw_error)
        medians = np.median(displacement[:, candidate], axis=1) if candidate.any() else None
        p90 = np.percentile(displacement[:, candidate], 90, axis=1) if candidate.any() else None
        gates = dict(current_alignment_p90_le025mm=bool(np.percentile(error[current_good], 90) <= .25),
                     at_least32_background_audit_candidates=bool(candidate.sum() >= 32),
                     background_median_le10mm=bool(medians is not None and medians.max() <= 10),
                     background_p90_le25mm=bool(p90 is not None and p90.max() <= 25),
                     at_least256_current_points=bool(current_good.sum() >= 256),
                     at_least25percent_future_valid=bool(d['point_target_valid'].mean() >= .25))
        if meta.get('label_calibrations'):
            gates['all_scale_iqr_le035'] = all(c['relative_iqr'] <= .35 for c in meta['label_calibrations'])
        records.append(dict(session=record['session'], query_points=len(chosen),
            label_depth_backend=meta.get('label_depth_backend', 'offline'),
            current_disagreement_percentiles_mm=np.percentile(error[current_good], [25, 50, 75, 90]).tolist(),
            original_offline_current_disagreement_percentiles_mm=np.percentile(raw_error[raw_good], [25, 50, 75, 90]).tolist(),
            candidate_static_tracks=int(candidate.sum()), candidate_rule='all11 visible/depth valid; camera-compensated pixel residual<=2px at every frame, audit only',
            static_candidate_median_displacement_mm=medians.tolist() if medians is not None else None,
            static_candidate_p90_displacement_mm=p90.tolist() if p90 is not None else None,
            original_measurement_offsets_s=((frame_ids-frame_ids[0])/30).tolist(),
            initial_tracker_query_residual_px=float(np.median(residual_px[0, current_good])),
            admission_screen=gates, admission_screen_pass=all(gates.values()),
            warning='Low image residual is not proof of physical staticness; depth drift and motion along viewing rays remain confounded'))
    a.output.write_text(json.dumps(dict(status='COMPLETED', manifest_sha256=sha256(a.data/'manifest.json'),
                                       scope='no model inference/training; label-quality audit only', records=records), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
