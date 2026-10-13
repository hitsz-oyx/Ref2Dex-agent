#!/usr/bin/env python3
"""Audit fixed current-image target annotations against unmodified future tracks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from goal_point_wam.object_motion_goal import pack_object_motion_goal, rigid_consistency


def project_current_coordinates(xyz, camera0, camera_t, intrinsic):
    """Native camera poses map camera coordinates to the common origin."""
    transform = np.linalg.inv(camera_t) @ camera0
    camera = xyz @ transform[:3, :3].T + transform[:3, 3]
    pixels = camera @ intrinsic.T
    visible = np.isfinite(pixels).all(-1) & (camera[:, 2] > 0)
    uv = pixels[:, :2] / np.where(visible, pixels[:, 2], 1)[:, None]
    return uv, visible


def run(root):
    manifest = json.loads((root / 'inputs/manifest.json').read_text())
    annotations = json.loads((root / 'annotations.json').read_text())
    lookup = {r['id']: r for r in annotations['records']}
    (root / 'goals').mkdir(exist_ok=True)
    (root / 'overlays').mkdir(exist_ok=True)
    results = []
    for record in manifest['records']:
        rid = record['id']; ann = lookup[rid]
        row = dict(id=rid, split=record['record']['split'], session=record['record']['session'])
        if 'excluded_reason' in ann:
            row.update(excluded_reason=ann['excluded_reason'], numeric_admitted=False)
            results.append(row); continue
        if ann.get('identity_from_current_only') is not True:
            raise ValueError('current-only annotation required')
        base = root / 'inputs' / rid
        with np.load(base / 'clip.npz') as data:
            sample = {key: data[key] for key in data.files}
        with np.load(base / 'projection.npz') as data:
            projection = {key: data[key] for key in data.files}
        polygon = np.asarray(ann['polygon'], np.float32)
        inside = np.array([cv2.pointPolygonTest(polygon, tuple(map(float, uv)), False) >= 0
                           for uv in projection['current_uv']])
        ids = np.flatnonzero(inside & projection['projection_valid'] & sample['query_valid'])
        goal = pack_object_motion_goal(sample, ids)
        np.savez_compressed(root / 'goals' / (rid + '.npz'), **goal)
        diagnostics = rigid_consistency(goal)
        selected = int(goal['current_object_valid'].sum())
        reasons = []
        if selected < 16:
            reasons.append('fewer_than_16_current_points')
        for d in diagnostics:
            d['valid_fraction'] = d['valid_points'] / max(selected, 1)
            if d['valid_fraction'] < .8:
                reasons.append('insufficient_visibility_at_%d' % d['offset'])
            if (not d['identifiable'] or d.get('residual_median_mm', float('inf')) > 5
                    or d.get('residual_p95_mm', float('inf')) > 15):
                reasons.append('rigid_inconsistency_at_%d' % d['offset'])
        row.update(target=ann['target'], candidate_points=len(ids), selected_points=selected,
                   diagnostics=diagnostics, numeric_admitted=not reasons, rejection_reasons=reasons,
                   intent_ambiguous=ann.get('intent_ambiguous', False), visual_review='pending')
        results.append(row)
        panels = []
        for j, tick in enumerate([0, 8, 16, 24]):
            frame = cv2.imread(str(base / ('frame%02d.png' % tick)))
            if frame is None:
                raise ValueError('missing RGB frame')
            xyz = goal['start_object_points'] if j == 0 else goal['desired_object_points'][j-1]
            mask = goal['current_object_valid'] if j == 0 else goal['desired_object_valid'][j-1]
            uv, front = project_current_coordinates(xyz, projection['camera_to_origin'][0],
                                                   projection['camera_to_origin'][j], projection['intrinsic'])
            if j == 0:
                cv2.polylines(frame, [polygon.astype(np.int32)], True, (255, 255, 0), 1)
            for point, valid in zip(uv, mask & front):
                if valid and 0 <= point[0] < 640 and 0 <= point[1] < 360:
                    cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, (0, 255, 0), -1)
            panel = np.full((390, 640, 3), 245, np.uint8); panel[:360] = frame
            text = '%s +%d: %d valid' % (rid, tick, mask.sum())
            if j:
                d = diagnostics[j-1]
                text += ' residual p95 %.1f mm' % d.get('residual_p95_mm', float('nan'))
            cv2.putText(panel, text, (8, 380), 0, .5, (0, 0, 0), 1)
            panels.append(panel)
        cv2.imwrite(str(root / 'overlays' / (rid + '.jpg')),
                    np.concatenate([np.concatenate(panels[:2], 1), np.concatenate(panels[2:], 1)], 0))
    summary = dict(schema='ref2dex.video-object-goal-audit.v1', training_allowed=False,
                   visual_review_required=True, records=results)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in results:
        print(row['id'], row['split'], row.get('selected_points', '-'),
              'numeric_pass' if row['numeric_admitted'] else row.get('rejection_reasons', row.get('excluded_reason')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
