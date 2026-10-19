"""Export compact timelines and actual hand/object onset geometry for inspection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v5.storage import read_json, load_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'events', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    parent, events = read_json(a.manifest), read_json(a.events)
    rows = {r['key']: r for r in parent['sequences']}
    geometry_manifest = Path(parent['geometry_manifest'])
    geometries = {r['key']: r for r in read_json(geometry_manifest)['objects']}
    timelines, examples, sensitivity = [], {}, {}
    for sid in parent['source_ids']:
        candidates = [r for r in events['sequences'] if r['source_id'] == sid]
        counts = {}
        for dwell in ('2', '3', '4'):
            view = [r['counts'] if dwell == '3' else r['persistence_sensitivity'][dwell] for r in candidates]
            counts[dwell] = {k: sum(v[k] for v in view) for k in view[0]}
        sensitivity[str(sid)] = counts
        chosen = max(candidates, key=lambda r: (
            min(r['counts']['contact_transitions'], r['counts']['release_transitions']),
            r['counts']['contact_transitions']+r['counts']['release_transitions']))
        folder = Path(chosen['path']); meta = read_json(folder/'meta.json')
        transitions = load_array(folder/'transitions.npy')
        slot = int(transitions[0, 1]) if len(transitions) else 0
        arrays = {k: load_array(folder/(k+'.npy'))[:, slot].tolist()
                  for k in ('enter_count', 'leave_count', 'state', 'event_valid', 'raw_valid')}
        transitions = transitions[transitions[:, 1] == slot]
        timelines.append(dict(source_id=sid, key=chosen['key'], slot=slot, **arrays,
            transitions=transitions.tolist(), counts=chosen['counts'], observation_stride=meta['observation_stride']))
        # Bound geometry export: two CONTACT and one RELEASE source-distinct
        # examples, selected to expose transitions rather than estimate accuracy.
        if sid not in (0, 4, 10) or not len(transitions): continue
        preferred = transitions[transitions[:, 2] == (2 if sid == 10 else 1)]
        event = preferred[0] if len(preferred) else transitions[0]
        tick = int(event[0]); row = rows[chosen['key']]; hand = Path(row['path'])
        joints = load_array(hand/'joints.npy', mmap_mode='r')
        valid = load_array(hand/'joint_valid.npy', mmap_mode='r')
        poses = load_array(hand/'object_poses.npy', mmap_mode='r')
        name = row['objects'][slot]; key = 's{}:{}'.format(sid, name) if sid < 3 else name
        with load_array(geometry_manifest.parent/geometries[key]['path']) as g:
            examples['s{}_object'.format(sid)] = g['surface'].copy()
        frames = np.clip([tick-6, tick, tick+6], 0, len(joints)-1)
        anchors = np.linalg.inv(poses[frames, slot])
        points = np.einsum('fij,fhpj->fhpi', anchors[:, :3, :3], joints[frames])+anchors[:, None, None, :3, 3]
        dense = (load_array(hand/'surface_offsets.npy', mmap_mode='r')[frames].astype(np.float32)
                 +joints[frames, :, None, 0, :])
        dense = np.einsum('fij,fhpj->fhpi', anchors[:, :3, :3], dense)+anchors[:, None, None, :3, 3]
        dense_valid = load_array(hand/'surface_valid.npy', mmap_mode='r')[frames]
        confidence = load_array(Path(row['contact_path'])/'contact.npy', mmap_mode='r')[frames, slot]
        prefix = 's{}_'.format(sid)
        examples.update({prefix+'hand': points, prefix+'valid': valid[frames],
                         prefix+'surface': dense, prefix+'surface_valid': dense_valid,
                         prefix+'contact': confidence, prefix+'frames': frames, prefix+'event': event})
    value = dict(timelines=timelines, sensitivity=sensitivity, parent=str(a.manifest),
                 events=str(a.events), scope='transition-enriched diagnostic examples, not a population estimate')
    (a.output/'panel.json').write_text(json.dumps(value)+'\n')
    np.savez_compressed(a.output/'geometry.npz', **examples)
    for path in a.output.iterdir(): path.chmod(0o664)


if __name__ == '__main__': main()
