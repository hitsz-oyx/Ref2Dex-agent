"""Read the fixed old window selections with nested 64/256 object queries."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.prepare_surface_prior_data import RAW_FIELDS, FILES, file_stat, read_row
from scripts.run_contact_response_probe import sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    out = a.output.resolve(); assert ROOT in out.parents and not out.exists()
    parent_path = a.source/'data/provenance.json'; parent = json.loads(parent_path.read_text())
    original = np.asarray(parent['object_point_ids'], dtype=np.int64)
    remaining = np.setdiff1d(np.arange(4096), original)
    obj_ids = np.concatenate((original, np.sort(np.random.default_rng(3902).choice(remaining, 192, replace=False))))
    assert len(set(obj_ids.tolist())) == 256
    out.mkdir(); selection = {}; selected_stat = {}; meta_hashes = {}; packets = {}; groups = {}
    for group, all_records in parent['selection'].items():
        records = all_records[:2048] if group.endswith('train') else all_records
        raw = {key: [None]*len(records) for key in RAW_FIELDS}
        for geometry in sorted(set(row['geometry'] for row in records)):
            directory = Path(geometry)
            for name in FILES:
                path = directory/name
                assert file_stat(path) == parent['source_stat'][str(path)]
                selected_stat[str(path)] = file_stat(path)
            assert sha(directory/'manifest.json') == parent['metadata_sha256'][str(directory/'manifest.json')]
            meta_hashes[str(directory/'manifest.json')] = sha(directory/'manifest.json')
            arrays = {name[:-4]: np.load(directory/name, mmap_mode='r') for name in FILES if name.endswith('.npy')}
            for i, record in enumerate(records):
                if record['geometry'] != geometry: continue
                row = read_row(arrays, record['current'], obj_ids)
                for key, value in row.items(): raw[key][i] = value
            for name in FILES:
                path = directory/name; assert file_stat(path) == selected_stat[str(path)]
        packet = {key: np.stack(value) for key, value in raw.items()}
        assert sha(a.source/'data'/(group+'.npz')) == parent['selected_packet_sha256'][group]
        with np.load(a.source/'data'/(group+'.npz')) as old:
            for key, value in packet.items():
                inherited = value if key in ('pose', 'global_hand_flow') else value[:, :64]
                assert np.array_equal(inherited, old[key][:len(records)]), (group, key)
        np.savez(out/(group+'.npz'), **packet)
        packets[group] = sha(out/(group+'.npz')); selection[group] = records
        groups[group] = dict(windows=len(records), parents=len(set(row['parent'] for row in records)), objects=len(set(row['object'] for row in records)))
        print(json.dumps(dict(group=group, **groups[group])), flush=True)
    train_parents = {r['parent'] for g, rows in selection.items() if g.endswith('train') for r in rows}
    eval_parents = {r['parent'] for g, rows in selection.items() if g.endswith('eval') for r in rows}
    assert not train_parents & eval_parents
    for path, stat in selected_stat.items(): assert file_stat(Path(path)) == stat
    result = dict(parent_provenance=str(parent_path.resolve()), parent_sha256=sha(parent_path), object_point_ids=obj_ids.tolist(),
                  selection=selection, groups=groups, source_stat=selected_stat, metadata_sha256=meta_hashes,
                  selected_packet_sha256=packets, source_files_full_content_not_hashed=True,
                  legacy_inspire_physics=True, realized_hand_motion_offline=True)
    (out/'provenance.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
