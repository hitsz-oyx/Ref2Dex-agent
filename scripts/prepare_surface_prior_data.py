"""Read-only cache subset extraction; no learned models or simulation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha

RAW_FIELDS = ('pose', 'obj', 'normal', 'previous_obj', 'next_obj', 'hand', 'next_hand', 'hand_normal', 'global_hand_flow')
FILES = ('manifest.json', 'obj_pose_world.npy', 'obj_points_pool_world.npy', 'obj_normals_pool_world.npy',
         'knn_hand_points_world.npy', 'knn_hand_normals_world.npy', 'obj_knn_indices.npy',
         'frame_time.npy', 'source_frame_id.npy', 'obj_candidate_mask_2cm.npy')


def file_stat(path):
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def read_row(arrays, current, obj_ids):
    pose = np.array(arrays['obj_pose_world'][current], dtype=np.float32)
    obj = np.array(arrays['obj_points_pool_world'][current, obj_ids], dtype=np.float32)
    ids = np.array(arrays['obj_knn_indices'][current, obj_ids, :4], dtype=np.int64)
    hand = np.array(arrays['knn_hand_points_world'][current, ids], dtype=np.float32)
    assert ids.min() >= 0 and ids.max() < arrays['knn_hand_points_world'].shape[1]
    assert abs(float(arrays['frame_time'][current+1]-arrays['frame_time'][current])-1/30) < 1e-4
    assert np.max(np.abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3))) < 1e-3
    return dict(pose=pose, obj=obj,
                normal=np.array(arrays['obj_normals_pool_world'][current, obj_ids], dtype=np.float32),
                previous_obj=np.array(arrays['obj_points_pool_world'][current-1, obj_ids], dtype=np.float32),
                next_obj=np.array(arrays['obj_points_pool_world'][current+1, obj_ids], dtype=np.float32), hand=hand,
                next_hand=np.array(arrays['knn_hand_points_world'][current+1, ids], dtype=np.float32),
                hand_normal=np.array(arrays['knn_hand_normals_world'][current, ids], dtype=np.float32),
                global_hand_flow=(np.asarray(arrays['knn_hand_points_world'][current+1], dtype=np.float32)
                                  - np.asarray(arrays['knn_hand_points_world'][current], dtype=np.float32)).mean(0))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--index', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); assert ROOT in out.parents and not out.exists()
    index = json.loads(args.index.read_text()); base = args.index.parent
    groups = dict(mano_train=('train', 'grab'), inspire_train=('train', 'inspire_f1'), mano_eval=('test', 'grab'), inspire_eval=('val', 'inspire_f1'))
    obj_ids = np.sort(np.random.default_rng(3802).choice(4096, 64, replace=False))
    packets = {}; provenance = {}; stats = {}; metadata_hashes = {}; parent_sets = {}
    for group, (split, source) in groups.items():
        rows = []; raw = {key: [] for key in RAW_FIELDS}; records = []
        for entry in index['sequences'][split]:
            if entry['source'] != source: continue
            directory = base/entry['path']/'geometry'; meta = json.loads((directory/'manifest.json').read_text())
            assert meta['effective_fps'] == 30 and meta['coordinate_frame'] == 'object_pose_t'
            expected = 2048 if source == 'grab' else 10135
            assert meta['knn_hand_points'] == expected and meta['knn_index_semantics'] == 'object_pool_point_to_current_frame_knn_hand_point'
            for name in FILES:
                path = directory/name; stats[str(path)] = file_stat(path)
            metadata_hashes[str(directory/'manifest.json')] = sha(directory/'manifest.json')
            arrays = {name[:-4]: np.load(directory/name, mmap_mode='r') for name in FILES if name.endswith('.npy')}
            candidates = np.arange(1, entry['frame_count']-1, 4)
            mask = arrays['obj_candidate_mask_2cm']
            candidates = candidates[np.any(mask[candidates], axis=1)]
            if len(candidates) > 64: candidates = candidates[np.linspace(0, len(candidates)-1, 64, dtype=int)]
            for current in candidates:
                row = read_row(arrays, int(current), obj_ids)
                for key, value in row.items(): raw[key].append(value)
                records.append(dict(parent=entry['parent_seq_id'], object=entry['object_name'], geometry=str(directory),
                                    current=int(current), raw_frame=int(arrays['source_frame_id'][current]), hand_points=expected))
            print(json.dumps(dict(group=group, parent=entry['parent_seq_id'], windows=len(candidates))), flush=True)
        permutation = np.random.default_rng(3801).permutation(len(records))
        packet = {key: np.stack(value)[permutation] for key, value in raw.items()}
        records = [records[int(i)] for i in permutation]
        assert all(np.isfinite(value).all() for value in packet.values())
        packets[group] = packet; provenance[group] = records; parent_sets[group] = set(r['parent'] for r in records)
    fit_parents = parent_sets['mano_train'] | parent_sets['inspire_train']
    eval_parents = parent_sets['mano_eval'] | parent_sets['inspire_eval']
    assert not fit_parents & eval_parents
    assert len(provenance['mano_train']) >= 7168 and len(provenance['inspire_train']) >= 7168
    for path, record in stats.items(): assert file_stat(Path(path)) == record
    out.mkdir()
    for group, packet in packets.items(): np.savez(out/(group+'.npz'), **packet)
    summary = dict(run_status='COMPLETED', index=str(args.index), index_sha256=sha(args.index), object_point_ids=obj_ids.tolist(),
                   selection=provenance, source_stat=stats, metadata_sha256=metadata_hashes,
                   selected_packet_sha256={group: sha(out/(group+'.npz')) for group in packets},
                   source_files_full_content_not_hashed=True, selected_bytes_reaudited_from_source=True,
                   groups={group: dict(windows=len(records), parents=len(parent_sets[group]), objects=len(set(r['object'] for r in records))) for group, records in provenance.items()},
                   source_inspire_physics_legacy=True, realized_hand_motion_offline=True)
    (out/'provenance.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__': main()
