"""Independent selected-source, Torch64 geometry and NumPy network audit."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha
from scripts.prepare_surface_prior_data import file_stat
from src.task.CmResidual.surface_motion_prior import numpy_predict


def independent_features(raw):
    values = {key: torch.from_numpy(value.astype(np.float64)) for key, value in raw.items()}
    r = values['pose'][:, :3, :3]; origin = values['pose'][:, None, :3, 3]
    def vector(value): return torch.matmul(value.reshape(len(r), -1, 3), r).reshape(value.shape)
    obj = vector(values['obj']-origin)
    normal = torch.nn.functional.normalize(vector(values['normal']), dim=-1, eps=1e-8)
    relative = vector(values['hand']-values['obj'][:, :, None])
    normal_hand = torch.nn.functional.normalize(vector(values['hand_normal']), dim=-1, eps=1e-8)
    local_flow = vector(values['next_hand']-values['hand']).mean(2)
    global_flow = vector(values['global_hand_flow'][:, None]).expand_as(obj)
    distance = relative.norm(dim=-1).min(-1).values[..., None]
    previous = vector(values['obj']-values['previous_obj'])
    features = torch.cat((obj/.05, normal, relative.mean(2)/.05, normal_hand.mean(2), local_flow/.01,
                          distance/.05, global_flow/.01, previous/.01), -1)
    target = vector(values['next_obj']-values['obj'])/.01
    return features.numpy().astype(np.float32), target.numpy().astype(np.float32)


def parent_error(prediction, target, records):
    errors = np.sqrt(((prediction.astype(np.float64)-target.astype(np.float64))**2).sum(-1)).mean(-1)*10
    names = sorted(set(record['parent'] for record in records))
    grouped = {name: float(np.mean([errors[i] for i, row in enumerate(records) if row['parent']==name])) for name in names}
    return float(np.mean(list(grouped.values())))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root; data_dir = root/'data'; fit = root/'fit'
    torch.set_num_threads(2)
    p = json.loads((data_dir/'provenance.json').read_text()); result = json.loads((fit/'results.json').read_text())
    prediction_file = np.load(fit/'predictions.npz'); decoded = {}; feature_error = 0.; raw_rows = 0; kd_error = 0.; rigid_error = 0.
    object_ids = np.asarray(p['object_point_ids']); index = json.loads(Path(p['index']).read_text()); base = Path(p['index']).parent
    assert sha(Path(p['index'])) == p['index_sha256']
    assert np.array_equal(object_ids, np.sort(np.random.default_rng(3802).choice(4096,64,replace=False)))
    from src.task.CmResidual.surface_motion_prior import encode_raw
    for group, records in p['selection'].items():
        split, source = {'mano_train':('train','grab'),'inspire_train':('train','inspire_f1'),
                         'mano_eval':('test','grab'),'inspire_eval':('val','inspire_f1')}[group]
        expected_rows = []
        for entry in index['sequences'][split]:
            if entry['source'] != source: continue
            geometry = base/entry['path']/'geometry'
            mask = np.load(geometry/'obj_candidate_mask_2cm.npy',mmap_mode='r')
            candidates = [t for t in range(1,entry['frame_count']-1,4) if bool(mask[t].any())]
            if len(candidates)>64: candidates = [candidates[i] for i in np.linspace(0,len(candidates)-1,64,dtype=int)]
            expected_rows.extend((entry['parent_seq_id'],str(geometry),t) for t in candidates)
        order = np.random.default_rng(3801).permutation(len(expected_rows))
        expected_rows = [expected_rows[i] for i in order]
        assert expected_rows == [(row['parent'],row['geometry'],row['current']) for row in records]
        assert sha(data_dir/(group+'.npz')) == p['selected_packet_sha256'][group]
        payload = np.load(data_dir/(group+'.npz')); raw = {key: payload[key] for key in payload.files}
        # All selected source bytes are reconstructed, never treating metadata as a content hash.
        for geometry in sorted(set(row['geometry'] for row in records)):
            directory = Path(geometry)
            arrays = {name: np.load(directory/(name+'.npy'), mmap_mode='r') for name in
                      ('obj_pose_world', 'obj_points_pool_world', 'obj_normals_pool_world', 'knn_hand_points_world',
                       'knn_hand_normals_world', 'obj_knn_indices', 'obj_candidate_mask_2cm', 'source_frame_id', 'frame_time')}
            chosen = [i for i, row in enumerate(records) if row['geometry']==geometry]
            for i in chosen:
                row = records[i]; t = row['current']; hids = arrays['obj_knn_indices'][t, object_ids, :4].astype(np.int64)
                assert np.any(arrays['obj_candidate_mask_2cm'][t]) and (t-1)%4 == 0
                assert int(arrays['source_frame_id'][t]) == row['raw_frame'] and abs(float(arrays['frame_time'][t+1]-arrays['frame_time'][t])-1/30)<1e-4
                rebuilt = dict(pose=arrays['obj_pose_world'][t], obj=arrays['obj_points_pool_world'][t, object_ids],
                               normal=arrays['obj_normals_pool_world'][t, object_ids], previous_obj=arrays['obj_points_pool_world'][t-1, object_ids],
                               next_obj=arrays['obj_points_pool_world'][t+1, object_ids], hand=arrays['knn_hand_points_world'][t, hids],
                               next_hand=arrays['knn_hand_points_world'][t+1, hids], hand_normal=arrays['knn_hand_normals_world'][t, hids],
                               global_hand_flow=(np.asarray(arrays['knn_hand_points_world'][t+1], dtype=np.float32)-np.asarray(arrays['knn_hand_points_world'][t], dtype=np.float32)).mean(0))
                for key, values in rebuilt.items(): assert np.array_equal(raw[key][i], np.asarray(values, dtype=np.float32)), (group, row, key)
                pose = np.asarray(arrays['obj_pose_world'][t], dtype=np.float64)
                local = (np.asarray(raw['obj'][i],dtype=np.float64)-pose[:3,3]) @ pose[:3,:3]
                for neighbor, field in ((t-1,'previous_obj'),(t+1,'next_obj')):
                    other_pose = np.asarray(arrays['obj_pose_world'][neighbor],dtype=np.float64)
                    other_local = (np.asarray(raw[field][i],dtype=np.float64)-other_pose[:3,3]) @ other_pose[:3,:3]
                    value = float(np.abs(local-other_local).max()); rigid_error = max(rigid_error,value); assert value<1e-4
                raw_rows += 1
            i = chosen[0]; t = records[i]['current']; hand = np.asarray(arrays['knn_hand_points_world'][t])
            distances, _ = cKDTree(hand).query(raw['obj'][i], k=4)
            cached = np.sort(np.linalg.norm(raw['hand'][i]-raw['obj'][i, :, None], axis=-1), axis=-1)
            error = float(np.abs(cached-distances).max()); kd_error = max(kd_error, error); assert error < 1e-5
        x, y = independent_features(raw); primary_x, primary_y = encode_raw(raw)
        assert np.allclose(x, primary_x, atol=2e-5, rtol=2e-6) and np.allclose(y, primary_y, atol=2e-5, rtol=2e-6)
        feature_error = max(feature_error, float(np.abs(x-primary_x).max()), float(np.abs(y-primary_y).max()))
        # Future object changes must alter only the label, never the input.
        original_next = raw['next_obj']; raw['next_obj'] = original_next[::-1]
        modified_x, _ = encode_raw(raw); assert np.array_equal(modified_x, primary_x); raw['next_obj'] = original_next
        decoded[group] = (primary_x, primary_y)
    fit_parents = set(row['parent'] for g in ('mano_train', 'inspire_train') for row in p['selection'][g])
    eval_parents = set(row['parent'] for g in ('mano_eval', 'inspire_eval') for row in p['selection'][g]); assert not fit_parents & eval_parents
    network_error = 0.; metric_error = 0.; counters = {}; template = None; recomputed = {}
    for name, record in result['training'].items():
        checkpoint = torch.load(fit/(name+'.pt'), map_location='cpu', weights_only=False)
        expected_steps = 600 if name.startswith('adapt_') else 1500
        steps = [int(state['step']) for state in checkpoint['optimizer']['state'].values()]
        assert steps and set(steps)=={expected_steps} and checkpoint['steps']==expected_steps
        counters[name] = steps[0]
        generator = torch.Generator().manual_seed(3804)
        expected_schedule = torch.randint(checkpoint['size'], (expected_steps,32), generator=generator)
        assert torch.equal(expected_schedule, checkpoint['batch_schedule']) and torch.equal(generator.get_state(),checkpoint['generator_state'])
        if template is None: template = checkpoint['common_initial']
        assert all(torch.equal(template[key], value) for key, value in checkpoint['common_initial'].items())
        if not name.startswith('adapt_'): assert all(torch.equal(template[key], value) for key, value in checkpoint['initial'].items())
        elif name != 'adapt_scratch':
            previous = torch.load(fit/(name[len('adapt_'):]+'.pt'), map_location='cpu', weights_only=False)
            assert all(torch.equal(previous['state'][key], value) for key, value in checkpoint['initial'].items())
        else: assert all(torch.equal(template[key], value) for key, value in checkpoint['initial'].items())
        assert record['size'] == (256 if name.startswith('adapt_') else checkpoint['size']) and checkpoint['maximum_parameter_change']>0
        if checkpoint['shuffled']: assert np.array_equal(checkpoint['target_permutation'].numpy(), np.random.default_rng(3805).permutation(7168))
        for group in ('mano_eval', 'inspire_eval'):
            x, y = decoded[group]; x = x.copy()
            if record['motion_off']: x[..., [12,13,14,16,17,18]] = 0
            actual = prediction_file[name+'__'+group]; predictions = []
            for start in range(0, len(x), 32): predictions.append(numpy_predict(checkpoint['state'], x[start:start+32]))
            rebuilt = np.concatenate(predictions); error = float(np.abs(rebuilt-actual).max()); network_error = max(network_error, error); assert error < 2e-4
            metric = parent_error(actual, y, p['selection'][group]); metric_error = max(metric_error, abs(metric-result['reports'][name][group]['parent_epe_mm']))
            assert abs(metric-result['reports'][name][group]['parent_epe_mm']) < 1e-4
            if group == ('mano_eval' if name.startswith('mano_') and name not in ('mano_motion_off','mano_shuffled') else 'inspire_eval'): recomputed[name] = metric
    persistence = parent_error(decoded['inspire_eval'][0][...,19:22], decoded['inspire_eval'][1], p['selection']['inspire_eval'])
    gates = {hand+'_scale': recomputed[hand+'_7168'] <= .9*recomputed[hand+'_512'] and recomputed[hand+'_2048'] <= 1.02*recomputed[hand+'_512'] for hand in ('mano','inspire')}
    error = recomputed['adapt_mano_7168']
    gates['cross_prior'] = error <= .9*recomputed['adapt_scratch'] and error <= .95*recomputed['adapt_mano_motion_off'] and error <= .95*recomputed['adapt_mano_shuffled'] and error <= .95*persistence
    label = 'PROMISING' if gates['cross_prior'] else ('UNCLEAR' if any(gates.values()) else 'UNPROMISING')
    assert gates == result['gates'] and label == result['label'] and sum(counters.values()) == result['actual_optimizer_updates']
    for path, stat in p['source_stat'].items(): assert file_stat(Path(path)) == stat
    for path, digest in p['metadata_sha256'].items(): assert sha(Path(path)) == digest
    report = dict(run_status='COMPLETED', source_rows_rebuilt=raw_rows, kd_max_m=kd_error, rigid_point_correspondence_max_m=rigid_error, feature_maximum=feature_error,
                  numpy_network_maximum=network_error, metric_maximum_mm=metric_error, optimizer_counters=counters,
                  gates=gates, label=label, source_selected_bytes_verified=True, source_full_files_hash_not_claimed=True,
                  future_object_input_leak_check=True, full_optimizer_replay=False,
                  cpu_reason='Independent Torch64 geometry and NumPy arithmetic audit, deliberately separate from CUDA inference')
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
