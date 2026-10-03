"""Independent source rows, float64 features, NumPy forwards and budget audit."""
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
from src.task.CmResidual.surface_granularity_prior import encode_granularity


def independent_features(raw, aggregation):
    v = {k: torch.from_numpy(x.astype(np.float64)) for k, x in raw.items()}
    r = v['pose'][:, :3, :3]
    def vector(x): return torch.bmm(x.reshape(len(r), -1, 3), r).reshape(x.shape)
    obj = vector(v['obj']-v['pose'][:, None, :3, 3])
    normal = torch.nn.functional.normalize(vector(v['normal']), dim=-1, eps=1e-8)
    relative = vector(v['hand']-v['obj'][:, :, None])
    hn = torch.nn.functional.normalize(vector(v['hand_normal']), dim=-1, eps=1e-8)
    flow = vector(v['next_hand']-v['hand'])
    parts = []
    for x, scale in ((relative,.05), (hn,1.), (flow,.01)):
        if aggregation == 'mean': x = x.mean(2, keepdim=True).expand_as(x)
        parts.append(x.flatten(2)/scale)
    x = torch.cat((obj/.05, normal, *parts, relative.norm(dim=-1).min(-1).values[..., None]/.05,
                   vector(v['global_hand_flow'][:, None]).expand_as(obj)/.01,
                   vector(v['obj']-v['previous_obj'])/.01), dim=-1)
    y = vector(v['next_obj']-v['obj'])/.01
    return x.numpy().astype(np.float32), y.numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root; prov = json.loads((root/'data/provenance.json').read_text()); parent = json.loads(Path(prov['parent_provenance']).read_text())
    assert sha(Path(prov['parent_provenance'])) == prov['parent_sha256']
    result = json.loads((root/'fit/results.json').read_text()); predictions = np.load(root/'fit/predictions.npz')
    original_ids = np.asarray(parent['object_point_ids']); ids = np.asarray(prov['object_point_ids'])
    expected_ids = np.concatenate((original_ids, np.sort(np.random.default_rng(3902).choice(np.setdiff1d(np.arange(4096), original_ids), 192, replace=False))))
    assert np.array_equal(ids, expected_ids)
    decoded = {}; checked_rows = 0; feature_error = 0.; kd_error = 0.; network_error = 0.; metric_error = 0.
    torch.set_num_threads(2)
    for group, records in prov['selection'].items():
        assert records == (parent['selection'][group][:2048] if group.endswith('train') else parent['selection'][group])
        assert sha(root/'data'/(group+'.npz')) == prov['selected_packet_sha256'][group]
        with np.load(root/'data'/(group+'.npz')) as f: raw = {key: f[key] for key in f.files}
        for geometry in sorted(set(row['geometry'] for row in records)):
            d = Path(geometry)
            fields = ('obj_pose_world', 'obj_points_pool_world', 'obj_normals_pool_world', 'knn_hand_points_world', 'knn_hand_normals_world', 'obj_knn_indices')
            arrays = {key: np.load(d/(key+'.npy'), mmap_mode='r') for key in fields}
            chosen = [i for i, row in enumerate(records) if row['geometry']==geometry]
            for i in chosen:
                t = records[i]['current']; hids = arrays['obj_knn_indices'][t, ids, :4]
                rebuilt = dict(pose=arrays['obj_pose_world'][t], obj=arrays['obj_points_pool_world'][t, ids],
                               normal=arrays['obj_normals_pool_world'][t, ids], previous_obj=arrays['obj_points_pool_world'][t-1, ids],
                               next_obj=arrays['obj_points_pool_world'][t+1, ids], hand=arrays['knn_hand_points_world'][t, hids],
                               next_hand=arrays['knn_hand_points_world'][t+1, hids], hand_normal=arrays['knn_hand_normals_world'][t, hids],
                               global_hand_flow=(np.array(arrays['knn_hand_points_world'][t+1])-np.array(arrays['knn_hand_points_world'][t])).mean(0))
                for key, value in rebuilt.items(): assert np.array_equal(raw[key][i], value.astype(np.float32)), (group, key)
                distances = np.linalg.norm(raw['hand'][i]-raw['obj'][i,:,None], axis=-1)
                assert np.all(np.diff(distances,axis=-1)>=-1e-7), 'neighbor rank changed'
                checked_rows += 1
            i = chosen[0]; t = records[i]['current']
            nearest, _ = cKDTree(arrays['knn_hand_points_world'][t]).query(raw['obj'][i], k=4)
            cached = np.linalg.norm(raw['hand'][i]-raw['obj'][i,:,None], axis=-1)
            error = float(np.abs(cached-nearest).max()); kd_error = max(kd_error,error); assert error < 1e-5
        for aggregation in ('mean', 'detail'):
            chunks = []
            for begin in range(0,len(records),64):
                subset = {k: v[begin:begin+64] for k,v in raw.items()}
                x, y = independent_features(subset, aggregation); px, py = encode_granularity(subset, aggregation)
                assert np.allclose(x,px,atol=2e-5,rtol=2e-6) and np.allclose(y,py,atol=2e-5,rtol=2e-6)
                feature_error = max(feature_error,float(np.abs(x-px).max()),float(np.abs(y-py).max()))
                if group.endswith('eval'): chunks.append(px)
                # Future object labels cannot enter either input condition.
                subset['next_obj'] = subset['next_obj']+1
                altered, _ = encode_granularity(subset, aggregation); assert np.array_equal(px,altered)
            if group.endswith('eval'):
                decoded[group,aggregation] = (np.concatenate(chunks),encode_granularity(raw,aggregation)[1][:,:64])
    expected_schedule = torch.randint(2048,(1500,32),generator=torch.Generator().manual_seed(3904))
    common = None
    for name, train in result['training'].items():
        checkpoint = torch.load(root/'fit'/(name+'.pt'), map_location='cpu', weights_only=False)
        assert checkpoint['steps']==1500 and torch.equal(checkpoint['batch_schedule'],expected_schedule)
        assert all(int(value['step'])==1500 for value in checkpoint['optimizer']['state'].values())
        initial = checkpoint['initial']
        if common is None: common = initial
        assert all(torch.equal(v, common[k]) for k,v in initial.items())
        assert all(torch.isfinite(v).all() for v in checkpoint['state'].values())
        assert max(float((v-initial[k]).abs().max()) for k,v in checkpoint['state'].items()) > 0
        assert sum(v.numel() for v in checkpoint['state'].values())==train['parameters']==23107
        queries = checkpoint['queries']; aggregation = checkpoint['aggregation']
        for group in ('mano_eval','inspire_eval'):
            x,y = decoded[group,aggregation]; stored = predictions[name+'__'+group]
            assert stored.shape == y.shape and y.shape[1]==64
            for begin in range(0,len(x),32):
                rebuilt = numpy_predict(checkpoint['state'],x[begin:begin+32,:queries])[:,:64]
                error = float(np.abs(rebuilt-stored[begin:begin+32]).max()); network_error = max(network_error,error); assert error<2e-4
            errors = np.linalg.norm(stored.astype(np.float64)-y.astype(np.float64),axis=-1).mean(-1)*10
            names = np.asarray([r['parent'] for r in prov['selection'][group]])
            per_parent = {p:float(errors[names==p].mean()) for p in sorted(set(names))}
            error = abs(float(np.mean(list(per_parent.values())))-result['reports'][name][group]['parent_epe_mm'])
            metric_error=max(metric_error,error); assert error<1e-4
    independent_gains = {}
    for hand in ('mano','inspire'):
        baseline=result['reports'][hand+'_64_mean'][hand+'_eval']['parent_epe_mm']
        for variant in ('256_mean','64_detail','256_detail'):
            name=hand+'_'+variant
            independent_gains[name]=(baseline-result['reports'][name][hand+'_eval']['parent_epe_mm'])/baseline
            assert abs(independent_gains[name]-result['gains'][name])<1e-12
            assert (independent_gains[name]>=.10)==result['gates'][name]
    label='PROMISING' if max(independent_gains.values())>=.10 else ('UNCLEAR' if max(independent_gains.values())>=.05 else 'UNPROMISING')
    assert label==result['label'] and sum(r['steps'] for r in result['training'].values())==result['actual_optimizer_updates']==12000
    for path,stat in prov['source_stat'].items(): assert file_stat(Path(path))==stat
    for path,digest in prov['metadata_sha256'].items(): assert sha(Path(path))==digest
    audit=dict(run_status='COMPLETED',label=label,source_rows=checked_rows,feature_max=feature_error,
               nearest_distance_max_m=kd_error,numpy_forward_max=network_error,metric_max_mm=metric_error,
               identical_schedule_initial_and_target_budget=True,full_optimizer_replay=False)
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit),flush=True)


if __name__=='__main__': main()
