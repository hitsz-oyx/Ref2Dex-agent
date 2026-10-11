#!/usr/bin/env python3
"""Common endpoint-camera scorer; predictions never receive future camera poses."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_points import sample_window


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_errors(predicted, target, w2c, intrinsic):
    """All labels retained, including predictions behind the endpoint camera."""
    camera_target = target @ w2c[:3, :3].T + w2c[:3, 3]
    camera_pred = predicted @ w2c[:3, :3].T + w2c[:3, 3]
    if not np.isfinite(camera_target).all() or not np.isfinite(camera_pred).all() or (camera_target[:, 2] <= 0).any():
        raise ValueError('nonfinite forecast or invalid scored target; no denominator change')
    if not np.allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError('camera rotation not rigid')
    error = camera_pred - camera_target
    ray = camera_target / np.linalg.norm(camera_target, axis=1, keepdims=True)
    radial = (error * ray).sum(1)
    transverse = error - radial[:, None] * ray
    valid_projection = camera_pred[:, 2] > 1e-6
    projected_error = np.zeros(len(target), dtype='float64')
    if valid_projection.any():
        a, b = camera_pred[valid_projection] @ intrinsic.T, camera_target[valid_projection] @ intrinsic.T
        projected_error[valid_projection] = np.linalg.norm(a[:, :2]/a[:, 2:] - b[:, :2]/b[:, 2:], axis=1)
    return dict(world_epe=np.linalg.norm(error, axis=1), radial_abs=np.abs(radial),
                transverse=np.linalg.norm(transverse, axis=1), radial_sq=radial**2,
                total_sq=(error**2).sum(1), projected_epe=projected_error,
                invalid_projection=~valid_projection)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if manifest['status'] != 'QUALIFIED_VIDEO_PROBE_ONLY' or manifest['protocol']['supervised_horizon'] != 8:
        raise ValueError('fixed qualified h8 view only')
    for entry in manifest['sequences']:
        if sha(args.data / entry['file']) != entry['sha256'] or sha(args.raw / entry['scene'] / 'spatracker.npz') != entry['source_sha256']['spatracker.npz']:
            raise ValueError('point/camera source hash drift')
    model = None
    result = dict(schema='ref2dex.video-common-camera-error.v1', horizon=8, metrics={},
        data_manifest_sha256=sha(args.data/'manifest.json'), script_sha256=sha(__file__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        future_camera_access='endpoint scorer ONLY; all prediction inputs are observed HISTORY',
        invalid_prediction_policy='retain world/radial errors and full denominator; full-support projected EPE is null if any prediction behind camera',
        gpu_samples=[])
    if args.checkpoint:
        import torch
        from oakink_wm.video_points import collate_video
        from oakink_wm.video_pointworld import VideoPointWorldWM
        from oakink_wm.pointworld import VENDOR
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if ck['identity']['data_manifest_sha256'] != result['data_manifest_sha256']:
            raise ValueError('checkpoint data identity differs')
        for name, expected in ck['identity']['implementation_sha256'].items():
            if sha(TASK/name) != expected:
                raise ValueError('checkpoint implementation drift')
        for name, expected in ck['identity']['vendor_sources_sha256'].items():
            if sha(VENDOR/name) != expected:
                raise ValueError('checkpoint vendor drift')
        stats = ck['identity']['normalization']
        model = VideoPointWorldWM(stats['scene_mean'],stats['scene_std']).cuda().eval().requires_grad_(False)
        model.load_state_dict(ck['model'],strict=True)
        result['checkpoint_sha256'] = sha(args.checkpoint)
        result['checkpoint_identity'] = ck['identity']
        result['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        del ck
    started, last_sample = time.monotonic(), -10.
    for e in manifest['sequences']:
        with np.load(args.raw/e['scene']/'spatracker.npz',allow_pickle=False) as z:
            ks, ex = z['intrinsics'].astype(float),z['extrinsics'].astype(float)
        w2cs = ex if e['camera_inference']['convention']=='w2c' else np.linalg.inv(ex)
        totals = {}
        with np.load(args.data/e['file'],allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in ('points','valid','kind','timestamps','source_frame_ids')}
        for block in e['window_starts']:
            points, valid, kind, ts, frame_ids = [arrays[k][block] for k in ('points','valid','kind','timestamps','source_frame_ids')]
            s = sample_window(points,valid,kind,ts,0)
            selected = (s['point_kind']==1)&s['target_valid'][:,7]
            ids = s['selected_track_ids'][selected]
            if not len(ids):
                continue
            row = int(frame_ids[11]-e['source_range'][0])
            if row<0 or row>=len(ks):
                raise ValueError('endpoint source camera row outside source')
            dt, elapsed = float(ts[3]-ts[2]),float(ts[11]-ts[3])
            if not np.allclose(np.diff(ts[:4]),dt,atol=1e-8,rtol=1e-6):
                raise ValueError('nonuniform HISTORY clock')
            ph = points[:4,ids].astype(float)
            predictions = dict(static=ph[3],last_two=ph[3]+(ph[3]-ph[2])*elapsed/dt,
                history_ols=ph[3]+(.3*(ph[3]-ph[0])+.1*(ph[2]-ph[1]))*elapsed/dt)
            if model is not None:
                batch = {k:v.cuda() for k,v in collate_video([s]).items()}
                observed = {k:batch[k] for k in ('xyz','features','point_valid')}
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    flow = model(observed)
                predictions['model'] = ph[3] + flow[0,selected,7].float().cpu().numpy()
                if torch.cuda.max_memory_allocated() > 2*1024**3:
                    raise RuntimeError('2GiB CUDA allocation cap')
            for name,pred in predictions.items():
                errors = endpoint_errors(pred,points[11,ids].astype(float),w2cs[row],ks[row])
                total = totals.setdefault(name,dict(support=0,**{k:0. for k in errors}))
                total['support'] += len(ids)
                for key,value in errors.items():
                    total[key] += float(value.sum())
            elapsed_run = time.monotonic()-started
            if elapsed_run > (90 if model is not None else 30):
                raise TimeoutError('fixed evaluation time budget')
            if model is not None and elapsed_run-last_sample>=10:
                monitor = dict(elapsed_seconds=elapsed_run,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                    nvml_gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines())
                result['gpu_samples'].append(monitor)
                print(json.dumps(monitor),flush=True)
                last_sample=elapsed_run
        result['metrics'][e['split']+'/'+e['scene']]={name:dict(support=t['support'],world_epe_m=t['world_epe']/t['support'],
            radial_abs_m=t['radial_abs']/t['support'],transverse_m=t['transverse']/t['support'],
            radial_error_energy_fraction=t['radial_sq']/max(t['total_sq'],1e-20),
            projected_epe_px=t['projected_epe']/t['support'] if not t['invalid_projection'] else None,
            invalid_projection_count=int(t['invalid_projection'])) for name,t in totals.items()}
    dev=[v for k,v in result['metrics'].items() if k.startswith('dev/') and v.get('static',{}).get('support',0)>=16]
    result['decision_signal']=('RADIAL_CONCENTRATED_HISTORY_ERROR' if len(dev)>=2 and all(
        v['history_ols']['radial_error_energy_fraction']>=.7 and v['history_ols']['projected_epe_px'] is not None and
        v['history_ols']['projected_epe_px']<=.9*v['static']['projected_epe_px'] for v in dev) else
        'REPROJECTION_HISTORY_ERROR' if len(dev)>=2 and all(v['history_ols']['projected_epe_px'] is None or
        v['history_ols']['projected_epe_px']>=1.1*v['static']['projected_epe_px'] for v in dev) else 'UNCLEAR')
    result.update(elapsed_seconds=time.monotonic()-started,
        limitations=['inferred depth/camera are not GT; radial/transverse decomposition is an error diagnostic, not sensor accuracy',
            'future camera/ray serve only the label-side scorer; no future pose for prediction',
            'pixels and radial energy use full same h8 point-window support, no optimistic invalid-forecast deletion',
            'no new training, altered3D gate, native corpus promotion or physical/robot claim'])
    if model is not None:
        result['peak_memory_mib']=torch.cuda.max_memory_allocated()/1024**2
    serialized = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if len(serialized.encode()) > 1024**2:
        raise RuntimeError('1MiB audit JSON cap')
    args.output.write_text(serialized)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','checkpoint_identity','gpu_samples')},indent=2))


if __name__=='__main__':
    main()
