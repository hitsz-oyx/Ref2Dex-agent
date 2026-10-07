"""Small SpaTrackerV2 depth probe with native EgoDex camera calibration.

Native camera poses always govern hand/object composition. Predicted camera
poses are diagnostics only. Estimated depth scale is audited against native
hand joint depths, whose surface-depth offset remains an approximation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from native_data import sha
from prepare_egodex_hands import native_hands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--hdf5',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=96)
    p.add_argument('--frames',type=int,default=48)
    a=p.parse_args()
    if not 28 <= a.frames <= 64:
        p.error('frames must be 28..64')
    a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(2)
    sys.path.insert(0,str(a.root/'vendor/ObjectForesight-Data/SpaTrackerV2'))
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    report=dict(status='RUNNING',stage='SPATRACKER_DEPTH',fps=30,
                metric_ground_truth=False,training_allowed=False,
                script_sha256=sha(Path(__file__)),video_sha256=sha(a.video),
                hdf5_sha256=sha(a.hdf5))
    try:
        ids=np.arange(a.start_frame,a.start_frame+a.frames)
        cap=cv2.VideoCapture(str(a.video))
        if abs(cap.get(cv2.CAP_PROP_FPS)-30)>1e-3:
            raise ValueError('expected original 30 Hz video')
        cap.set(cv2.CAP_PROP_POS_FRAMES,a.start_frame);rgb=[]
        for _ in ids:
            ok,f=cap.read()
            if not ok:raise ValueError('frame decode failed')
            rgb.append(cv2.cvtColor(cv2.resize(f,(518,294)),cv2.COLOR_BGR2RGB))
        cap.release();rgb=np.stack(rgb)
        with h5py.File(a.hdf5,'r') as hf:
            camera=hf['transforms/camera'][ids]
            K=hf['camera/intrinsic'][:].copy()
            hands,valid,conf,known,names=native_hands(hf)
            hands,valid=hands[ids],valid[ids]
        if K.shape!=(3,3):raise ValueError(f'unexpected intrinsics {K.shape}')
        K[0]*=518/1920;K[1]*=294/1080
        video=torch.from_numpy(rgb).permute(0,3,1,2).float()[None]/255
        model=VGGT4Track.from_pretrained(str(a.root/'models/spatracker-front'),
                                        local_files_only=True).eval().cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(video.cuda())
        depth=pred['points_map'][...,2].float().cpu().numpy().reshape(a.frames,294,518)
        confidence=pred['unc_metric'].float().cpu().numpy().reshape(depth.shape)
        pred_camera=pred['poses_pred'].float().cpu().numpy().reshape(a.frames,4,4)
        pred_K=pred['intrs'].float().cpu().numpy().reshape(a.frames,3,3)
        del model,pred;torch.cuda.empty_cache()
        # Native joints are in the stationary origin. Project into known camera.
        inv=np.linalg.inv(camera)
        xyz=np.einsum('tij,thpj->thpi',inv[:,:3,:3],hands)+inv[:,:3,3,None,None].transpose(0,2,3,1)
        uv=xyz@K.T;uv=uv[...,:2]/uv[...,2:3]
        ratios=[];calib_frames=min(8,a.frames)
        for t in range(calib_frames):
            for h,j in np.argwhere(valid[t]):
                u,v=np.rint(uv[t,h,j]).astype(int)
                z=xyz[t,h,j,2]
                if 0<=u<518 and 0<=v<294 and z>0 and depth[t,v,u]>0 and confidence[t,v,u]>.5:
                    ratios.append(float(z/depth[t,v,u]))
        if len(ratios)<20:raise ValueError('insufficient native-hand/depth alignment samples')
        ratio=np.asarray(ratios);scale=float(np.median(ratio))
        spread=float((np.quantile(ratio,.75)-np.quantile(ratio,.25))/scale)
        if not .02<scale<50 or spread>.35:
            raise ValueError(f'unstable metric calibration: scale={scale} relative_IQR={spread}')
        scaled=depth*scale;scaled[~np.isfinite(scaled)|(confidence<.5)]=0
        np.savez_compressed(a.output/'depth.npz',depth=scaled,raw_depth=depth,
                            confidence=confidence,camera_to_origin=camera,
                            intrinsics=K,source_frame_ids=ids,timestamps=ids/30,
                            predicted_camera_to_world=pred_camera,predicted_intrinsics=pred_K)
        panels=[]
        for t in [0,a.frames//2,a.frames-1]:
            viz=cv2.applyColorMap(np.clip(scaled[t]/2*255,0,255).astype('uint8'),cv2.COLORMAP_TURBO)
            panels.append(np.concatenate((cv2.cvtColor(rgb[t],cv2.COLOR_RGB2BGR),viz),axis=0))
        cv2.imwrite(str(a.output/'depth_review.jpg'),np.concatenate(panels,axis=1))
        report.update(status='COMPLETED',frames=a.frames,first_source_frame=int(ids[0]),
                      last_source_frame=int(ids[-1]),scale_native_hand_over_predicted_depth=scale,
                      calibration_samples=len(ratio),calibration_relative_iqr=spread,
                      calibration_source_frames=ids[:calib_frames].tolist(),
                      scale_note='approximate joint/surface-depth calibration; requires visual alignment audit',
                      valid_depth_fraction=float(np.mean(scaled>0)),
                      depth_sha256=sha(a.output/'depth.npz'))
    except Exception as e:
        report.update(status='FAILED',error=repr(e));raise
    finally:
        report['elapsed_s']=time.monotonic()-start
        (a.output/'depth_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
