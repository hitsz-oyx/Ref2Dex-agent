"""ObjectForesight's SpaTracker offline refinement, keeping native calibration.

Predicted/refined camera poses are saved for diagnostics and never replace
EgoDex camera-to-origin in downstream object/hand coordinate composition.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from native_data import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--depth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(2)
    sys.path.insert(0,str(a.root/'vendor/ObjectForesight-Data/SpaTrackerV2'))
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.utils import get_points_on_a_grid
    report=dict(status='RUNNING',stage='SPATRACKER_OFFLINE',training_allowed=False,
                depth_sha256=sha(a.depth),script_sha256=sha(Path(__file__)))
    try:
        d=np.load(a.depth);ids=d['source_frame_ids'];n=len(ids)
        if not 28<=n<=64 or not np.all(np.diff(ids)==1):raise ValueError('bounded contiguous frames required')
        H,W=d['depth'].shape[1:];cap=cv2.VideoCapture(str(a.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(ids[0]));frames=[]
        for _ in ids:
            ok,f=cap.read()
            if not ok:raise ValueError('video decode failed')
            frames.append(cv2.cvtColor(cv2.resize(f,(W,H)),cv2.COLOR_BGR2RGB))
        cap.release();video=torch.from_numpy(np.stack(frames)).permute(0,3,1,2).float()
        model=Predictor.from_pretrained(str(a.root/'models/spatracker-offline'),local_files_only=True)
        model.spatrack.track_num=100;model.eval();model.to('cuda')
        pts=get_points_on_a_grid(8,(H,W),device='cpu')
        queries=torch.cat((torch.zeros_like(pts[:,:,:1]),pts),dim=2)[0].numpy()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            ret=model.forward(video,depth=d['depth'],unc_metric=d['confidence']>.5,
                              intrs=np.repeat(d['intrinsics'][None],n,axis=0),
                              extrs=d['camera_to_origin'],queries=queries,fps=1,
                              full_point=False,iters_track=4,query_no_BA=True,
                              fixed_cam=True,stage=1,support_frame=n-1,replace_ratio=.2)
        cams,intrs,point_map,confidence,tracks,_,visibility,*_=ret
        depth=point_map[:n,2].float().cpu().numpy()
        confidence=confidence[:n].float().cpu().numpy()
        if confidence.shape!=depth.shape:raise ValueError('unexpected depth confidence shape')
        depth[~np.isfinite(depth)|(confidence<.5)]=0
        if np.mean(depth>0)<.1:raise ValueError('insufficient refined depth coverage')
        arrays={k:d[k] for k in d.files}
        arrays.update(depth=depth,confidence=confidence,
                      refined_camera=cams[:n].float().cpu().numpy(),
                      refined_intrinsics=intrs[:n].float().cpu().numpy(),
                      auxiliary_tracks=tracks[:n].float().cpu().numpy(),
                      auxiliary_visibility=visibility[:n].float().cpu().numpy())
        np.savez_compressed(a.output/'depth.npz',**arrays)
        report.update(status='COMPLETED',frames=n,valid_depth_fraction=float(np.mean(depth>0)),
                      native_camera_preserved=True,fixed_camera_refinement=True,
                      clock_note='tensor input has no temporal resampling; fps is ignored for tensors',
                      refined_depth_sha256=sha(a.output/'depth.npz'))
    except Exception as e:
        report.update(status='FAILED',error=repr(e));raise
    finally:
        report['elapsed_s']=time.monotonic()-start
        (a.output/'refinement_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
