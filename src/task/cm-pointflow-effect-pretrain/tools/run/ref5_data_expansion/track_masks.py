"""Small free SAM2 object-mask stage; native hands are never reconstructed.

Pilot annotations are explicit clicks on an inspected rigid target. This
bypasses the expensive automatic EgoHOS/VLM selection stages, retaining the
ObjectForesight SAM2 -> geometry -> depth -> FoundationPose route.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--point', type=float, nargs=2, required=True, help='click coordinates in original 1920x1080 video')
    p.add_argument('--box', type=float, nargs=4, help='whole-object box in original pixels; disambiguates small texture patches')
    p.add_argument('--prompt-frame', type=int, default=0,
                   help='native frame where the target is visible; use original SAM2 reverse propagation for earlier frames')
    p.add_argument('--seconds', type=int, default=300)
    p.add_argument('--max-frames', type=int, default=300)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(a.video))
    raw_width, raw_height = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    if n > a.max_frames or n < 28 or abs(fps-30) > 1e-3:
        raise ValueError('pilot length/clock outside bounds')
    if not 0 <= a.prompt_frame < n:
        raise ValueError('prompt frame outside source video')
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            raise ValueError('video decode failed')
        frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)))
    cap.release()
    device = 'cuda:0'
    model = Sam2VideoModel.from_pretrained(str(a.model), local_files_only=True).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(str(a.model), local_files_only=True)
    session = processor.init_video_session(video=frames, inference_device=device,
                                           processing_device='cpu', video_storage_device='cpu')
    point = [a.point[0]*640/raw_width, a.point[1]*360/raw_height]
    box = ([a.box[0]*640/raw_width, a.box[1]*360/raw_height,
            a.box[2]*640/raw_width, a.box[3]*360/raw_height] if a.box else None)
    processor.add_inputs_to_inference_session(inference_session=session, frame_idx=a.prompt_frame, obj_ids=1,
                                              input_points=[[[point]]], input_labels=[[[1]]],
                                              input_boxes=[[box]] if box else None)
    masks = [None] * n
    with torch.inference_mode():
        model(inference_session=session, frame_idx=a.prompt_frame)
        for reverse in ([False, True] if a.prompt_frame else [False]):
            for outputs in model.propagate_in_video_iterator(
                    session, start_frame_idx=a.prompt_frame, reverse=reverse):
                if time.monotonic()-start > a.seconds:
                    raise TimeoutError('mask stage deadline')
                fid = int(outputs.frame_idx)
                if not 0 <= fid < n:
                    raise ValueError('SAM2 returned an invalid native frame')
                if masks[fid] is not None:
                    continue  # The annotated frame occurs in both directions.
                mask = processor.post_process_masks([outputs.pred_masks], original_sizes=[[360, 640]],
                                                     binarize=False)[0].reshape(360, 640).cpu().numpy() > 0
                masks[fid] = mask
                if fid % 30 == 0:
                    print(json.dumps({'frame': fid, 'area': int(mask.sum()), 'reverse': reverse}), flush=True)
    if any(mask is None for mask in masks):
        raise ValueError('tracking did not preserve all frame identities')
    areas = [int(mask.sum()) for mask in masks]
    tiles = []
    for fid in sorted(set([0, n//4, n//2, 3*n//4, n-1, a.prompt_frame])):
        im = np.asarray(frames[fid]).copy()
        im[masks[fid]] = (.5*im[masks[fid]] + .5*np.array([255, 50, 50])).astype('uint8')
        cv2.putText(im, f'frame {fid} area {areas[fid]}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
        tiles.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    np.savez_compressed(a.output/'masks.npz', masks=np.stack(masks), source_frame_ids=np.arange(n),
                        source_image_shape=[raw_height,raw_width], mask_image_shape=[360,640])
    cv2.imwrite(str(a.output/'mask_review.jpg'), np.concatenate(tiles, axis=1))
    report = dict(status='COMPLETED', stage='SAM2_MASK_ONLY', source_video=str(a.video.resolve()),
                  frames=n, fps=fps, manual_initial_prompt_original_pixels=a.point,
                  manual_initial_box_original_pixels=a.box,
                  manual_prompt_source_frame=a.prompt_frame,
                  propagation_directions=['forward','reverse'] if a.prompt_frame else ['forward'],
                  areas=areas, nonempty_fraction=float(np.mean(np.asarray(areas)>0)),
                  object_6dof_available=False, elapsed_s=time.monotonic()-start)
    (a.output/'mask_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','frames','nonempty_fraction','elapsed_s')}))


if __name__ == '__main__':
    main()
