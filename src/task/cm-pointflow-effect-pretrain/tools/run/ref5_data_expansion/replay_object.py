"""Export native RGB/hand projection and stationary-origin estimated-object replay."""
import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from prepare_egodex_hands import native_hands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--depth', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    os.environ.setdefault('MPLCONFIGDIR', str(Path('tmp/ref5-data-expansion/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(a.input/'object_poses.npz') as f:
        ids = f['source_frame_ids']; poses = f['poses'][:, 0]
        valid = f['pose_valid'][:, 0]; iou = f['iou']
    with np.load(a.depth) as d:
        camera = d['camera_to_origin']; K = d['intrinsics']
        H, W = d['depth'].shape[1:]
    with h5py.File(a.hdf5) as f:
        hands, hvalid, *_ = native_hands(f)
        hands, hvalid = hands[ids], hvalid[ids]
    with np.load(next((a.input/'processed/canonical').glob('*.npz'))) as f:
        canonical = f['points']
    cloud = np.einsum('tij,nj->tni', poses[:, :3, :3], canonical) + poses[:, None, :3, 3]
    points = np.concatenate((hands[hvalid], cloud[valid].reshape(-1, 3)), axis=0)
    center = (points.min(0)+points.max(0))/2
    extent = max(.1, np.ptp(points, axis=0).max()/2)*1.05
    cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W*2, H))
    if not writer.isOpened():
        raise RuntimeError('MP4 writer unavailable')
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    try:
        for t, fid in enumerate(ids):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('native frame decode failed')
            bgr = cv2.resize(bgr, (W, H))
            inv = np.linalg.inv(camera[t])
            hp = hands[t] @ inv[:3, :3].T + inv[:3, 3]
            uv = hp @ K.T; uv = uv[..., :2]/uv[..., 2:3]
            ax.clear()
            for side, color, bgr_color in [(0, 'green', (0, 255, 0)), (1, 'blue', (255, 100, 0))]:
                for j in np.flatnonzero(hvalid[t, side]):
                    if hp[side, j, 2] > 0 and np.isfinite(uv[side, j]).all():
                        u, v = np.rint(uv[side, j]).astype(int)
                        if 0 <= u < W and 0 <= v < H:
                            cv2.circle(bgr, (u, v), 2, bgr_color, -1)
                h = hands[t, side]; keep = hvalid[t, side]
                ax.scatter(*h[keep].T, s=6, c=color)
                for mcp, tip in zip(range(1, 6), range(6, 11)):
                    if keep[[0, mcp, tip]].all():
                        ax.plot(*h[[0, mcp, tip]].T, c=color, lw=.7)
            if valid[t]:
                ax.scatter(*cloud[t].T, s=1, c='orange')
                pc = cloud[t] @ inv[:3, :3].T + inv[:3, 3]
                puv = pc @ K.T; puv = puv[:, :2]/puv[:, 2:3]
                for point in puv[np.isfinite(puv).all(-1)]:
                    u, v = np.rint(point).astype(int)
                    if 0 <= u < W and 0 <= v < H:
                        bgr[v, u] = (0, 170, 255)
            for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setter(center[i]-extent, center[i]+extent)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -65)
            ax.set_xlabel('x (m)', fontsize=7); ax.set_ylabel('y (m)', fontsize=7); ax.set_zlabel('z (m)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title('Native ARKit origin; orange = estimated object', fontsize=8)
            fig.canvas.draw()
            canvas = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            canvas = cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f'frame {fid} /30Hz IoU {iou[t]:.2f} valid {bool(valid[t])}',
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            writer.write(np.concatenate((bgr, canvas), axis=1))
    finally:
        cap.release(); writer.release(); plt.close(fig)
    print(a.output, flush=True)


if __name__ == '__main__':
    main()
