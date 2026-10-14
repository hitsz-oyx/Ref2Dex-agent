"""Static native geometry review; no video or simulation replay."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from pose_mesh_data import PoseMeshWindows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-id', type=int, help='review only this source; missing dev is displayed explicitly')
    a = p.parse_args()
    fig = plt.figure(figsize=(12,5))
    for panel, split in enumerate(('train','dev'),1):
        data = PoseMeshWindows(a.root,split,engineering=True)
        ax = fig.add_subplot(1,2,panel,projection='3d')
        choices = [i for i, (row, _) in enumerate(data.index)
                   if a.source_id is None or data.meta['sequences'][row]['source_id'] == a.source_id]
        if not choices:
            ax.set_title(f'{split}: no independent source{a.source_id} sequence')
            continue
        sample = data[choices[len(choices)//2]]
        ax.scatter(*sample['xyz'].T,s=1,c='gray',alpha=.4,label='current CAD')
        target = sample['start_object_points']
        ax.scatter(*target.T,s=7,c='orange',label='target current')
        future = sample['goal_object_points'][sample['object_goal_valid']]
        ax.scatter(*future.T,s=7,c='red',label='target +0.8s')
        for side, name, color in [(0,'right','blue'),(1,'left','purple')]:
            valid = sample['start_hand_valid'][side]
            hand = sample['start_hand'][side]
            ax.scatter(*hand[valid].T,s=15,c=color,label=name+' current')
            for finger in range(5):
                ids = [0,1+finger,6+finger]
                if valid[ids].all():
                    ax.plot(*hand[ids].T,color=color,lw=1)
            known = sample['trajectory_valid'][-1,side]
            future_hand = hand+sample['trajectory_target'][-1,side]
            ax.scatter(*future_hand[known].T,s=14,c=color,marker='x',label=name+' +0.8s')
        cloud = sample['xyz']
        center = (cloud.min(0)+cloud.max(0))/2
        radius = max(np.ptp(cloud,axis=0).max()/2,.1)
        for setter, value in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
            setter(value-radius,value+radius)
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('x / m'); ax.set_ylabel('y / m'); ax.set_zlabel('z / m')
        ax.set_title(f'{split}, source{int(sample["source_id"])}, current target frame')
        ax.legend(fontsize=6)
    fig.suptitle('Native 3D engineering review: CAD / hand identity / future labels')
    fig.tight_layout()
    filename = 'geometry-review.png' if a.source_id is None else f'geometry-review-source{a.source_id}.png'
    fig.savefig(a.root/filename,dpi=160)


if __name__ == '__main__':
    main()
