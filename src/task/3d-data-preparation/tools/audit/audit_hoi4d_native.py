"""Audit native HOI4D rigid geometry and draw current-frame CAD/hand overlays without a fake clock."""
import argparse
import collections
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.root/'manifest.json').read_text())
    if meta['schema']!='ref2dex.native-pose-mesh.v1' or meta['training_allowed'] is not False:raise ValueError('native provisional source required')
    objects={v['key']:v for v in meta['objects']};groups=collections.defaultdict(set)
    counts=collections.Counter();labels=collections.Counter();ratios=[];frames=0;valid_hands=0;valid_poses=0;max_error=0;choices={}
    for row in meta['sequences']:
        groups[row['split']].add(row['group_id']);counts[row['split']]+=1;labels[row['clock_status']]+=1;ratios.append(row['bbox_to_cad_extent_ratio_median'])
        with np.load(a.root/row['path'],allow_pickle=False) as z:
            assert 'time' not in z.files
            ids=z['native_frame_id'];hand=z['hand'];hv=z['hand_valid'];poses=z['object_poses'];pv=z['object_pose_valid']
            assert np.all(np.diff(ids)==1) and np.isfinite(hand[hv]).all()
            current=poses[pv];rot=current[:,:3,:3]
            np.testing.assert_allclose(rot.transpose(0,2,1)@rot,np.broadcast_to(np.eye(3),rot.shape),atol=2e-4)
            np.testing.assert_allclose(np.linalg.det(rot),1,atol=2e-4)
            np.testing.assert_allclose(current[:,3],np.broadcast_to([0,0,0,1],current[:,3].shape),atol=1e-6)
            good=np.flatnonzero(pv[:,0]&z['native_camera_valid'])
            for k in good[::50]:
                recovered=np.linalg.inv(z['native_camera_to_world'][k])@poses[k,0]
                max_error=max(max_error,float(abs(recovered-z['native_object_pose_camera'][k]).max()))
            frames+=len(ids);valid_hands+=int(hv.any(-1).sum());valid_poses+=int(pv.sum())
            category=row['objects'][0].split('/')[1]
            both=np.flatnonzero(pv[:,0]&hv.any(axis=(1,2)))
            if category not in choices and len(both):
                k=int(both[len(both)//2]);choices[category]=(row,k,hand[k].copy(),hv[k].copy(),poses[k,0].copy(),int(ids[k]))
    assert not groups['train']&groups['dev'];assert max_error<1e-5
    report=dict(status='NATIVE_GEOMETRY_ENGINEERING_PASS',training_allowed=False,sequences=len(meta['sequences']),frames=frames,
        split_sequences=dict(counts),clock_status_counts=dict(labels),known_hand_frames=valid_hands,known_object_frames=valid_poses,
        world_camera_object_composition_max_error=max_error,object_instance_split_disjoint=True,
        bbox_to_cad_extent_ratio_quantiles=np.quantile(np.array(ratios),[0,.5,.95,1],axis=0).tolist(),
        limits=['Raw frame geometry only, no uniform30Hz windows admitted.','Missing labels remain invalid; original metric CAD retained.'],checked_at=time.time())
    (a.root/'native-geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    selection=[v for key,v in sorted(choices.items())][:4];fig=plt.figure(figsize=(15,4*int(np.ceil(len(selection)/2))))
    for panel,(row,k,hand,hv,pose,frame) in enumerate(selection,1):
        ax=fig.add_subplot(int(np.ceil(len(selection)/2)),2,panel,projection='3d')
        with np.load(a.root/objects[row['objects'][0]]['path'],allow_pickle=False) as z:surface=z['surface']
        anchor=np.linalg.inv(pose);hand=hand@anchor[:3,:3].T+anchor[:3,3]
        ax.scatter(*surface.T,c='gray',s=2,alpha=.5,label='original metric CAD')
        visible=[surface]
        for side,color in [(0,'blue'),(1,'purple')]:
            ax.scatter(*hand[side,hv[side]].T,c=color,s=14,label=('right' if side==0 else 'left')+' native hand')
            visible.append(hand[side,hv[side]])
            for finger in range(5):
                q=[0,1+finger,6+finger]
                if hv[side,q].all():ax.plot(*hand[side,q].T,c=color,lw=1)
        cloud=np.concatenate(visible);center=(cloud.min(0)+cloud.max(0))/2;radius=max(np.ptp(cloud,axis=0).max()/2,.1)
        for setter,value in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setter(value-radius,value+radius)
        ax.set_box_aspect([1,1,1]);ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_zlabel('z / m')
        ax.set_title(row['objects'][0]+f', native frame {frame}');ax.legend(fontsize=7)
    fig.suptitle('HOI4D native CAD / hand review: current object frame; no resampled clock');fig.tight_layout()
    fig.savefig(a.root/'native-geometry-review.png',dpi=160);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
