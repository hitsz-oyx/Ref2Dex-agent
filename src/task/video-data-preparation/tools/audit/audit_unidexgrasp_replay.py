#!/usr/bin/env python3
"""Audit recorded simulation clock/commands and export provisional H4/K24 fixtures."""
import argparse
import json
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def poses(values):
    if np.max(abs(np.linalg.norm(values[...,3:7],axis=-1)-1)) > 1e-4:
        raise ValueError('recorded object quaternion is not unit length')
    out = np.tile(np.eye(4), (*values.shape[:-1],1,1))
    out[...,:3,:3] = Rotation.from_quat(values[...,3:7].reshape(-1,4)).as_matrix().reshape(*values.shape[:-1],3,3)
    out[...,:3,3] = values[...,:3]
    return out


def fixture(hand, obj, surface, tick):
    anchor = np.linalg.inv(obj[tick])
    hh = (hand[tick-3:tick+25]-obj[tick,:3,3]) @ obj[tick,:3,:3]
    start = np.zeros((2,11,3),np.float32);start[0] = hh[3]
    valid = np.zeros((2,11),bool);valid[0] = True
    history = np.zeros((2,11,9),np.float32)
    history[0] = np.concatenate(((hh[3]-hh[2])*30,(hh[3]-2*hh[2]+hh[1])*900,hh[3]-hh[0]),-1)
    future = np.zeros((24,2,11,3),np.float32);future[:,0] = hh[4:]-hh[3]
    rel = anchor @ obj[tick+np.arange(1,25)]
    query = surface[np.linspace(0,len(surface)-1,32,dtype=int)].copy()
    moving = np.einsum('tij,pj->tpi',rel[:,:3,:3],query)+rel[:,None,:3,3]
    target = (moving-query).astype(np.float32)
    # Current complete CAD and no estimated camera are an explicit sim profile.
    return dict(start_hand=start,start_hand_history=history,start_hand_valid=valid,
                trajectory_target=future,trajectory_valid=np.broadcast_to(valid,(24,2,11)).copy(),
                scene_points=query[None].astype(np.float32),scene_target=target[None],
                scene_valid=np.ones((1,24,32),bool),start_object_points=query.astype(np.float32),
                goal_object_points=moving[-1].astype(np.float32),object_goal_valid=np.ones(32,bool),
                current_cad_points=surface.astype(np.float32),anchor_world_to_current_object=anchor,
                source_profile=np.array('unidexgrasp-shadow-complete-cad-provisional'),
                training_allowed=np.array(False))


def view(data, env, tick, vertices, axis, label):
    pose = poses(data['object'][tick,env])
    cloud = vertices @ pose[:3,:3].T + pose[:3,3]
    hand = data['hand'][tick,env]
    axis.scatter(*cloud.T,s=2,color='#3b74b2',alpha=.5)
    for i in range(5):
        chain = hand[[0,1+i,6+i]]
        axis.plot(*chain.T,marker='o',linewidth=2,markersize=4)
    xx,yy=np.meshgrid([-.23,.23],[-.23,.23])
    axis.plot_surface(xx,yy,np.full_like(xx,.6),color='gray',alpha=.12)
    axis.set(xlim=(-.25,.25),ylim=(-.25,.25),zlim=(.55,1.15),
             xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    axis.set_box_aspect((1,1,1.2));axis.view_init(elev=24,azim=45)
    axis.set_title('%s | %.2fs | recorded simulation geometry' % (label,data['time'][tick]))


def render(data, env, vertices, root, label):
    fig=plt.figure(figsize=(12,10))
    for i,tick in enumerate([0,60,120,198]):
        view(data,env,tick,vertices,fig.add_subplot(2,2,i+1,projection='3d'),label)
    fig.tight_layout();fig.savefig(root/('review-'+label+'.jpg'),dpi=120);plt.close(fig)
    fig=plt.figure(figsize=(6.4,6.4),dpi=100)
    axis=fig.add_subplot(projection='3d')
    writer=cv2.VideoWriter(str(root/('replay-'+label+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,640))
    if not writer.isOpened():raise IOError('failed to open bounded MP4 writer')
    try:
        for tick in range(0,len(data['time']),3):
            axis.clear();view(data,env,tick,vertices,axis,label)
            fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba())[...,:3]
            writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    finally:
        writer.release();plt.close(fig)


def main(root):
    begin=time.monotonic()
    result=json.loads((root/'result.json').read_text())
    config=json.loads((root/'simulation-config.json').read_text())
    if (result['status']!='COMPLETE' or config['substeps']!=2 or
            config['physx']['num_position_iterations']!=8 or
            not np.isclose(config['physx']['contact_offset'],.002)):
        raise ValueError('completed replay with official physical settings required')
    asset=Path(result['asset_root'])/'meshdatav3_scaled'/result['object_code']/'coacd'
    suffix='%03d'%round(result['scale']*100)
    mesh=trimesh.load_mesh(asset/('decomposed_'+suffix+'.obj'),process=False)
    rng=np.random.RandomState(0)
    surface=mesh.vertices[np.linspace(0,len(mesh.vertices)-1,512,dtype=int)]
    records=[];samples=[];representatives={}
    for episode,row in enumerate(result['episodes']):
        with np.load(root/row['path'],allow_pickle=False) as stream:
            data={key:stream[key].copy() for key in stream.files}
        frames,envs=data['hand'].shape[:2]
        if data['hand'].shape!=(frames,envs,11,3) or data['observation'].shape!=(frames,envs,300):
            raise ValueError('recorded hand/observation contract mismatch')
        if frames!=200 or not np.isfinite(data['hand']).all() or not np.isfinite(data['object']).all():
            raise ValueError('incomplete or nonfinite episode')
        if data['action'].shape!=(frames-1,envs,24) or not np.allclose(data['action'][...,6:],data['applied'][...,6:],atol=1e-6):
            raise ValueError('policy actuator commands differ from executed commands')
        if not np.allclose(np.diff(data['time']),1/60,atol=1e-7) or data['reset'][:-1].any():
            raise ValueError('episode clock/reset discontinuity')
        if not np.all(data['observation'][...,207:236]==0):
            raise ValueError('state checkpoint goal observation mismatch')
        # Only actual recorded frames are selected: no time/pose interpolation.
        indices=np.arange(0,frames,2)
        clock=data['time'][indices]
        if not np.allclose(np.diff(clock),1/30,atol=1e-7):raise ValueError('30Hz selection failed')
        object_poses=poses(data['object'][indices])
        anchors=np.arange(3,len(indices)-24,8)
        success=data['success'][-1]>0
        for env in range(envs):
            for tick in anchors:
                sample=fixture(data['hand'][indices,env],object_poses[:,env],surface,int(tick))
                if not np.allclose(sample['goal_object_points']-sample['start_object_points'],sample['scene_target'][0,-1],atol=2e-6):
                    raise ValueError('goal and label point identities differ')
                if sample['trajectory_valid'][:,1].any() or np.any(sample['start_hand'][1]!=0):
                    raise ValueError('missing left hand was fabricated')
                if int(tick)==27:
                    samples.append((episode,env,sample))
            label='success' if success[env] else 'failure'
            representatives.setdefault(label,(data,env))
        records.append(dict(episode=episode,envs=envs,raw_frames=frames,frames_30hz=len(indices),
                            anchors_per_env=len(anchors),official_success=int(success.sum()),
                            finite=True,clock_pass=True,no_internal_reset=True))
    # Export one fixed-time real window per recorded environment, including failures.
    dest=root/'windows';dest.mkdir(exist_ok=True)
    for episode,env,sample in samples:
        np.savez_compressed(dest/('episode-%02d-env-%02d.npz'%(episode,env)),**sample)
    for label,(data,env) in representatives.items():
        render(data,env,surface,root,label)
    summary=dict(status='COMPLETE',episodes=records,official_success=sum(r['official_success'] for r in records),
                 total_episodes=sum(r['envs'] for r in records),
                 hand_object_candidate_windows=sum(r['envs']*r['anchors_per_env'] for r in records),
                 exported_engineering_windows=len(samples),training_admitted=0,training_allowed=False,
                 fixture_shapes={key:list(value.shape) for key,value in samples[0][2].items()},
                 elapsed_s=time.monotonic()-begin,
                 visualization='CPU reconstruction of recorded hand landmarks/object poses; not camera RGB',
                 limitations=['single object/seed; no object-disjoint dev','pose-complete CAD profile',
                              'official success is goal proximity flag, not independently validated stable grasp'])
    (root/'audit-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run.resolve())
