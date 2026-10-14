#!/usr/bin/env python3
"""Audit recorded simulation clock/commands and export provisional H4/K24 fixtures."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from video_data_preparation.unidexgrasp_data import object_poses as poses, core_window as fixture


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


def main(root, render_objects=1, export_windows=True):
    begin=time.monotonic()
    result=json.loads((root/'result.json').read_text())
    if 'User aggregate size exceeded' in (root/'replay.log').read_text():
        raise ValueError('aggregate capacity warnings: preserve as diagnostic, exclude from admission')
    config=json.loads((root/'simulation-config.json').read_text())
    if (result['status']!='COMPLETE' or config['substeps']!=2 or
            config['physx']['num_position_iterations']!=8 or
            not np.isclose(config['physx']['contact_offset'],.002)):
        raise ValueError('completed replay with official physical settings required')
    specs = result.get('objects') or [dict(object_code=result['object_code'], scale=result['scale'], split='engineering')]
    env_objects = result.get('environment_objects') or [dict(env=i,object_id=0,**specs[0]) for i in range(result['envs'])]
    if len(env_objects) != result['envs']:
        raise ValueError('simulator environment-object identities missing')
    surfaces={}
    geometry=root/'geometry';geometry.mkdir(exist_ok=True)
    for spec in specs:
        code,scale=spec['object_code'],spec['scale']
        asset=Path(result['asset_root'])/'meshdatav3_scaled'/code/'coacd'
        mesh=trimesh.load_mesh(asset/('decomposed_%03d.obj'%round(scale*100)),process=False)
        if len(mesh.vertices)<32 or not np.isfinite(mesh.vertices).all():
            raise ValueError('insufficient finite CAD vertices')
        surface=mesh.vertices[np.linspace(0,len(mesh.vertices)-1,min(512,len(mesh.vertices)),dtype=int)]
        # Meshes contain duplicated face vertices; query identities must be 32
        # distinct physical local points, shared by every trajectory of the mesh.
        surface=np.unique(surface,axis=0)
        if len(surface)<32:raise ValueError('insufficient distinct CAD query positions')
        vertex_ids={tuple(point):i for i,point in enumerate(mesh.vertices)}
        normals=mesh.vertex_normals[[vertex_ids[tuple(point)] for point in surface]]
        if normals.shape!=surface.shape or not np.isfinite(normals).all():
            raise ValueError('invalid current CAD normals')
        surfaces[code]=surface
        np.savez_compressed(geometry/('object-%03d.npz'%specs.index(spec)),surface=surface.astype(np.float32),
                            normals=normals.astype(np.float32),
                            query=surface[np.linspace(0,len(surface)-1,32,dtype=int)].astype(np.float32),
                            object_code=code,scale=scale)
    records=[];samples=[];representatives={};trajectories=[];counts={};causal_checked=set()
    compact=root/'trajectories';compact.mkdir(exist_ok=True)
    dest=root/'windows'
    if export_windows:dest.mkdir(exist_ok=True)
    for episode,row in enumerate(result['episodes']):
        with np.load(root/row['path'],allow_pickle=False) as stream:
            data={key:stream[key].copy() for key in stream.files}
        frames,envs=data['hand'].shape[:2]
        if data['hand'].shape!=(frames,envs,11,3) or data['observation'].shape!=(frames,envs,300):
            raise ValueError('recorded hand/observation contract mismatch')
        if frames!=200 or any(not np.isfinite(v).all() for v in data.values()):
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
            identity=env_objects[env];code=identity['object_code'];surface=surfaces[code]
            tick=27
            sample=fixture(data['hand'][indices,env],object_poses[:,env],surface,tick)
            if not np.allclose(sample['goal_object_points']-sample['start_object_points'],sample['scene_target'][0,-1],atol=2e-6):
                raise ValueError('goal and label point identities differ')
            if sample['trajectory_valid'][:,1].any() or np.any(sample['start_hand'][1]!=0):
                raise ValueError('missing left hand was fabricated')
            if len(np.unique(sample['start_object_points'],axis=0))!=32:
                raise ValueError('duplicate physical object queries')
            if code not in causal_checked:
                changed_hand=data['hand'][indices,env].copy();changed_obj=object_poses[:,env].copy()
                changed_hand[tick+1:]+=1000;changed_obj[tick+1:,:3,3]+=500
                altered=fixture(changed_hand,changed_obj,surface,tick)
                for key in ('start_hand','start_hand_history','start_hand_valid','scene_points',
                            'start_object_points','current_cad_points','anchor_world_to_current_object'):
                    if not np.array_equal(sample[key],altered[key]):
                        raise ValueError('future leaked into current input: '+key)
                if np.array_equal(sample['scene_target'],altered['scene_target']):
                    raise ValueError('future perturbation did not reach labels')
                causal_checked.add(code)
            if export_windows:
                np.savez_compressed(dest/('episode-%02d-env-%03d.npz'%(episode,env)),**sample)
            samples.append(dict(episode=episode,env=env,object_code=code,scale=identity['scale'],split=identity['split'],tick_30hz=tick))
            name='episode-%02d-env-%03d.npz'%(episode,env)
            values={key:data[key][indices,env] for key in ('hand','object','q','dq','hand_root','success','reset')}
            if 'goal' in data:values['goal']=data['goal'][indices,env]
            values.update(time=clock,action=data['action'][:,env],action_time=data['time'][:-1],
                          applied=data['applied'][:,env],official_success=bool(success[env]),
                          terminal_time=data['time'][-1],terminal_reset=data['reset'][-1,env],
                          object_code=code,scale=identity['scale'],split=identity['split'],training_allowed=False)
            np.savez_compressed(compact/name,**values)
            end_distance=float(np.linalg.norm(data['observation'][-1,env,204:207]))
            if 'goal' in data and not np.isclose(end_distance,np.linalg.norm(data['goal'][-1,env]-data['object'][-1,env,:3]),atol=1e-5):
                raise ValueError('independent endpoint distance does not match observation')
            trajectories.append(dict(path='trajectories/'+name,episode=episode,env=env,object_code=code,
                scale=identity['scale'],split=identity['split'],official_success=bool(success[env]),
                endpoint_distance_m=end_distance,endpoint_within_5cm=end_distance<=.05,
                max_lift_m=float(data['object'][:,env,2].max()-data['object'][0,env,2])))
            count=counts.setdefault(code,dict(object_code=code,scale=identity['scale'],split=identity['split'],
                trajectories=0,official_success=0,endpoint_within_5cm=0))
            count['trajectories']+=1;count['official_success']+=int(success[env]);count['endpoint_within_5cm']+=int(end_distance<=.05)
            label='success' if success[env] else 'failure'
            representatives.setdefault((identity['object_id'],label),(data,env,surface))
        records.append(dict(episode=episode,envs=envs,raw_frames=frames,frames_30hz=len(indices),
                            anchors_per_env=len(anchors),official_success=int(success.sum()),
                            finite=True,clock_pass=True,no_internal_reset=True))
    rendered=0
    for (object_id,label),(data,env,surface) in representatives.items():
        if object_id>=render_objects:continue
        render(data,env,surface,root,label if len(specs)==1 else 'object-%03d-%s'%(object_id,label));rendered+=1
    (root/'trajectory-manifest.json').write_text(json.dumps(dict(profile='unidexgrasp-shadow-complete-cad-provisional',
        objects=specs,trajectories=trajectories,training_allowed=False),indent=2))
    (root/'window-manifest.json').write_text(json.dumps(dict(samples=samples,exported=export_windows,
        training_allowed=False),indent=2))
    summary=dict(status='COMPLETE',episodes=records,official_success=sum(r['official_success'] for r in records),
                 total_episodes=sum(r['envs'] for r in records),
                 hand_object_candidate_windows=sum(r['envs']*r['anchors_per_env'] for r in records),
                 exported_engineering_windows=len(samples) if export_windows else 0,training_admitted=0,training_allowed=False,
                 fixture_shapes={key:list(value.shape) for key,value in sample.items()},
                 objects=list(counts.values()),causality_checked_objects=sorted(causal_checked),
                 compact_trajectory_bytes=sum(p.stat().st_size for p in compact.glob('*.npz')),rendered_videos=rendered,
                 elapsed_s=time.monotonic()-begin,
                 visualization='CPU reconstruction of recorded hand landmarks/object poses; not camera RGB',
                 limitations=['bounded objects/single seed; no formal policy benchmark','pose-complete CAD profile',
                              'official success is goal proximity flag, not independently validated stable grasp'])
    (root/'audit-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--render-objects',type=int,default=1)
    parser.add_argument('--no-window-export',action='store_true',help='Audit fixed-time windows but store only native compact trajectories')
    args=parser.parse_args()
    if not 0<=args.render_objects<=5:raise ValueError('bounded visualization sample required')
    main(args.run.resolve(),args.render_objects,not args.no_window_export)
