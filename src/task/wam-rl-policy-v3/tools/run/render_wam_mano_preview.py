"""Render a fixed WAM forecast as collision-free MANO surfaces in Isaac Gym.

Run in the pinned native Gym environment, before importing torch. Camera frames
are saved as NumPy arrays; encode them separately in the model environment.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET


def write_rgb_png(path, pixels):
    """Small PNG writer for the pinned Gym runtime, which has no Pillow."""
    import struct
    import zlib
    height,width,channels=pixels.shape
    if channels!=3:raise ValueError('RGB camera frame required')
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    raw=b''.join(b'\0'+row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
                     +chunk(b'IDAT',zlib.compress(raw,3))+chunk(b'IEND',b''))


def tracking_camera(teacher_hand,object_pose,predicted_hand=None,predicted_object_pose=None):
    """Shared camera encloses both hands and object, including early approach."""
    import numpy as np
    center=np.asarray(object_pose)[:3,3]
    parts=[np.asarray(teacher_hand),center[None]+np.array([[-.06,-.06,-.06],[.06,.06,.06]])]
    if predicted_hand is not None:parts.append(np.asarray(predicted_hand))
    if predicted_object_pose is not None:
        parts.append(np.asarray(predicted_object_pose)[:3,3][None]+np.array([[-.06,-.06,-.06],[.06,.06,.06]]))
    points=np.concatenate(parts)
    target=(points.min(0)+points.max(0))/2
    radius=float(np.linalg.norm(points-target,axis=-1).max())
    direction=np.array([-.32,-.30,.27]);direction/=np.linalg.norm(direction)
    return target+direction*max(.48,3.2*radius),target


def visual_urdf(source, destination):
    """Copy only the description; keep licensed/native mesh files read-only."""
    tree = ET.parse(str(source))
    for link in tree.getroot().findall('link'):
        for collision in link.findall('collision'):
            link.remove(collision)
    for mesh in tree.getroot().iter('mesh'):
        filename = mesh.get('filename')
        mesh.set('filename', str((source.parent / filename).resolve()))
    tree.write(str(destination), encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--height', type=int, default=640)
    args = parser.parse_args()
    output = args.preview.resolve()
    if not (output / 'manifest.json').is_file():
        raise ValueError('prepared preview required')
    if not 256 <= args.width <= 1024 or not 256 <= args.height <= 1024:
        raise ValueError('bounded camera size required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2')
    sys.path.insert(0, '/home2/wyy/isaac-gym/isaacgym/python')
    from isaacgym import gymapi
    import numpy as np
    from scipy.spatial.transform import Rotation

    started = time.monotonic()
    data = np.load(output / 'preview.npz')
    metadata = json.loads((output / 'manifest.json').read_text())
    autoregressive=metadata.get('preview_mode')=='rolling_predicted_history'
    rolling=metadata.get('preview_mode') in ('rolling_measured_history','rolling_predicted_history')
    frame_count=len(data['teacher_q'])
    valid=data['prediction_valid'] if rolling else np.ones(frame_count,bool)
    mesh_ids=np.flatnonzero(valid)
    if rolling:
        image_dir=output/'camera-frames'
        image_dir.mkdir()  # Refuse to overwrite previous/partial camera output.
    source = Path(metadata['source'])
    receipt = json.loads((source.parents[1] / 'manifest.json').read_text())
    contract = receipt['observation_contract']
    admission = json.loads(Path(metadata['source_manifest']).read_text())
    object_name = next(row['object'] for row in admission['records'] if row['source'] == str(source))
    # The measured archive is native base/thumb/base-tip order; preview is WAM order.
    native_to_wam = [0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    body_names = [contract['body_names'][i] for i in contract['measured_hand_body_ids']]
    body_names = [body_names[i] for i in native_to_wam]
    asset_root = Path('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets')
    assets = output / 'scene-assets'
    assets.mkdir(exist_ok=True)
    sources = dict(hand=asset_root / 'inspire_hand_new/inspire_hand_right.urdf',
                   object=asset_root / ('mjcf/' + object_name + '.urdf'), table=asset_root / 'mjcf/table.urdf')
    for name, path in sources.items():
        visual_urdf(path, assets / (name + '.urdf'))

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.dt = 1 / 120
    params.physx.num_threads = 2
    # CUDA device is remapped; Vulkan graphics ordinal is the physical device.
    sim = gym.create_sim(0, args.gpu, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Gym graphics simulator unavailable')
    gym.set_light_parameters(sim, 0, gymapi.Vec3(.9, .9, .9),
                             gymapi.Vec3(.5, .5, .5), gymapi.Vec3(-.3, -.4, -1))
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    loaded = {name: gym.load_asset(sim, str(assets), name + '.urdf', options)
              for name in sources}
    mano_assets = [gym.load_asset(sim, str(output / 'mano-assets'),
                   ('%04d.urdf' if rolling else '%02d.urdf') % i, options) for i in mesh_ids]
    marker_asset = gym.create_sphere(sim, .0025, options)
    goal_asset = gym.create_sphere(sim, .006, options)

    def transform(pose):
        value = gymapi.Transform()
        pose = np.asarray(pose)
        xyz = pose[:3, 3] if pose.shape == (4, 4) else pose[:3]
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat() if pose.shape == (4, 4) else pose[3:7]
        value.p = gymapi.Vec3(*xyz)
        value.r = gymapi.Quat(*quat)
        return value

    def color(env, actor, rgb):
        for index in range(gym.get_actor_rigid_body_count(env, actor)):
            gym.set_rigid_body_color(env, actor, index, gymapi.MESH_VISUAL, gymapi.Vec3(*rgb))

    def actor_pose(env, actor, pose):
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        if len(states) != 1:
            raise ValueError('visual replay root requires a single rigid body')
        states['pose']['p'][0] = (pose.p.x, pose.p.y, pose.p.z)
        states['pose']['r'][0] = (pose.r.x, pose.r.y, pose.r.z, pose.r.w)
        states['vel']['linear'][0] = (0, 0, 0)
        states['vel']['angular'][0] = (0, 0, 0)
        gym.set_actor_rigid_body_states(env, actor, states, gymapi.STATE_ALL)

    envs, objects, cameras, manoes, goals = [], [], [], [], []
    hand = None
    camera = gymapi.CameraProperties()
    camera.width, camera.height, camera.horizontal_fov = args.width, args.height, 48
    center = data['object_poses'][:, :3, 3].mean(0)
    center[2] += .015
    eye = center + np.array([-.32, -.30, .27])
    if rolling:
        radius=float(np.linalg.norm(data['object_poses'][:,:3,3]-center,axis=-1).max())
        eye=center+np.array([-.40,-.42,.34])*max(1.,radius/.15)
    for lane in range(2):
        env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, -1.5), gymapi.Vec3(1.5, 1.5, 1.5), 2)
        envs.append(env)
        table = gym.create_actor(env, loaded['table'], transform(data['table_pose']), 'table', lane, 0)
        color(env, table, (.38, .42, .47))
        obj = gym.create_actor(env, loaded['object'], transform(data['object_poses'][0]), 'object', lane, 0)
        color(env, obj, (.86, .25, .12))
        objects.append(obj)
        if lane == 0:
            hand = gym.create_actor(env, loaded['hand'], gymapi.Transform(), 'measured_inspire', lane, 0)
            color(env, hand, (.76, .77, .8))
        else:
            hidden = gymapi.Transform()
            hidden.p = gymapi.Vec3(0, 0, -10)
            manoes = [gym.create_actor(env, asset, hidden, 'mano_%02d' % i, lane, 0)
                      for i, asset in enumerate(mano_assets)]
            for actor in manoes:
                color(env, actor, (.08, .67, .82))
        for index, pose in enumerate(data['object_poses'][::8] if rolling else data['object_poses']):
            marker = gym.create_actor(env, marker_asset, transform(pose), 'path_%02d' % index, lane, 0)
            color(env, marker, (.98, .68, .16))
        marker = gym.create_actor(env, goal_asset, transform(data['goal_pose']), 'local_goal', lane, 0)
        color(env, marker, (.98, .85, .18))
        goals.append(marker)
        sensor = gym.create_camera_sensor(env, camera)
        if sensor < 0:
            raise RuntimeError('camera creation failed')
        gym.set_camera_location(sensor, env, gymapi.Vec3(*eye), gymapi.Vec3(*center))
        cameras.append(sensor)
    gym.prepare_sim(sim)
    hand_ids = [gym.find_actor_rigid_body_index(envs[0], hand, name, gymapi.DOMAIN_ACTOR)
                for name in body_names]
    if min(hand_ids) < 0:
        raise ValueError('teacher semantic landmark link absent')
    frames, fk_errors, camera_poses = [], [], []
    mano_by_frame=dict(zip(mesh_ids,manoes));previous_mano=None;previous_camera=None
    for tick in range(frame_count):
        if time.monotonic()-started>300:raise TimeoutError('five-minute render deadline')
        joints = gym.get_actor_dof_states(envs[0], hand, gymapi.STATE_ALL)
        joints['pos'] = data['teacher_q'][tick]
        joints['vel'] = 0
        gym.set_actor_dof_states(envs[0], hand, joints, gymapi.STATE_ALL)
        for lane in range(2):
            object_pose=data['display_object_poses'][tick] if autoregressive and lane==1 else data['object_poses'][tick]
            actor_pose(envs[lane], objects[lane], transform(object_pose))
            if rolling:actor_pose(envs[lane],goals[lane],transform(data['goal_poses'][max(0,int(data['anchor_tick'][tick]))] if valid[tick] else data['goal_poses'][tick]))
        if previous_mano is not None:
            hidden=gymapi.Transform();hidden.p=gymapi.Vec3(0,0,-10)
            actor_pose(envs[1],previous_mano,hidden)
        previous_mano=mano_by_frame.get(tick)
        if previous_mano is not None:actor_pose(envs[1],previous_mano,gymapi.Transform())
        if rolling:
            eye,center=tracking_camera(data['teacher_hand'][tick],data['object_poses'][tick],
                data['target_hand'][tick] if valid[tick] else None,
                data['display_object_poses'][tick] if autoregressive else None)
            if autoregressive:
                if previous_camera is not None:
                    eye=.85*previous_camera[0]+.15*eye;center=.85*previous_camera[1]+.15*center
                previous_camera=(eye.copy(),center.copy())
            camera_poses.append(np.r_[eye,center])
            for env,sensor in zip(envs,cameras):
                gym.set_camera_location(sensor,env,gymapi.Vec3(*eye),gymapi.Vec3(*center))
        # Setters update articulation FK. Do not integrate unactuated visual-only
        # zero-inertia links: that would turn a kinematic replay into a new rollout.
        states = gym.get_actor_rigid_body_states(envs[0], hand, gymapi.STATE_POS)
        actual = np.array([[states['pose']['p'][i][axis] for axis in ('x', 'y', 'z')]
                           for i in hand_ids])
        # The per-actor API returns environment-local positions.
        fk_errors.append(np.linalg.norm(actual - data['teacher_hand'][tick], axis=-1))
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        images = [np.asarray(gym.get_camera_image(sim, env, sensor, gymapi.IMAGE_COLOR)).reshape(args.height, args.width, 4)[..., :3].copy()
                  for env, sensor in zip(envs, cameras)]
        frame=np.concatenate(images,axis=1)
        if rolling:
            write_rgb_png(image_dir/('%04d.png'%tick),frame)
            if tick%100==0:print(json.dumps(dict(stage='render',frame=tick,total=frame_count,elapsed_s=time.monotonic()-started)),flush=True)
        else:frames.append(frame)
    fk_errors = np.asarray(fk_errors)
    if rolling:
        np.save(output/'fk-errors.npy',fk_errors)
        np.save(output/'camera-poses.npy',np.asarray(camera_poses))
    else:np.savez_compressed(output / 'camera-frames.npz', frames=np.asarray(frames), fk_errors_m=fk_errors)
    report = dict(status='COMPLETED' if fk_errors.max() <= .002 else 'FAILED',
                  python=sys.executable, gpu=args.gpu, frames=frame_count, camera_hz=30,
                  fk_mean_mm=float(fk_errors.mean() * 1000), fk_max_mm=float(fk_errors.max() * 1000),
                  camera_eye=eye.tolist(), camera_target=center.tolist(), elapsed_s=time.monotonic() - started,
                  left='measured teacher q in native Inspire URDF', right='MANO fitted to rolling WAM24/display8' if rolling else 'MANO fitted to fixed WAM prediction',
                  object='measured future object poses for comparison only', path='gold teacher object path; larger gold local goal',
                  physics='kinematic setters only; no physics integration; all roots fixed and collision geometry removed',
                  exit_workaround='flush completed outputs then os._exit(0): legacy Gym runtime crashes during teardown')
    if rolling:report['camera_policy']='shared follow camera enclosing teacher hand, raw forecast hand and object; no prediction coordinates changed'
    if autoregressive:
        report.update(object='left recorded teacher; right rigid-projected WAM object prediction',
            camera_policy='same camera in both lanes; exponential camera-only smoothing .15; includes predicted object; hand/object outputs unchanged',
            right='MANO fitted to predicted-history autoregressive WAM24/display8')
    (output / 'render-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if report['fk_max_mm'] > 2:
        raise ValueError('native joint replay does not reproduce teacher link landmarks within 2 mm')
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
