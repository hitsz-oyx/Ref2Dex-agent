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
    mano_assets = [gym.load_asset(sim, str(output / 'mano-assets'), '%02d.urdf' % i, options)
                   for i in range(25)]
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

    envs, objects, cameras, manoes = [], [], [], []
    hand = None
    camera = gymapi.CameraProperties()
    camera.width, camera.height, camera.horizontal_fov = args.width, args.height, 48
    center = data['object_poses'][:, :3, 3].mean(0)
    center[2] += .015
    eye = center + np.array([-.32, -.30, .27])
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
        for index, pose in enumerate(data['object_poses']):
            marker = gym.create_actor(env, marker_asset, transform(pose), 'path_%02d' % index, lane, 0)
            color(env, marker, (.98, .68, .16))
        marker = gym.create_actor(env, goal_asset, transform(data['goal_pose']), 'local_goal', lane, 0)
        color(env, marker, (.98, .85, .18))
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
    frames, fk_errors = [], []
    for tick in range(25):
        joints = gym.get_actor_dof_states(envs[0], hand, gymapi.STATE_ALL)
        joints['pos'] = data['teacher_q'][tick]
        joints['vel'] = 0
        gym.set_actor_dof_states(envs[0], hand, joints, gymapi.STATE_ALL)
        for lane in range(2):
            actor_pose(envs[lane], objects[lane], transform(data['object_poses'][tick]))
        for index, actor in enumerate(manoes):
            pose = gymapi.Transform()
            if index != tick:
                pose.p = gymapi.Vec3(0, 0, -10)
            actor_pose(envs[1], actor, pose)
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
        frames.append(np.concatenate(images, axis=1))
    fk_errors = np.asarray(fk_errors)
    np.savez_compressed(output / 'camera-frames.npz', frames=np.asarray(frames), fk_errors_m=fk_errors)
    report = dict(status='COMPLETED' if fk_errors.max() <= .002 else 'FAILED',
                  python=sys.executable, gpu=args.gpu, frames=25, camera_hz=30,
                  fk_mean_mm=float(fk_errors.mean() * 1000), fk_max_mm=float(fk_errors.max() * 1000),
                  camera_eye=eye.tolist(), camera_target=center.tolist(), elapsed_s=time.monotonic() - started,
                  left='measured teacher q in native Inspire URDF', right='MANO fitted to fixed WAM prediction',
                  object='measured future object poses for comparison only', path='gold teacher object path; larger gold local goal',
                  physics='kinematic setters only; no physics integration; all roots fixed and collision geometry removed',
                  exit_workaround='flush completed outputs then os._exit(0): legacy Gym runtime crashes during teardown')
    (output / 'render-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if report['fk_max_mm'] > 2:
        raise ValueError('native joint replay does not reproduce teacher link landmarks within 2 mm')
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
