#!/usr/bin/env python3
"""Read four official static grasps with MuJoCo FK; never synthesize trajectories."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

CAR = 'sem/Car-669043a8ce40d9d78781f76a6db4ab62'
FINGERS = ['th', 'ff', 'mf', 'rf', 'lf']
BASES = ['thhub', 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle']


def fk_model(path):
    original = ET.parse(path).getroot()
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    root.append(copy.deepcopy(original.find('worldbody')))
    # Keep the exact joints/body transforms/inertias/sites. FK needs no visual
    # meshes, actuator defaults, sensors or legacy MuJoCo compiler settings.
    for parent in root.iter():
        for child in list(parent):
            if child.tag not in ('compiler', 'worldbody', 'body', 'joint', 'inertial', 'site'):
                parent.remove(child)
        for key in ('class', 'material', 'user'):
            parent.attrib.pop(key, None)
        if parent.tag == 'joint':
            parent.attrib.setdefault('type', 'hinge')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    if model.nq != 22 or model.nv != 22:
        raise ValueError('exact 22-joint official wrist-free ShadowHand required')
    return model


def world(values, rotation, translation):
    return np.asarray(values) @ rotation.T + translation


def audit(root):
    start = time.monotonic()
    manifest = json.loads((root/'download-manifest.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise ValueError('complete pinned official subset required')
    assets = root/'raw/dexgrasp_policy/assets'
    model = fk_model(assets/'mjcf/shadow_hand_wrist_free.xml')
    data = mujoco.MjData(model)
    fig = plt.figure(figsize=(15, 16))
    hand_list, surface_list, scales, rows = [], [], [], []
    for index in range(4):
        label = assets/'datasetv4.1'/CAR/('%05d.npz' % index)
        # Official NPZ encodes qpos as a dictionary object. The source is pinned
        # and recorded; this is not a reader for arbitrary untrusted pickles.
        with np.load(label, allow_pickle=True) as source:
            if set(source.files) != {'qpos', 'scale', 'plane'}:
                raise ValueError('unexpected static grasp schema')
            qpos = source['qpos'].item()
            inverse = float(source['scale'])
            plane = source['plane'].copy()
        if len(qpos) != 28 or not np.isfinite(list(qpos.values())).all() or inverse <= 0:
            raise ValueError('invalid grasp parameters')
        scale = round(1/inverse, 2)
        if scale not in (.06, .08, .10, .12, .15) or not np.isfinite(plane).all():
            raise ValueError('invalid scale or support plane')
        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                       for j in range(model.njnt)]
        if set(qpos) != set(joint_names + ['WRJRx', 'WRJRy', 'WRJRz', 'WRJTx', 'WRJTy', 'WRJTz']):
            raise ValueError('named joint/root parameters required')
        for j, name in enumerate(joint_names):
            data.qpos[model.jnt_qposadr[j]] = qpos[name]
        mujoco.mj_forward(model, data)
        rotation = Rotation.from_euler('xyz', [qpos['WRJR'+a] for a in 'xyz']).as_matrix()
        translation = np.array([qpos['WRJT'+a] for a in 'xyz'])
        body = lambda name: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0:'+name)]
        site = lambda name: data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'robot0:S_'+name+'tip')]
        points = world([body('palm')] + [body(name) for name in BASES] +
                       [site(name) for name in FINGERS], rotation, translation)
        # The URDF points to an already-scaled mesh; no extra scale is applied.
        suffix = '%03d' % round(scale*100)
        urdf = ET.parse(assets/'meshdatav3_scaled'/CAR/'coacd'/('coacd_'+suffix+'.urdf'))
        spec = urdf.getroot().find('link/visual/geometry/mesh')
        if not np.allclose(np.fromstring(spec.get('scale'), sep=' '), 1.):
            raise ValueError('unexpected additional URDF mesh scaling')
        mesh = trimesh.load_mesh(assets/'meshdatav3_scaled'/CAR/'coacd'/spec.get('filename'), process=False)
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
            raise ValueError('invalid rigid object mesh')
        # Deterministic area-weighted surface sample, for static geometry only.
        rng = np.random.RandomState(index)
        faces = rng.choice(len(mesh.faces), 512, p=mesh.area_faces/mesh.area)
        bary = rng.rand(512, 2)
        flip = bary.sum(1)>1
        bary[flip] = 1-bary[flip]
        triangles = mesh.triangles[faces]
        surface = triangles[:,0] + bary[:,:1]*(triangles[:,1]-triangles[:,0]) + bary[:,1:]*(triangles[:,2]-triangles[:,0])
        closest, distance, _ = trimesh.proximity.closest_point_naive(mesh, points[6:])
        hand = np.zeros((2,11,3), np.float32)
        hand[0] = points
        hand_list.append(hand)
        surface_list.append(surface.astype(np.float32))
        scales.append(scale)
        rows.append(dict(label=label.name, keys=['qpos','scale','plane'], inverse_scale=inverse,
                         physical_scale=scale, mesh_extents_m=mesh.extents.tolist(),
                         plane_normal_norm=float(np.linalg.norm(plane[:3])),
                         fingertip_unsigned_mesh_distance_m=distance.tolist(),
                         finite_hand=bool(np.isfinite(points).all()),
                         joint_limit_violations=int(((data.qpos<model.jnt_range[:,0]-1e-4) |
                                                    (data.qpos>model.jnt_range[:,1]+1e-4)).sum()),
                         has_time=False, has_trajectory=False))
        for view, angle in enumerate((35,135)):
            axis = fig.add_subplot(4,2,index*2+view+1,projection='3d')
            axis.scatter(*surface.T, s=2, color='gray', alpha=.35)
            for finger, base in zip(FINGERS, BASES):
                names = ([base, finger+'proximal', finger+'middle', finger+'distal'] if finger!='th'
                         else ['thbase','thproximal','thhub','thmiddle','thdistal'])
                chain = world([body('palm')] + [body(n) for n in names] + [site(finger)], rotation, translation)
                axis.plot(*chain.T, linewidth=2, marker='o', markersize=3)
            axis.scatter(*points.T, color='red', s=10)
            lo, hi = np.minimum(mesh.bounds[0],points.min(0)), np.maximum(mesh.bounds[1],points.max(0))
            center = (lo+hi)/2
            span = float((hi-lo).max())*.55
            axis.set_xlim(center[0]-span,center[0]+span)
            axis.set_ylim(center[1]-span,center[1]+span)
            axis.set_zlim(center[2]-span,center[2]+span)
            axis.set_box_aspect((1,1,1))
            axis.view_init(elev=25,azim=angle)
            axis.set_title('%s | physical scale %.2f | static grasp' % (label.name,scale))
            axis.set_xlabel('x (m)');axis.set_ylabel('y (m)');axis.set_zlabel('z (m)')
    valid = np.zeros((4,2,11), bool);valid[:,0] = True
    np.savez_compressed(root/'static-grasps.npz', hand=np.stack(hand_list), hand_valid=valid,
                        object_surface=np.stack(surface_list), physical_scale=scales, training_allowed=False)
    fig.tight_layout()
    fig.savefig(root/'review-static-grasps.jpg', dpi=110)
    plt.close(fig)
    summary = dict(status='COMPLETE', scope='static geometry and FK only', samples=rows,
                   hand_order=['right','left'], hand_landmarks=['palm origin']+BASES+['S_'+n+'tip' for n in FINGERS],
                   source_commit=manifest['upstream_commit'], elapsed_s=time.monotonic()-start,
                   code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   training_allowed=False, full_dynamic_samples=0,
                   remaining=['real policy rollouts and physical timestamps','robot landmark semantic review',
                              'source-specific scene observation profile','object-disjoint train/dev'])
    (root/'audit-summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.subset),indent=2))
