"""Read-only provenance for the first hand-derived decoder coverage Probe."""
import json
from pathlib import Path
import pickle
from consequence_evaluator.data import sha


def load_geometry(folder,reference,urdf):
    import numpy as np
    folder,reference,urdf=map(Path,(folder,reference,urdf))
    m=json.loads((folder/'manifest.json').read_text())
    if (m.get('status')!='COMPLETED' or m.get('schema')!='ref2dex.tau-geometry-reference.v1'
            or sha(reference)!=m['reference_sha256'] or sha(folder/'geometry.npz')!=m['geometry_sha256']
            or sha(urdf)!=m['urdf_sha256']):
        raise ValueError('hand-derived geometry provenance mismatch')
    with np.load(folder/'geometry.npz',allow_pickle=False) as s:geometry={k:s[k] for k in s.files}
    with reference.open('rb') as s:packet=pickle.load(s)
    # Future measured q/object/commands are deliberately not extracted.
    initial={k:packet[k][0,0].copy() for k in ('dof_position','dof_velocity','hand_keypoints','object_pose')}
    hand=packet['hand_keypoints'][:,0].copy()
    if not np.array_equal(hand,geometry['target_points']) or not np.array_equal(initial['dof_position'],geometry['reset_q']):
        raise ValueError('hand label/reset alignment mismatch')
    files=[folder/'manifest.json',folder/'geometry.npz',reference,urdf]
    return geometry,initial,hand,{str(p.resolve()):sha(p) for p in files}
