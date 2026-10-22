"""Inspire measured geometry adapter for the unchanged S2c stage predicate."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PALM_FEATURE_CONTRACT='inspire.actual-base-link-collision-vertices.object32-gap.v1'


class PalmSurface:
    def __init__(self, vertices):
        self.vertices=np.asarray(vertices,float)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not np.isfinite(self.vertices).all() or not len(self.vertices):
            raise ValueError('finite actual palm mesh vertices required')
        self.tree=cKDTree(self.vertices)

    @classmethod
    def from_urdf(cls,path):
        import trimesh
        path=Path(path);link=ET.parse(path).getroot().find("link[@name='hand_base_link']")
        collision=link.find('collision');mesh=collision.find('geometry/mesh');origin=collision.find('origin')
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        vertices=np.asarray(trimesh.load_mesh(path.parent/mesh.get('filename'),process=True).vertices)*scale
        return cls(vertices@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz)

    def gap(self,palm_pose,object_pose,object_points):
        palm_pose=np.asarray(palm_pose);object_pose=np.asarray(object_pose);points=np.asarray(object_points)
        if palm_pose.shape!=(4,4) or object_pose.shape!=(4,4) or points.ndim!=2 or points.shape[1]!=3:raise ValueError('live SE3 poses and canonical object points required')
        world=points@object_pose[:3,:3].T+object_pose[:3,3]
        local=(world-palm_pose[:3,3])@palm_pose[:3,:3]
        return float(self.tree.query(local,k=1)[0].min())

    def features(self,features,palm_pose,object_pose,object_points):
        # hand[0] remains the native wrist/base for WAM/R/feedback; only the
        # stage's palm-gap measurement uses the physical palm collision mesh.
        return replace(features,palm_gap_m=self.gap(palm_pose,object_pose,object_points))


def terminal_success(stage,goal_error_m,tick,horizon,physical_abort):
    """Successful progress cannot stand in for a full-horizon terminal state."""
    return bool(tick==horizon and not physical_abort and stage['current']['hold'] and goal_error_m<=.015)
