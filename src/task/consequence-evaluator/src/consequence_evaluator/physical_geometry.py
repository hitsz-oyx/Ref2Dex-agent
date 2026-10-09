"""Measured hand/object geometry; no force or quality flags in oracle futures."""
from pathlib import Path

import torch

from .contracts import HAND_LINKS


def poses(states):
    from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
    return dexplore_root_pose(states.reshape(-1,13)).reshape(*states.shape[:-1],4,4)


class PhysicalGeometry:
    """Sample actual URDF surfaces at measured native rigid-body poses.

    Each object uses its own URDF, preserving collision-frame mesh origins.
    Native body poses are measured, including finger and floating-wrist motion.
    The surface gap is unsigned sampled proximity, not an exact collision pair.
    """
    def __init__(self, task, assets, distance_device=None):
        from src.task.CmResidual.v118_planner import QUERY_LINKS
        from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
        names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
        self.query_ids=[names.index(name) for name in QUERY_LINKS]
        self.key_ids=[names.index(name) for name in HAND_LINKS]
        self.distance_device = distance_device or task.device
        self.surfaces={}
        for name in task.object_name:
            self.surfaces[name]=_surface_geometry_class()(
                hand_urdf=Path(assets)/'inspire_hand_new/inspire_hand_right.urdf',
                object_urdf=Path(assets)/('mjcf/'+name+'.urdf'),query_links=QUERY_LINKS,
                object_count=1024,hand_count=1538,seed=42,device=task.device)

    @torch.inference_mode()
    def measure(self, task):
        bodies=task._rigid_body_state.view(task.num_envs,-1,13)
        link_poses=poses(bodies[:,self.query_ids])
        object_pose=poses(task._target_states)
        gaps=torch.empty(task.num_envs,device=task.device)
        object_ids=task.object_id[task.data_id]
        for index,name in enumerate(task.object_name):
            ids=torch.nonzero(object_ids==index).flatten()
            if not len(ids):continue
            surface=self.surfaces[name]
            hand,_=surface.hand(link_poses[ids])
            obj,_=surface.object(object_pose[ids])
            center=object_pose[ids,:3,3][:,None]
            obj=center+(obj-center)*task.ball_size
            # Dense diagnostic samples, independently of the force proxy.
            # A CPU simulator tensor interface need not put dense proximity
            # arithmetic on CPU. Bound temporary pairwise matrices to16envs.
            for begin in range(0,len(ids),16):
                chunk=slice(begin,begin+16)
                value=torch.cdist(hand[chunk].to(self.distance_device),obj[chunk].to(self.distance_device)).amin(dim=(1,2))
                gaps[ids[chunk]]=value.to(task.device)
        return bodies[:,self.key_ids,:3].clone(),gaps
