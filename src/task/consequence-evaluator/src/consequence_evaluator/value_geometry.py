"""Measured tabletop support proxies, kept outside evaluator model inputs."""
from pathlib import Path
import numpy as np
import torch
from .physical_geometry import poses


class TableSupport:
    def __init__(self, assets, device):
        import trimesh
        mesh = trimesh.load(Path(assets)/'mjcf/objects/table/table.obj',force='mesh',process=False)
        low,high = mesh.bounds
        corners = np.array([[x,y,z] for x in (low[0],high[0])
                            for y in (low[1],high[1]) for z in (low[2],high[2])])
        self.corners = torch.as_tensor(corners,dtype=torch.float32,device=device)

    @torch.no_grad()
    def measure(self, task, geometry):
        table = poses(task._table_states)
        # The world up direction must coincide with a mesh bounding-box axis;
        # otherwise this simple tabletop-plane proxy would be invalid.
        if (table[:,2,:3].abs().max(-1).values < .999).any():
            raise ValueError('tabletop support proxy requires an upright axis-aligned vertical mesh axis')
        world = torch.einsum('bij,pj->bpi',table[:,:3,:3],self.corners)+table[:,:3,3,None].transpose(1,2)
        low,high = world.amin(1),world.amax(1)
        object_pose = poses(task._target_states)
        gap = torch.empty(task.num_envs,device=task.device)
        footprint = torch.empty(task.num_envs,dtype=torch.bool,device=task.device)
        object_ids = task.object_id[task.data_id]
        for index,name in enumerate(task.object_name):
            ids = torch.nonzero(object_ids==index).flatten()
            if not len(ids):
                continue
            points,_ = geometry.surfaces[name].object(object_pose[ids])
            center = object_pose[ids,:3,3][:,None]
            points = center+(points-center)*task.ball_size
            gap[ids] = points[:,:,2].amin(1)-high[ids,2]
            xy = object_pose[ids,:2,3]
            footprint[ids] = ((xy>=low[ids,:2])&(xy<=high[ids,:2])).all(-1)
        return gap,footprint
