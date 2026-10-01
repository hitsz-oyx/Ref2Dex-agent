"""Conservative separation from the actual thin tabletop's upper support plane."""
import torch
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose


def tabletop_geometry(table_vertices,table_root):
    extent=table_vertices.amax(0)-table_vertices.amin(0)
    axis=int(extent.argmin())
    if extent[axis]<=0 or float(extent.max()/extent[axis])<10:
        raise ValueError('asset is not a thin tabletop; no assumed surface normal')
    rotation=dexplore_root_pose(table_root)[:,:3,:3]
    vertical=rotation[:,2,axis]
    if (vertical.abs()<.999).any():
        raise ValueError('thin tabletop normal is not aligned with world up')
    normal=torch.zeros((len(table_root),3),device=table_root.device,dtype=table_root.dtype)
    normal[:,axis]=vertical.sign()
    top=(normal@table_vertices.T).amax(-1)
    return normal,top,axis


def clearance(object_root,table_root,object_vertices,table_vertices):
    normal,top,axis=tabletop_geometry(table_vertices,table_root)
    obj=dexplore_root_pose(object_root);table=dexplore_root_pose(table_root)
    inv=table[:,:3,:3].transpose(1,2)
    rotation=inv@obj[:,:3,:3]
    translation=(inv@(obj[:,:3,3]-table[:,:3,3]).unsqueeze(-1)).squeeze(-1)
    direction=torch.einsum('bi,bij->bj',normal,rotation)
    height=direction@object_vertices.T+(normal*translation).sum(-1)[:,None]
    return height.amin(-1)-top
