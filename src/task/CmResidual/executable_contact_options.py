"""Explicit native-PD hold plans and source collision-mesh clearance."""
from pathlib import Path
import torch

INDEPENDENT=(0,1,2,3,4,5,6,8,10,12,14,15)
COUPLINGS=((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))


def hold_target(position,offset,scale):
    target=position.clone()
    target[:,6:]=torch.maximum(offset[6:],torch.minimum(target[:,6:],offset[6:]+scale[6:]))
    for dst,src,ratio in COUPLINGS:target[:,dst]=target[:,src]*ratio
    return target


def hold_action(target,position,offset,scale):
    if (scale[list(INDEPENDENT)]<=0).any():raise ValueError('nonpositive native independent scale')
    raw=(target-offset)/scale
    raw[:,:6]=(target[:,:6]-position[:,:6]-offset[:6])/scale[:6]
    raw[:,6:]=2*raw[:,6:]-1
    raw[:,[7,9,11,13,16,17]]=0
    if not torch.isfinite(raw).all():raise ValueError('nonfinite hold command')
    return raw.clamp(-1,1)


def obj_vertices(path,device):
    vertices=[list(map(float,s.split()[1:4])) for s in Path(path).read_text().splitlines() if s.startswith('v ')]
    result=torch.tensor(vertices,dtype=torch.float32,device=device)
    if result.ndim!=2 or result.shape[1]!=3 or not torch.isfinite(result).all():raise ValueError('invalid source mesh')
    return result


def rotate(q,v):
    xyz=q[:,:3];w=q[:,3:4]
    return v+2*(w*torch.cross(xyz,v,dim=-1)+torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1))


class TableClearance:
    def __init__(self,object_vertices,table_vertices,object_scale=1.):
        self.vertices=object_vertices*object_scale
        extent=table_vertices.max(0).values-table_vertices.min(0).values
        self.axis=int(extent.argmin());self.low=table_vertices.min(0).values[self.axis];self.high=table_vertices.max(0).values[self.axis]
        if not extent[self.axis]<extent.max()/10:raise ValueError('table is not a thin planar asset')

    def clearance(self,object_pose,table_pose):
        q=table_pose[:,3:7];oq=object_pose[:,3:7]
        if not torch.allclose(q.norm(dim=-1),torch.ones(len(q),device=q.device),atol=1e-4) or not torch.allclose(oq.norm(dim=-1),torch.ones(len(oq),device=oq.device),atol=1e-4):raise ValueError('nonunit geometry quaternion')
        local=torch.zeros_like(table_pose[:,:3]);local[:,self.axis]=1
        normal=rotate(q,local);sign=torch.where(normal[:,2]>=0,1.,-1.);normal*=sign[:,None]
        if (normal[:,2]<.95).any():raise ValueError('table plane not approximately horizontal')
        inverse=oq.clone();inverse[:,:3]*=-1
        direction=rotate(inverse,normal)
        support=(self.vertices@direction.T).min(0).values
        plane=torch.where(sign>0,self.high,-self.low)
        return support+((object_pose[:,:3]-table_pose[:,:3])*normal).sum(-1)-plane
