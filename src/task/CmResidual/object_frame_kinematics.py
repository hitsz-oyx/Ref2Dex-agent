"""Differentiable URDF link origins/rotations with explicit native DOF ordering."""
import math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch

def rpy_matrix(rpy):
    x,y,z=rpy;cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

def quaternion_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True);x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)

class UrdfKinematics:
    def __init__(self,asset,dof_names):
        xml=ET.parse(Path(asset)).getroot();self.dof_names=list(dof_names);self.joints=[]
        links={n.attrib['name'] for n in xml.findall('link')};children={j.find('child').attrib['link'] for j in xml.findall('joint')};roots=links-children
        if len(roots)!=1:raise ValueError('single URDF tree root')
        self.root=next(iter(roots));ordered=[];remaining=list(xml.findall('joint'));known={self.root}
        while remaining:
            ready=[j for j in remaining if j.find('parent').attrib['link'] in known]
            if not ready:raise ValueError('URDF disconnected/cyclic joint tree')
            for j in ready:ordered.append(j);remaining.remove(j);known.add(j.find('child').attrib['link'])
        movable={j.attrib['name'] for j in ordered if j.attrib['type']!='fixed'}
        if movable!=set(self.dof_names):raise ValueError('exact native/URDF movable joint coverage')
        for j in ordered:
            origin=j.find('origin');xyz=[float(v) for v in origin.attrib.get('xyz','0 0 0').split()] if origin is not None else [0.,0.,0.];rpy=[float(v) for v in origin.attrib.get('rpy','0 0 0').split()] if origin is not None else [0.,0.,0.]
            axis=j.find('axis');a=np.array([float(v) for v in axis.attrib.get('xyz','1 0 0').split()]) if axis is not None else np.array([1.,0.,0.]);a=a/np.linalg.norm(a);ax,ay,az=a;skew=np.array([[0,-az,ay],[az,0,-ax],[-ay,ax,0]])
            self.joints.append(dict(name=j.attrib['name'],kind=j.attrib['type'],parent=j.find('parent').attrib['link'],child=j.find('child').attrib['link'],translation=np.array(xyz),rotation=rpy_matrix(rpy),axis=a,skew=skew,index=self.dof_names.index(j.attrib['name']) if j.attrib['name'] in self.dof_names else None))

    def forward(self,q,root):
        if q.ndim!=2 or q.shape[-1]!=len(self.dof_names) or root.shape!=(len(q),13):raise ValueError('actual native q/root packets')
        n=len(q);eye=torch.eye(3,dtype=q.dtype,device=q.device);positions={self.root:root[:,:3]};rotations={self.root:quaternion_matrix(root[:,3:7])}
        def const(v):return torch.as_tensor(v,dtype=q.dtype,device=q.device)
        for j in self.joints:
            rp=rotations[j['parent']];p=positions[j['parent']]+(rp@const(j['translation'])[None,:,None]).squeeze(-1);r=rp@const(j['rotation'])
            if j['kind']=='prismatic':p=p+(r@const(j['axis'])[None,:,None]).squeeze(-1)*q[:,j['index'],None]
            elif j['kind'] in ('revolute','continuous'):
                theta=q[:,j['index']];s=const(j['skew']);motion=eye+torch.sin(theta)[:,None,None]*s+(1-torch.cos(theta))[:,None,None]*(s@s);r=r@motion
            elif j['kind']!='fixed':raise ValueError('unsupported URDF joint type')
            positions[j['child']]=p;rotations[j['child']]=r
        return positions,rotations

def object_frame_points(points,object_root):
    rotation=quaternion_matrix(object_root[:,3:7]);return (rotation.transpose(-1,-2)[:,None]@(points-object_root[:,None,:3])[...,None]).squeeze(-1)
