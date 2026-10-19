"""Offline FK and collision-surface sampling of the recorded Shadow MJCF asset."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def vector(node,key,default):
    return np.fromstring(node.get(key,default),sep=' ')


def local_pose(node):
    t=np.eye(4);t[:3,3]=vector(node,'pos','0 0 0')
    if 'quat' in node.attrib:t[:3,:3]=Rotation.from_quat(vector(node,'quat','')[ [1,2,3,0] ]).as_matrix()
    elif 'euler' in node.attrib:t[:3,:3]=Rotation.from_euler('xyz',vector(node,'euler','')).as_matrix()
    return t


class ShadowSurface:
    def __init__(self,asset):
        self.root=ET.parse(asset).getroot().find('body')
        self.bodies=list(self.root.iter('body'))
        self.joints=[j for b in self.bodies for j in b.findall('joint')]
        if len(self.joints)!=22:raise ValueError('exact recorded22-DOF replay asset required')
        self.names=[j.get('name') for j in self.joints]
        # Sample the actual collision primitives, excluding visualization meshes.
        import trimesh
        meshes=[];owners=[];labels=[]
        for body in self.bodies:
            name=body.get('name');part=0
            for f,prefix in enumerate(('th','ff','mf','rf','lf')):
                if name.startswith('robot0:'+prefix):
                    part=1+3*f+(2 if 'distal' in name else 1 if 'middle' in name else 0)
            for geom in body.findall('geom'):
                if geom.get('class')!='robot0:DC_Hand':continue
                size=vector(geom,'size','');kind=geom.get('type')
                if kind=='box':mesh=trimesh.creation.box(extents=size*2)
                elif kind=='capsule':
                    mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[12,12])
                    mesh.vertices[:,2]-=size[1]
                elif kind=='sphere':mesh=trimesh.creation.icosphere(subdivisions=2,radius=size[0])
                else:raise ValueError('unsupported recorded collision geometry: '+str(kind))
                transform=local_pose(geom);mesh.apply_transform(transform)
                meshes.append(mesh);owners.append(name);labels.append(part)
        areas=np.array([m.area for m in meshes]);rng=np.random.default_rng(20261019)
        # Reserve coverage for every primitive, then distribute remaining points
        # by area; no random per-frame resampling or fabricated MANO topology.
        counts=np.ones(len(meshes),int)*4
        counts+=rng.multinomial(778-int(counts.sum()),areas/areas.sum())
        samples=[];sample_owner=[];sample_part=[]
        for mesh,owner,part,count in zip(meshes,owners,labels,counts):
            tri=mesh.triangles;area=mesh.area_faces
            chosen=tri[rng.choice(len(tri),count,p=area/area.sum())]
            uv=rng.random((count,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1]
            samples.append(chosen[:,0]+uv[:,:1]*(chosen[:,1]-chosen[:,0])+uv[:,1:]*(chosen[:,2]-chosen[:,0]))
            sample_owner.extend([owner]*count);sample_part.extend([part]*count)
        self.surface=np.concatenate(samples);self.owners=np.asarray(sample_owner);self.parts=np.asarray(sample_part,np.uint8)

    def reconstruct(self,q,root):
        n=len(q)
        if q.shape!=(n,22) or root.shape!=(n,13):raise ValueError('recorded q22/root13 required')
        transforms={};counter=[0]
        def visit(body,parent):
            transform=parent@local_pose(body)
            for joint in body.findall('joint'):
                axis=vector(joint,'axis','0 0 1');axis/=np.linalg.norm(axis)
                r=Rotation.from_rotvec(q[:,counter[0],None]*axis).as_matrix();counter[0]+=1
                a=vector(joint,'pos','0 0 0')
                jt=np.broadcast_to(np.eye(4),(n,4,4)).copy();jt[:,:3,:3]=r;jt[:,:3,3]=a-np.einsum('nij,j->ni',r,a)
                transform=transform@jt
            transforms[body.get('name')]=transform
            for child in body.findall('body'):visit(child,transform)
        rt=np.broadcast_to(np.eye(4),(n,4,4)).copy();rt[:,:3,:3]=Rotation.from_quat(root[:,3:7]).as_matrix();rt[:,:3,3]=root[:,:3]
        visit(self.root,rt)
        joints=[transforms['robot0:palm'][:,:3,3]]
        for f in ('th','ff','mf','rf','lf'):
            names=['thhub','thmiddle','thdistal'] if f=='th' else [f+'knuckle',f+'middle',f+'distal']
            joints.extend(transforms['robot0:'+name][:,:3,3] for name in names)
            tip=transforms['robot0:'+f+'distal'];offset=np.array([0,0,.0275 if f=='th' else .026])
            joints.append(np.einsum('nij,j->ni',tip[:,:3,:3],offset)+tip[:,:3,3])
        jp=np.zeros((n,2,21,3),np.float32);jp[:,0]=np.stack(joints,axis=1)
        vertices=np.zeros((n,2,778,3),np.float32)
        for name in np.unique(self.owners):
            ids=np.flatnonzero(self.owners==name);tf=transforms[name]
            vertices[:,0,ids]=np.einsum('nij,vj->nvi',tf[:,:3,:3],self.surface[ids])+tf[:,None,:3,3]
        valid=np.zeros((n,2),bool);valid[:,0]=True
        parts=np.zeros((2,778),np.uint8);parts[0]=self.parts
        return jp,np.broadcast_to(valid[...,None],(n,2,21)).copy(),vertices,valid,parts
