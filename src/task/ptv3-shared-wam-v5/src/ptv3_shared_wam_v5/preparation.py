"""Offline geometric reconstruction and contact labels; never imported by the model."""
from pathlib import Path
import numpy as np
from .contracts import MANO_JOINTS_IN_21, MANO_JOINT_TO_PART, SPARSE11_IN_21


def load_base(job):
    if 'sequence_folder' in job:
        p=Path(job['sequence_folder'])
        hand=np.load(p/'hand.npy',mmap_mode='r')
        valid=np.load(p/'hand_valid.npy',mmap_mode='r')
        if valid.ndim==2:valid=np.broadcast_to(valid[...,None],hand.shape[:-1])
        return dict(hand=hand,hand_valid=valid,object_poses=np.load(p/'poses.npy',mmap_mode='r'),
            object_pose_valid=np.load(p/'pose_valid.npy',mmap_mode='r'),observed_frame=np.ones(len(hand),bool))
    with np.load(job['payload'],allow_pickle=False) as z:
        b={k:z[k] for k in z.files}
    if job['source_id']==4:
        from scipy.spatial.transform import Rotation
        root=b['object'];poses=np.broadcast_to(np.eye(4),(len(root),1,4,4)).copy()
        poses[:,0,:3,:3]=Rotation.from_quat(root[:,3:7]).as_matrix()
        poses[:,0,:3,3]=root[:,:3]
        hand=np.zeros((len(root),2,11,3),np.float32);hand[:,0]=b['hand']
        valid=np.zeros((len(root),2,11),bool);valid[:,0]=True
        return dict(hand=hand,hand_valid=valid,object_poses=poses,object_pose_valid=np.ones((len(root),1),bool),
                    observed_frame=np.ones(len(root),bool))
    b.setdefault('observed_frame',np.ones(len(b['hand']),bool))
    return b


def reconstruct(packet, mano_root, device='cpu', chunk_size=128):
    """Exact source conventions are verified against cached11 before publication."""
    for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),
                       ('object',object),('str',str),('unicode',str)]:
        if name not in np.__dict__:setattr(np,name,value)
    import torch
    from smplx import MANO
    n=len(packet['pose'])
    joints=np.zeros((n,2,21,3),np.float32)
    verts=np.zeros((n,2,778,3),np.float32)
    valid=packet['parameter_valid'].copy()
    parts=np.zeros((2,778),np.uint8)
    pca=int(packet['pca'])
    for side in (0,1):
        kw=dict(is_rhand=side==0,use_pca=bool(pca),num_pca_comps=pca or 45,
                flat_hand_mean=bool(packet['flat_mean']))
        if 'templates' in packet:kw['v_template']=packet['templates'][side]
        model=MANO(str(mano_root),**kw).to(device)
        if side==1 and bool(packet.get('left_shape_fix',False)):
            other=MANO(str(mano_root),is_rhand=True,use_pca=False)
            if (model.shapedirs[:,0].cpu()-other.shapedirs[:,0]).abs().sum()<1:
                model.shapedirs[:,0]*=-1
        parts[side]=np.asarray(MANO_JOINT_TO_PART,np.uint8)[model.lbs_weights.argmax(-1).cpu().numpy()]
        ids=np.flatnonzero(valid[:,side])
        for start in range(0,len(ids),chunk_size):
            chosen=ids[start:start+chunk_size]
            pose=packet['pose'][chosen,side]
            hp=packet['pca_pose'][chosen,side] if pca else pose[:,3:]
            # smplx disables PCA at45 components, whereas DexYCB's manopth
            # convention still applies the full45-component basis.
            if pca==45:hp=(hp@model.np_hand_components).astype(np.float32)
            with torch.no_grad():
                result=model(global_orient=torch.tensor(pose[:,:3],device=device),
                    hand_pose=torch.tensor(hp,device=device),
                    betas=torch.tensor(packet['betas'][chosen,side],device=device))
            v,j=result.vertices.cpu().numpy(),result.joints[:,:16].cpu().numpy()
            if bool(packet['center_wrist']):
                v=v-j[:,:1];j=j-j[:,:1]
            v+=packet['transl'][chosen,side,None]
            j+=packet['transl'][chosen,side,None]
            if str(packet['tip_convention'])=='smplx-cache':tips=[744,320,443,554,671]
            else:tips=[745,317,444 if side==0 else 445,556,673]
            j=np.concatenate((j,v[:,tips]),1)[:,MANO_JOINTS_IN_21]
            world=packet['world'][chosen]
            verts[chosen,side]=np.einsum('nij,nvj->nvi',world[:,:3,:3],v)+world[:,None,:3,3]
            joints[chosen,side]=np.einsum('nij,nvj->nvi',world[:,:3,:3],j)+world[:,None,:3,3]
    jv=np.broadcast_to(valid[...,None],joints.shape[:-1]).copy()
    if bool(packet['resample']):
        count=2*n-1
        def midpoints(x):
            y=np.zeros((count,)+x.shape[1:],x.dtype);y[::2]=x;y[1::2]=(x[:-1]+x[1:])/2
            return y
        joints,verts=midpoints(joints),midpoints(verts)
        vv=np.zeros((count,2),bool);vv[::2]=valid;vv[1::2]=valid[:-1]&valid[1:]
        valid=vv;jv=np.broadcast_to(valid[...,None],joints.shape[:-1]).copy()
    if 'native_joints' in packet:
        shared=valid[...,None]&packet['native_joint_valid']
        if shared.any():
            error=float(np.abs(joints[shared]-packet['native_joints'][shared]).max())
            if error>5e-5:raise ValueError(f'native21 and reconstructed surface convention mismatch: {error:.8f}m')
        joints=packet['native_joints'].astype(np.float32)
        jv=packet['native_joint_valid'].copy()
    if not np.isfinite(joints[jv]).all() or not np.isfinite(verts[valid]).all():
        raise ValueError('nonfinite reconstructed geometry')
    return joints,jv,verts,valid,parts


def validate_alignment(joints, valid, base, tolerance=5e-5,source_id=None):
    # The old Shadow scaffold's thumb-base slot is physically MCP (thhub),
    # whereas human MANO's first thumb point is CMC. Check the actual matching
    # landmark without perpetuating the old mismatch in the new21 targets.
    indices=(0,2,5,9,13,17,4,8,12,16,20) if source_id==4 else SPARSE11_IN_21
    sparse=joints[:,:,indices]
    mask=valid[:,:,indices]&base['hand_valid']
    if not mask.any():raise ValueError('no authoritative overlapping scaffold for alignment audit')
    error=float(np.abs(sparse[mask]-base['hand'][mask]).max())
    if error>tolerance:raise ValueError(f'source/clock/frame convention mismatch: {error:.8f}m')
    return error


def surface_contact(vertices, vertex_parts, queries, radius=.003, falloff=.010):
    """Offline nearest-surface-vertex proximity, explicitly not measured force.

    This KD tree belongs exclusively to preparation. Training reads cached values.
    Sparse MANO vertices approximate the continuous surface; this is not CEDex's
    normal-oriented capsule formula and is recorded under a distinct identity.
    """
    from scipy.spatial import cKDTree
    distance,which=cKDTree(vertices).query(queries,k=1)
    contact=np.exp(-np.maximum(distance-radius,0)/falloff)
    contact[distance>radius+3*falloff]=0
    return contact.astype(np.float16),vertex_parts[which].astype(np.uint8)
