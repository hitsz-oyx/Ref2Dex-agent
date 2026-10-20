"""Approximate MANO surface reconstruction of canonical11; never an executor."""
import numpy as np

MANO_BASE = (0,13,1,4,10,7)
MANO_TIPS = (744,320,443,554,671)


def forecast_world(current_hand, displacement, world_from_anchor):
    current=np.asarray(current_hand,dtype=np.float64);delta=np.asarray(displacement,dtype=np.float64)
    pose=np.asarray(world_from_anchor,dtype=np.float64)
    if current.shape!=(11,3) or delta.shape!=(24,11,3) or pose.shape!=(4,4):
        raise ValueError('current11/full24 displacement/anchor SE3 required')
    rotation=pose[:3,:3]
    if not np.isfinite(np.r_[current.ravel(),delta.ravel(),pose.ravel()]).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1):
        raise ValueError('finite points and proper anchor rotation required')
    # Displacements receive rotation only; current measured points already carry world translation.
    return np.concatenate((current[None],current[None]+delta@rotation.T),0).astype(np.float32)


def palm_scale(current,neutral):
    # Only the present four non-thumb bases calibrate a fixed shape/scale.
    a=np.asarray(current)[2:6];b=np.asarray(neutral)[2:6]
    ids=np.triu_indices(4,1)
    x=np.linalg.norm(a[:,None]-a[None],axis=-1)[ids]
    y=np.linalg.norm(b[:,None]-b[None],axis=-1)[ids]
    if not np.isfinite(np.r_[x,y]).all() or np.min(y)<1e-6 or np.median(x)<1e-5:raise ValueError('nondegenerate palm bases required')
    raw=float(np.median(x/y))
    return float(np.clip(raw,.6,1.8)),raw


def load_mano(model_root,device):
    for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
        if name not in np.__dict__:setattr(np,name,value)
    from smplx import MANO
    return MANO(str(model_root),is_rhand=True,use_pca=True,num_pca_comps=12,flat_hand_mean=False).to(device).eval().requires_grad_(False)


def fit_mano(target,model_root,*,device='cuda:0',steps=300):
    import torch
    from scipy.spatial.transform import Rotation
    if not str(device).startswith('cuda') or not torch.cuda.is_available():raise ValueError('GPU MANO fit required')
    target=np.asarray(target,np.float32)
    if target.ndim!=3 or target.shape[1:]!=(11,3) or not np.isfinite(target).all() or not 1<=len(target)<=25 or not 50<=steps<=1000:raise ValueError('bounded finite MANO fit required')
    layer=load_mano(model_root,device);n=len(target)
    goal=torch.as_tensor(target,device=device)
    def geometry(root,pose,translation,scale):
        result=layer(global_orient=root,hand_pose=pose,betas=torch.zeros((len(root),10),device=device))
        landmarks=torch.cat((result.joints[:,MANO_BASE],result.vertices[:,MANO_TIPS]),1)
        center=result.joints[:,:1]
        return (result.vertices-center)*scale+translation[:,None],(landmarks-center)*scale+translation[:,None]
    with torch.no_grad():
        zero=torch.zeros((1,3),device=device);pose0=torch.zeros((1,12),device=device)
        _,neutral=geometry(zero,pose0,zero,1.)
    neutral=neutral[0].cpu().numpy();scale,raw_scale=palm_scale(target[0],neutral)
    source=neutral[1:6]-neutral[1:6].mean(0);dest=target[0,1:6]-target[0,1:6].mean(0)
    u,_,vt=np.linalg.svd(source.T@dest);rotation=vt.T@u.T
    if np.linalg.det(rotation)<0:vt[-1]*=-1;rotation=vt.T@u.T
    root=torch.nn.Parameter(torch.as_tensor(Rotation.from_matrix(rotation).as_rotvec()[None],device=device,dtype=torch.float32))
    pose=torch.nn.Parameter(torch.zeros((1,12),device=device));translation=torch.nn.Parameter(goal[:1,0].clone())
    weights=torch.ones((1,11,1),device=device);weights[:,0]=4
    opt=torch.optim.Adam([root,pose,translation],lr=.025)
    for _ in range(180):
        _,points=geometry(root,pose,translation,scale)
        loss=(((points-goal[:1])/.02).square()*weights).mean()+.002*pose.square().mean()
        opt.zero_grad();loss.backward();opt.step()
    root=torch.nn.Parameter(root.detach().expand(n,-1).clone());pose=torch.nn.Parameter(pose.detach().expand(n,-1).clone())
    translation=torch.nn.Parameter(goal[:,0].clone());opt=torch.optim.Adam([root,pose,translation],lr=.008)
    for _ in range(steps):
        _,points=geometry(root,pose,translation,scale)
        loss=(((points-goal)/.02).square()*weights).mean()+.002*pose.square().mean()
        if n>1:loss=loss+.02*(pose[1:]-pose[:-1]).square().mean()+.02*(root[1:]-root[:-1]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():vertices,points=geometry(root,pose,translation,scale)
    vertices=vertices.cpu().numpy();points=points.cpu().numpy();errors=np.linalg.norm(points-target,axis=-1)
    if not np.isfinite(vertices).all():raise ValueError('nonfinite MANO fit')
    return dict(vertices=vertices,faces=np.asarray(layer.faces,np.int32),landmarks=points,errors_m=errors,
        root=root.detach().cpu().numpy(),pca_pose=pose.detach().cpu().numpy(),translation=translation.detach().cpu().numpy(),
        scale=scale,raw_scale=raw_scale,fit_mean_mm=float(errors.mean()*1000),fit_max_mm=float(errors.max()*1000),
        shape='neutral betas fixed; scale calibrated from current palm only',pose='12D MANO PCA with pose/temporal priors; approximate, no collision constraint')
