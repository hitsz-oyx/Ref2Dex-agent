"""Causal v6 inference from measured sparse11; unobserved joints/skin are invalid."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import paths
from .model import SharedWAMV6
from .data import collate
from ptv3_shared_wam_v5.data import history, transform
from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21
from .executor_bridge import full21_to_sparse11


def voxel_config(config):
    """Require checkpoint voxel coordinates; never guess training-time defaults."""
    missing = [key for key in ('voxel_origin_m', 'voxel_m') if key not in config]
    if missing:
        raise ValueError('v6 runtime config missing required key(s): '+', '.join(missing))
    try:
        origin = np.asarray(config['voxel_origin_m'], dtype=float)
        size = np.asarray(config['voxel_m'], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('v6 runtime config requires numeric voxel_origin_m/voxel_m') from error
    if origin.shape != (3,) or not np.isfinite(origin).all() or size.shape != () or not np.isfinite(size) or size <= 0:
        raise ValueError('v6 runtime config requires finite voxel_origin_m[3] and positive scalar voxel_m')
    return origin, float(size)


class EncodingUnavailable(ValueError):
    """A state/forecast cannot be encoded; mask tau without judging physics."""

def online(geo, poses, joints, dense, joint_valid, dense_valid, parts, goal, event=None):
    """Only H4 observed/generated geometry and instruction inputs; no future labels."""
    anchor = np.linalg.inv(poses[-1]); local = joints@anchor[:3,:3].T+anchor[:3,3]
    skin = dense@anchor[:3,:3].T+anchor[:3,3]
    cloud = transform(np.broadcast_to(geo['surface'], (4, len(geo['surface']), 3)), anchor@poses)
    features = np.zeros((len(geo['surface']),18),np.float32)
    features[:,:3]=geo['normals'];features[:,3:12]=history(cloud);features[:,12]=1
    features[:,13]=np.linalg.norm(features[:,9:12],axis=-1)<.002
    flow=geo['query'];relative=anchor@goal;endpoint=flow@relative[:3,:3].T+relative[:3,3]
    sample=dict(xyz=geo['surface'].astype(np.float32),features=features,point_valid=np.ones(len(cloud[0]),bool),
        start_hand=np.where(joint_valid[...,None],local[-1],0).astype(np.float32),
        start_hand_history=np.where(joint_valid[...,None],history(local),0).astype(np.float32),start_hand_valid=joint_valid,
        dense_hand=np.where(dense_valid[...,None],skin[-1],0).astype(np.float32),
        dense_history=np.where(dense_valid[...,None],history(skin),0).astype(np.float32),dense_hand_valid=dense_valid,
        dense_hand_part=parts.astype(np.int64),scene_points=flow[None].astype(np.float32),object_valid=np.ones(1,bool),
        start_object_points=flow.astype(np.float32),goal_object_points=endpoint.astype(np.float32),
        goal_object_delta=(endpoint-flow).astype(np.float32),object_goal_valid=np.ones(32,bool),
        interaction_points=geo['surface'][None].astype(np.float32),interaction_normals=geo['normals'][None].astype(np.float32),
        interaction_query_valid=np.ones((1,len(geo['surface'])),bool),
        event_type=np.int64(0),event_delta_steps=np.float32(0),event_valid=np.bool_(False),event_hz=np.int64(30))
    if event:
        sample.update({k:np.asarray(v,dtype=sample[k].dtype) for k,v in event.items()})
    return sample


class OnlineV6Runtime:
    def __init__(self, checkpoint, stats, geometry_cache, device="cuda:0", seed=185, inference_batch=8, forecast_only=False):
        self.device=torch.device(device)
        self.inference_batch=int(inference_batch);self.forecast_only=bool(forecast_only)
        if not 1<=self.inference_batch<=96:raise ValueError("bounded inference batch required")
        packet=torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.config=packet["config"]; self.step=packet["step"]
        voxel_config(self.config)
        self.model=SharedWAMV6(json.loads(Path(stats).read_text()),self.config).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(packet["model"],strict=True)
        self.root=Path(geometry_cache)
        manifest=json.loads((self.root/"surface-manifest.json").read_text())
        self.geometries={}
        for row in manifest["objects"]:
            with np.load(self.root/row["path"]) as z:
                self.geometries[row["key"]]={k:z[k].copy() for k in z.files}
        self.calls=0;self.seconds=0.;self.samples=0

    def _synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _trajectory(self, batch):
        if not self.forecast_only:return self.model(batch)["trajectory"]
        # Exact hand path of forward(), omitting the unused independent C head.
        model=self.model
        shared,shared_valid,geometry=model.encode(batch)
        offset=geometry["tokens"].shape[1]
        hand_z=model.hand_future(shared[:,offset:offset+42]).reshape(len(shared),24,2,21,3)
        if model.config.get("flow_matching"):
            from ptv3_shared_wam.model import clean
            points=batch["scene_points"];valid=batch["object_valid"].bool()[...,None].expand(*points.shape[:-1])
            query=model.scene_query(clean(points,valid)/model.position_scale).flatten(1,2)
            context=model.scene_context(shared)
            update,_=model.scene_readout(query,context,context,key_padding_mask=~shared_valid,need_weights=False)
            latent=model.scene_norm(query+update).reshape(len(points),points.shape[1],points.shape[2],-1)
            scene_z=model.scene_future(latent.flatten(1,2)).reshape(len(points),24,*points.shape[1:3],3).transpose(1,2)
            hand_z,_,_=model.flow_future(hand_z,scene_z,shared,shared_valid,batch,model.normalizer)
        return torch.where(batch["start_hand_valid"][:,None,...,None],model.normalizer.denormalize("hand",hand_z),0.)

    def workspace_valid(self, objects, hands):
        from scipy.spatial.transform import Rotation
        origin, size = voxel_config(self.config)
        objects=np.asarray(objects);hands=np.asarray(hands)
        rotation=Rotation.from_quat(objects[:,-1,3:7]).as_matrix()
        local=np.einsum("bhki,bij->bhkj",hands-objects[:,-1,None,None,:3],rotation)
        grid=np.floor((local[:,-1]-origin)/size)
        return np.isfinite(grid).all((1,2)) & (grid>=0).all((1,2)) & (grid<=65535).all((1,2))

    def predict(self, objects, hands, goals, events, object_codes):
        from scipy.spatial.transform import Rotation
        origin, size = voxel_config(self.config)
        objects=np.asarray(objects);hands=np.asarray(hands);goals=np.asarray(goals)
        if objects.shape[1:]!=(4,7) or hands.shape!=(len(objects),4,11,3):
            raise ValueError("measured Bx4 objects/hand11 required")
        if not len(objects) or goals.shape != (len(objects),4,4) or len(events) != len(objects) or len(object_codes) != len(objects):
            raise ValueError('one goal/event/object code per nonempty H4 batch required')
        ids=np.asarray(SPARSE11_IN_21);samples=[];rotations=[]
        for i,code in enumerate(object_codes):
            geo=self.geometries[code if "@" in code else code+"@1"]
            if geo["surface"].shape!=(2048,3):raise ValueError("v6 needs genuine cached dense2048 surface")
            poses=np.broadcast_to(np.eye(4),(4,4,4)).copy()
            poses[:,:3,:3]=Rotation.from_quat(objects[i,:,3:7]).as_matrix();poses[:,:3,3]=objects[i,:,:3]
            joints=np.zeros((4,2,21,3),np.float32);joints[:,0,ids]=hands[i]
            valid=np.zeros((2,21),bool);valid[0,ids]=True
            sample=online(geo,poses,joints,np.zeros((4,2,778,3),np.float32),valid,
                np.zeros((2,778),bool),np.zeros((2,778),np.int64),goals[i],events[i])
            xyz=np.r_[sample["xyz"],sample["start_hand"][valid]]
            grid=np.floor((xyz-origin)/size)
            if not np.isfinite(grid).all() or grid.min()<0 or grid.max()>65535:
                raise EncodingUnavailable("measured state outside v6 voxel workspace")
            samples.append(sample);rotations.append(poses[-1,:3,:3])
        self._synchronize()
        started=time.monotonic()
        chunks=[]
        for offset in range(0,len(samples),self.inference_batch):
            batch={k:v.to(self.device) for k,v in collate(samples[offset:offset+self.inference_batch]).items()}
            with torch.inference_mode(),torch.autocast("cuda",enabled=False):
                chunks.append(full21_to_sparse11(self._trajectory(batch))[:,:,0].float().cpu().numpy())
        prediction=np.concatenate(chunks)
        self._synchronize()
        self.seconds+=time.monotonic()-started;self.calls+=1;self.samples+=len(objects)
        result=np.einsum("btki,bji->btkj",prediction,np.asarray(rotations))
        if not np.isfinite(result).all():raise EncodingUnavailable("nonfinite v6 world tau")
        return result.astype(np.float32)
