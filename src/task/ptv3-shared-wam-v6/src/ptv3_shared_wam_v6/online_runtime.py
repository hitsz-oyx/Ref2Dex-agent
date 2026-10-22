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
    def __init__(self, checkpoint, stats, geometry_cache, device="cuda:0", seed=185):
        self.device=torch.device(device)
        packet=torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.config=packet["config"]; self.step=packet["step"]
        self.model=SharedWAMV6(json.loads(Path(stats).read_text()),self.config).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(packet["model"],strict=True)
        self.root=Path(geometry_cache)
        manifest=json.loads((self.root/"surface-manifest.json").read_text())
        self.geometries={}
        for row in manifest["objects"]:
            with np.load(self.root/row["path"]) as z:
                self.geometries[row["key"]]={k:z[k].copy() for k in z.files}
        self.calls=0;self.seconds=0.;self.samples=0

    def predict(self, objects, hands, goals, events, object_codes):
        from scipy.spatial.transform import Rotation
        objects=np.asarray(objects);hands=np.asarray(hands);goals=np.asarray(goals)
        if objects.shape[1:]!=(4,7) or hands.shape!=(len(objects),4,11,3):
            raise ValueError("measured Bx4 objects/hand11 required")
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
            grid=np.floor((xyz-np.asarray(self.config["voxel_origin_m"]))/self.config["voxel_m"])
            if not np.isfinite(grid).all() or grid.min()<0 or grid.max()>65535:
                raise ValueError("measured state outside v6 voxel workspace")
            samples.append(sample);rotations.append(poses[-1,:3,:3])
        torch.cuda.synchronize(self.device);started=time.monotonic()
        chunks=[]
        for offset in range(0,len(samples),8):
            batch={k:v.to(self.device) for k,v in collate(samples[offset:offset+8]).items()}
            with torch.inference_mode(),torch.autocast("cuda",enabled=False):
                chunks.append(self.model(batch)["trajectory"][:,:,0][:,:,ids].float().cpu().numpy())
        prediction=np.concatenate(chunks)
        torch.cuda.synchronize(self.device);self.seconds+=time.monotonic()-started;self.calls+=1;self.samples+=len(objects)
        result=np.einsum("btki,bji->btkj",prediction,np.asarray(rotations))
        if not np.isfinite(result).all():raise ValueError("nonfinite v6 world tau")
        return result.astype(np.float32)
