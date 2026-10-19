"""Frozen v5 overlays: no MANO, FK, spatial search or contact computation in training."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import DATA_SCHEMA,INPUT_SEMANTICS,LABEL_SEMANTICS
from .storage import read_json,load_array


def transform(points,pose):
    return np.einsum('...ij,...pj->...pi',pose[...,:3,:3],points)+pose[...,None,:3,3]


def history(values):
    return np.concatenate(((values[3]-values[2])*30,(values[3]-2*values[2]+values[1])*900,values[3]-values[0]),-1)


def sample_window(sequence,geometry,contact,tick,slot,source_id,sample_id):
    n=len(sequence['joints'])
    if not 3<=tick<n-24:raise ValueError('H4/K24 exceeds sequence')
    window=slice(tick-3,tick+25)
    poses=sequence['object_poses'][window,slot]
    known=sequence['object_pose_valid'][window,slot]
    if not known.all():raise ValueError('frozen target poses must remain known')
    anchor=np.linalg.inv(poses[3]).astype(np.float32)
    rel=anchor@poses
    joints=sequence['joints'][window]@anchor[:3,:3].T+anchor[:3,3]
    jvalid=sequence['joint_valid'][window]
    start_valid=jvalid[:4].all(0)
    if not start_valid.any():raise ValueError('no observed joint21 history')
    dense=(sequence['surface_offsets'][tick-3:tick+1].astype(np.float32)
           +sequence['joints'][tick-3:tick+1,:,None,0,:])
    dense=dense@anchor[:3,:3].T+anchor[:3,3]
    dv=np.broadcast_to(sequence['surface_valid'][tick-3:tick+1].all(0)[:,None],(2,778)).copy()
    points=geometry['surface'];normals=geometry['normals'];flow=geometry['query']
    cloud=transform(np.broadcast_to(points,(4,2048,3)),rel[:4])
    features=np.zeros((2048,18),np.float32);features[:,:3]=normals
    features[:,3:12]=history(cloud);features[:,12]=1
    features[:,13]=np.linalg.norm(features[:,9:12],axis=-1)<.002
    future_flow=transform(np.broadcast_to(flow,(24,32,3)),rel[4:])-flow
    future_valid=jvalid[4:]&start_valid[None]
    confidence=contact['contact'][tick+1:tick+25,slot].astype(np.float32)
    parts=contact['part'][tick+1:tick+25,slot].astype(np.int64)
    cv=np.broadcast_to(contact['valid'][tick+1:tick+25,slot,None],(24,2048)).copy()
    return dict(xyz=points.copy(),features=features,point_valid=np.ones(2048,bool),
        start_hand=np.where(start_valid[...,None],joints[3],0.).astype(np.float32),
        start_hand_history=np.where(start_valid[...,None],history(joints[:4]),0.).astype(np.float32),
        start_hand_valid=start_valid,
        trajectory_target=np.where(future_valid[...,None],joints[4:]-joints[3],0.).astype(np.float32),
        trajectory_valid=future_valid,
        dense_hand=np.where(dv[...,None],dense[3],0.).astype(np.float32),
        dense_history=np.where(dv[...,None],history(dense),0.).astype(np.float32),
        dense_hand_valid=dv,dense_hand_part=sequence['surface_parts'].astype(np.int64),
        scene_points=flow[None].copy(),scene_target=future_flow[None].astype(np.float32),
        scene_valid=np.ones((1,24,32),bool),object_valid=np.ones(1,bool),
        start_object_points=flow.copy(),goal_object_points=(flow+future_flow[-1]).astype(np.float32),
        goal_object_delta=future_flow[-1].astype(np.float32),object_goal_valid=np.ones(32,bool),
        interaction_points=points[None].copy(),interaction_normals=normals[None].copy(),
        interaction_query_valid=np.ones((1,2048),bool),
        contact_target=confidence[None],part_target=parts[None],interaction_valid=cv[None],
        source_id=np.int64(source_id),sample_id=np.asarray(sample_id,np.int64))


class V5Windows(Dataset):
    def __init__(self,manifest,split,cache_size=2):
        self.path=Path(manifest).resolve();self.meta=read_json(self.path);self.split=split
        if (self.meta.get('schema')!=DATA_SCHEMA or self.meta.get('status')!='FROZEN'
                or self.meta.get('input_semantics')!=INPUT_SEMANTICS
                or self.meta.get('label_semantics')!=LABEL_SEMANTICS):raise ValueError('frozen independent v5 overlay required')
        if split not in ('train','dev') or not 1<=cache_size<=8:raise ValueError('known split and bounded sequence cache required')
        self.index=load_array(self.meta['indices'][split],mmap_mode='r')
        self.cache=OrderedDict();self.cache_size=cache_size
        geometry=read_json(self.meta['geometry_manifest'])
        self.geometry={r['key']:r for r in geometry['objects']}
        self.geometry_root=Path(self.meta['geometry_manifest']).parent
        source_ids=np.array([r['source_id'] for r in self.meta['sequences']])[self.index[:,0]]
        self.pools={str(s):np.flatnonzero(source_ids==s) for s in self.meta['source_ids']}
        self.pools={k:v for k,v in self.pools.items() if len(v)}

    def __len__(self):return len(self.index)

    def _sequence(self,seq_id):
        if seq_id not in self.cache:
            row=self.meta['sequences'][seq_id];folder=Path(row['path']);cf=Path(row['contact_path'])
            seq={p.stem:load_array(p,mmap_mode='r') for p in folder.glob('*.npy')}
            contact={p.stem:load_array(p,mmap_mode='r') for p in cf.glob('*.npy')}
            self.cache[seq_id]=(seq,contact);self.cache.move_to_end(seq_id)
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return self.cache[seq_id]

    def __getitem__(self,index):
        seq_id,slot,tick=map(int,self.index[int(index)]);row=self.meta['sequences'][seq_id]
        seq,contact=self._sequence(seq_id)
        name=row['objects'][slot];key=f"s{row['source_id']}:{name}" if row['source_id']<3 else name
        geo=self.geometry[key]
        with load_array(self.geometry_root/geo['path'],allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
        return sample_window(seq,geometry,contact,tick,slot,row['source_id'],[seq_id,slot,tick])

    def draw(self,count,seed,step=0,rank=0):
        if self.split!='train':raise ValueError('optimizer sampling must use TRAIN')
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,rank]))
        sources=list(self.pools);weights=np.array([self.meta['source_probabilities'][s] for s in sources]);weights/=weights.sum()
        chosen=rng.choice(sources,count,p=weights);result=np.empty(count,np.int64)
        for sid in sources:
            mask=chosen==sid;result[mask]=rng.choice(self.pools[sid],int(mask.sum()))
        return result.tolist()


def collate(samples):
    if not samples:raise ValueError('empty batch')
    return {k:torch.as_tensor(np.stack([s[k] for s in samples])) for k in samples[0]}
