"""Lazy full-sequence views of all nine TRAIN/DEV sources, in physical metres."""
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from pose_mesh_data.windows import PoseMeshWindows

NAMES={0:'OakInk2',1:'GRAB',2:'ARCTIC',4:'UniDexGrasp++ · 仿真采集',5:'TACO',6:'HOT',8:'DexYCB',9:'H2O',10:'HOI4D'}


def read_json(path):
    return json.loads(Path(path).read_text())


def transform(points,pose):
    return np.asarray(points)@pose[:3,:3].T+pose[:3,3]


@dataclass
class Record:
    source_id:int
    row:int
    sequence:str
    split:str
    windows:int
    root:Path
    kind:str
    hint:str=''

    @property
    def key(self):
        return f'{self.source_id}:{self.row}:{self.split}'

    @property
    def label(self):
        return f'{self.row} · {self.hint} · {self.sequence} · {self.split}'


class Catalog:
    def __init__(self,recipe_directory):
        self.recipe=read_json(Path(recipe_directory)/'recipe.json')
        self.main_root=Path(self.recipe['main3_root'])
        self.native_root=Path(self.recipe['native_root'])
        self.main=read_json(self.main_root/'processed/manifest.json')
        self.native=read_json(self.native_root/'manifest.json')
        if self.main.get('status')!='COMPLETED' or self.native.get('status')!='COMPLETED':
            raise ValueError('completed data views required')
        self.records={sid:[] for sid in NAMES};self.indices={};self.sources={}
        for sid,source in enumerate(self.main['sources']):
            root=Path(source['root']);meta=read_json(root/'processed/manifest.json')
            self.sources[sid]=(source,meta)
            for split,old in (('train','train'),('dev','val')):
                rows=np.load(source['indices'][old]['path'],mmap_mode='r')
                self.indices[sid,split]=rows
                ids,counts=np.unique(rows[:,0],return_counts=True)
                for seq,count in zip(ids,counts):
                    self.records[sid].append(Record(sid,int(seq),str(meta['sequences'][seq]),split,int(count),root,'main3'))
        self.geometry={r['key']:r for r in self.native['objects']}
        with np.load(self.native_root/self.native['window_index'],allow_pickle=False) as z:
            for split in ('train','dev'):
                rows=z['verified_'+split].copy()
                ids,counts=np.unique(rows[:,0],return_counts=True)
                for row,count in zip(ids,counts):
                    meta=self.native['sequences'][row];sid=int(meta['source_id'])
                    if sid not in NAMES:continue
                    # Batch path disambiguates repeated simulator episode/env names.
                    batches=[p for p in Path(meta['source_manifest']).parts if p.startswith('batch')]
                    hint=batches[-1] if batches else str(meta.get('profile',''))[:28]
                    self.records[sid].append(Record(sid,int(row),meta['source_sequence'],split,int(count),self.native_root,'native',hint))
                self.indices['native',split]=rows
        for sid in self.records:self.records[sid].sort(key=lambda r:(r.split,r.row))
        # Reuse the canonical compact decoder without invoking an eager dataset constructor.
        self.decoder=PoseMeshWindows.__new__(PoseMeshWindows)
        self.decoder.root=self.native_root;self.decoder.meta=self.native
        self.decoder.cache_size=2;self.decoder.cache=OrderedDict()

    def select(self,source_id,split='all',query=''):
        query=query.strip().lower()
        return [r for r in self.records[source_id] if (split=='all' or r.split==split)
            and (not query or query in r.label.lower())]

    def rows(self,record):
        rows=self.indices[(record.source_id if record.kind=='main3' else 'native'),record.split]
        return np.asarray(rows[rows[:,0]==record.row])

    def load(self,record):
        rows=self.rows(record)
        if record.kind=='main3':
            path=record.root/'processed/sequences'/record.sequence
            meta=read_json(path/'meta.json')
            arrays={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ('hand','hand_valid','poses','pose_valid','frame_ids')}
            clouds=[];queries=[]
            for name in meta['objects']:
                with np.load(record.root/'processed/canonical'/(name+'.npz'),allow_pickle=False) as z:
                    cloud=z['points'].copy()
                clouds.append(cloud);queries.append(cloud[np.linspace(0,len(cloud)-1,32,dtype=int)])
            valid=np.broadcast_to(arrays['hand_valid'][:,:,None],arrays['hand'].shape[:-1])
            if self.sources[record.source_id][0]['kind']=='native':
                # The training SourceWindows replaces raw camera frame IDs here:
                # GRAB/ARCTIC canonical timestamps, not raw sensor IDs, define 30Hz.
                times=np.asarray(np.load(path/'timestamps.npy',mmap_mode='r'),float)
                segment=np.cumsum(np.r_[False,~np.isclose(np.diff(times),1/30,atol=1e-5,rtol=0)])
            else:
                times=(arrays['frame_ids'].astype(float)-arrays['frame_ids'][0])/120
                segment=np.cumsum(np.r_[False,np.diff(arrays['frame_ids'])!=4])
            data=dict(time=times,hand=arrays['hand'],hand_valid=valid,object_poses=arrays['poses'],
                object_pose_valid=arrays['pose_valid'],segment=segment)
            target=int(rows[0,1]);ticks=rows[:,2]
            objects=list(meta['objects']);origin=str(path)
        else:
            row=self.native['sequences'][record.row]
            path=self.native_root/row['path'];stamp=path.stat()
            if (stamp.st_size,stamp.st_mtime_ns)!=(row['payload_size_bytes'],row['payload_mtime_ns']):
                raise ValueError('frozen native payload changed')
            data=self.decoder._sequence(record.row)
            clouds=[];queries=[]
            for key in row['objects']:
                with np.load(self.native_root/self.geometry[key]['path'],allow_pickle=False) as z:
                    cloud=z['surface'].copy()
                    query=z['query'].copy() if 'query' in z else cloud[np.linspace(0,len(cloud)-1,32,dtype=int)]
                clouds.append(cloud);queries.append(query)
            target=int(row['target_slot']);ticks=rows[:,1]
            objects=list(row['objects']);origin=str(path)
        return Sequence(record,data,clouds,queries,objects,target,rows,np.asarray(ticks),origin)


class Sequence:
    def __init__(self,record,data,clouds,queries,objects,target,rows,ticks,origin):
        self.record,self.data,self.clouds,self.queries=record,data,clouds,queries
        self.objects,self.target,self.rows,self.ticks,self.origin=objects,target,rows,ticks,origin
        self.frames=len(data['time']);self.time=np.asarray(data['time'],float)
        self.poses=data['object_poses'];self.known=np.asarray(data['object_pose_valid'],bool)
        centers=np.asarray([p.mean(0) for p in clouds])
        self.centers=np.einsum('toij,oj->toi',self.poses[:,:,:3,:3],centers)+self.poses[:,:,:3,3]
        self.segment=np.asarray(data['segment'])
        dt=np.diff(self.time)
        links=self.known[1:] & self.known[:-1] & (self.segment[1:]==self.segment[:-1])[:,None] & (dt>0)[:,None]
        velocity=np.full_like(self.centers,np.nan,dtype=float)
        velocity[1:]=np.where(links[:,:,None],np.diff(self.centers,axis=0)/np.maximum(dt[:,None,None],1e-12),np.nan)
        self.speed=np.linalg.norm(velocity,axis=-1)
        acceleration=np.full_like(velocity,np.nan)
        acceleration[1:]=np.diff(velocity,axis=0)/np.maximum(dt[:,None,None],1e-12)
        self.acceleration=np.linalg.norm(acceleration,axis=-1)

    def frame(self,t,anchor=None):
        pose=np.eye(4) if anchor is None else anchor
        clouds=[transform(p,pose@self.poses[t,i]) if self.known[t,i] else np.empty((0,3)) for i,p in enumerate(self.clouds)]
        hand=transform(self.data['hand'][t].reshape(-1,3),pose).reshape(2,11,3)
        valid=np.asarray(self.data['hand_valid'][t],bool)
        return clouds,hand,valid

    def path_segments(self,object_id,anchor=None,start=0,end=None):
        end=self.frames if end is None else end
        centers=self.centers[start:end,object_id]
        if anchor is not None:centers=transform(centers,anchor)
        links=(self.known[start:end-1,object_id] & self.known[start+1:end,object_id]
            & (self.segment[start:end-1]==self.segment[start+1:end]))
        return np.stack((centers[:-1],centers[1:]),axis=1)[links].astype(np.float32)

    def horizon_info(self,t,target):
        end=min(t+24,self.frames-1)
        a=transform(self.queries[target],self.poses[t,target])
        maxima=[]
        for f in range(t+1,end+1):
            if self.known[t,target] and self.known[f,target] and self.segment[t]==self.segment[f]:
                maxima.append(float(np.linalg.norm(transform(self.queries[target],self.poses[f,target])-a,axis=-1).max()))
        permitted=bool(((self.rows[:,1]==target)&(self.rows[:,2]==t)).any()) if self.record.kind=='main3' else t in self.ticks
        return dict(end=end,seconds=float(self.time[end]-self.time[t]),
            max_query_displacement_m=max(maxima) if maxima else None,training_window=permitted,
            observed=bool(self.data.get('observed_frame',np.ones(self.frames,bool))[t]),
            speed_mps=float(self.speed[t,target]) if np.isfinite(self.speed[t,target]) else None,
            acceleration_mps2=float(self.acceleration[t,target]) if np.isfinite(self.acceleration[t,target]) else None)
