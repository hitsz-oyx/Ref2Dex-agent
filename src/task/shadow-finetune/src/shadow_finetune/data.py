"""Same raw control clock for BC and execution; WAM H4/K24 stays at30Hz."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from . import paths
from video_data_preparation.unidexgrasp_data import object_poses

STACK=('start_hand','start_hand_history','start_hand_valid','object_valid','scene_points',
       'start_object_points','goal_object_delta','object_goal_valid',
       'trajectory_target','trajectory_valid','scene_target','scene_valid','policy_state','action_target')


def current_inputs(obj_history,hand_history,goal_world,geometry):
    """Only observed H4 and decision-known task goal; no future-state oracle."""
    if obj_history.shape!=(4,4,4) or hand_history.shape!=(4,11,3):raise ValueError('H4 contract')
    now=obj_history[-1];r=now[:3,:3];center=now[:3,3]
    points=geometry['surface'];normals=geometry['normals']
    if points.shape!=normals.shape or points.shape[1:]!=(3,):raise ValueError('geometry')
    hand=(hand_history-center)@r
    poses=np.linalg.inv(now)@obj_history
    moved=np.einsum('tij,pj->tpi',poses[:,:3,:3],points)+poses[:,None,:3,3]
    dynamics=lambda x:np.concatenate(((x[-1]-x[-2])*30,(x[-1]-2*x[-2]+x[-3])*900,x[-1]-x[0]),-1)
    features=np.zeros((len(points),18),np.float32);features[:,:3]=normals
    features[:,3:12]=dynamics(moved);features[:,12]=1
    start=np.zeros((2,11,3),np.float32);start[0]=hand[-1]
    history=np.zeros((2,11,9),np.float32);history[0]=dynamics(hand)
    valid=np.zeros((2,11),bool);valid[0]=True
    query=points[np.linspace(0,len(points)-1,32,dtype=int)].astype(np.float32)
    delta=(np.asarray(goal_world)-center)@r
    return dict(xyz=points.astype(np.float32),features=features,start_hand=start,start_hand_history=history,
        start_hand_valid=valid,object_valid=np.ones(1,bool),scene_points=query[None],start_object_points=query,
        # Keep current orientation; the original task specifies only goal position.
        goal_object_delta=np.broadcast_to(delta,query.shape).astype(np.float32).copy(),object_goal_valid=np.ones(32,bool))


def sample_sequence(sequence,geometry,tick,supervised=True):
    if not 0<=tick<len(sequence['hand']):raise ValueError('state index')
    past=np.maximum(tick+np.array([-6,-4,-2,0]),0)
    obj=sequence['poses'];hand=sequence['hand']
    sample=current_inputs(obj[past],hand[past],sequence['goal'][tick],geometry)
    sample['policy_state']=np.clip(sequence['observation'][tick],-5,5).astype(np.float32)
    sample.update(trajectory_target=np.zeros((24,2,11,3),np.float32),
        trajectory_valid=np.zeros((24,2,11),bool),scene_target=np.zeros((1,24,32,3),np.float32),
        scene_valid=np.zeros((1,24,32),bool),action_target=np.zeros((8,24),np.float32))
    if supervised:
        future=tick+np.arange(1,25)*2
        if future[-1]>=len(hand) or tick+8>len(sequence['action']):raise ValueError('complete future/action support')
        r=obj[tick,:3,:3];center=obj[tick,:3,3]
        sample['trajectory_target'][:,0]=(hand[future]-hand[tick])@r
        sample['trajectory_valid'][:,0]=True
        query=sample['scene_points'][0];relative=np.linalg.inv(obj[tick])@obj[future]
        moving=np.einsum('tij,pj->tpi',relative[:,:3,:3],query)+relative[:,None,:3,3]
        sample['scene_target'][0]=(moving-query).astype(np.float32);sample['scene_valid'][:]=True
        sample['action_target']=sequence['action'][tick:tick+8].astype(np.float32).copy()
    return sample


def collate(samples):
    out={key:torch.as_tensor(np.stack([s[key] for s in samples])) for key in STACK}
    n=max(len(s['xyz']) for s in samples);b=len(samples)
    out.update(xyz=torch.zeros(b,n,3),features=torch.zeros(b,n,18),point_valid=torch.zeros(b,n,dtype=torch.bool))
    for i,s in enumerate(samples):
        count=len(s['xyz']);out['xyz'][i,:count]=torch.as_tensor(s['xyz'])
        out['features'][i,:count]=torch.as_tensor(s['features']);out['point_valid'][i,:count]=True
    return out


class ShadowWindows:
    def __init__(self,root,split,limit_episodes=None):
        self.root=Path(root);self.meta=json.loads((self.root/'manifest.json').read_text())
        if self.meta.get('schema')!=paths.SCHEMA or self.meta.get('status')!='COMPLETED':raise ValueError('admitted Shadow cache')
        self.split=split;self.records=[r for r in self.meta['records'] if r['split']==split]
        if limit_episodes:self.records=self.records[:limit_episodes]
        self.keys=[];self.pools=[]
        for i,row in enumerate(self.records):
            start=len(self.keys);self.keys.extend((i,int(t)) for t in row['ticks'])
            self.pools.append(np.arange(start,len(self.keys)))
        self.pools=[p for p in self.pools if len(p)]
        if not self.keys:raise ValueError('nonempty Shadow split')

    @lru_cache(maxsize=128)
    def sequence(self,index):
        with np.load(self.root/self.records[index]['path'],allow_pickle=False) as f:return {k:f[k] for k in f.files}

    @lru_cache(maxsize=64)
    def geometry(self,index):
        with np.load(self.records[index]['geometry'],allow_pickle=False) as f:
            points=f['surface'].copy();normals=f['normals'].copy()
        length=np.linalg.norm(normals,axis=-1)
        keep=length>1e-6
        points=points[keep];normals=normals[keep]/length[keep,None]
        if len(points)<32:raise ValueError('insufficient nonzero-normal CAD points')
        return dict(surface=points,normals=normals)

    def __len__(self):return len(self.keys)
    def __getitem__(self,index):
        seq,tick=self.keys[int(index)]
        return sample_sequence(self.sequence(seq),self.geometry(seq),tick)

    def draw(self,rng,size):
        return [int(rng.choice(self.pools[int(rng.integers(len(self.pools)))])) for _ in range(size)]

    def draw_startup(self,rng,size,fraction):
        """Mixture of the original draw and TRAIN-only sequence-uniform tick0..7."""
        if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('startup fraction in [0,1]')
        if fraction==0:return self.draw(rng,size)
        if self.split!='train':raise ValueError('startup sampling is TRAIN-only')
        if not hasattr(self,'_startup_pools'):
            self._startup_pools=[p[[self.keys[int(i)][1]<8 for i in p]] for p in self.pools]
            self._startup_pools=[p for p in self._startup_pools if len(p)]
        if not self._startup_pools:raise ValueError('no admitted startup queries')
        result=[]
        for _ in range(size):
            pools=self._startup_pools if rng.random()<fraction else self.pools
            result.append(int(rng.choice(pools[int(rng.integers(len(pools)))])))
        return result

    def normalization(self):
        if self.split!='train':raise ValueError('TRAIN normalization only')
        state=[];action=[]
        for i,row in enumerate(self.records):
            s=self.sequence(i);ticks=np.asarray(row['ticks'])
            if len(ticks):
                state.append(np.clip(s['observation'][ticks],-5,5))
                action.append(s['action'][ticks[:,None]+np.arange(8)])
        s=np.concatenate(state);a=np.concatenate(action)
        return dict(state_mean=s.mean(0),state_scale=np.maximum(s.std(0),.05),
                    action_mean=a.mean(0),action_scale=np.maximum(a.std(0),.01))
