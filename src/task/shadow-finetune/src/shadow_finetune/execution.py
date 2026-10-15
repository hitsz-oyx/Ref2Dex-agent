"""Native60Hz replanning; same feature construction as offline supervision."""
import json
from pathlib import Path
import numpy as np
import torch
from . import paths
from .model import ShadowPolicy
from .data import current_inputs,collate
from video_data_preparation.unidexgrasp_data import object_poses


class Controller:
    def __init__(self,checkpoint,env_objects,teacher,cache):
        self.teacher=teacher;self.history=[];self.invalid=set();self.env_objects=env_objects
        self.model=None
        if checkpoint:
            packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if packet.get('schema')!=paths.SCHEMA:raise ValueError('Shadow checkpoint required')
            self.model=ShadowPolicy(packet['stats'],packet['config']).cuda().eval()
            self.model.load_state_dict(packet['model'],strict=True)
            meta=json.loads((Path(cache)/'manifest.json').read_text());lookup={r['object_code']:r['geometry'] for r in meta['records']}
            self.geometry=[]
            for row in env_objects:
                with np.load(lookup[row['object_code']],allow_pickle=False) as f:
                    points=f['surface'];normals=f['normals'];length=np.linalg.norm(normals,axis=-1);keep=length>1e-6
                    self.geometry.append(dict(surface=points[keep],normals=normals[keep]/length[keep,None]))

    def act(self,snapshot,observation,tick):
        if tick==1:self.history=[];self.invalid=set()
        if self.model is None:return self.teacher(observation.clamp(-5,5)).clamp(-1,1)
        self.history.append(snapshot)
        past=[max(len(self.history)-1+offset,0) for offset in (-6,-4,-2,0)]
        samples=[];indices=[]
        for env,geometry in enumerate(self.geometry):
            if env in self.invalid:continue
            obj=object_poses(np.stack([self.history[i]['object'][env] for i in past]))
            hand=np.stack([self.history[i]['hand'][env] for i in past])
            sample=current_inputs(obj,hand,snapshot['goal'][env],geometry)
            if (sample['start_hand'][0]<-2).any() or not np.isfinite(sample['start_hand']).all():
                self.invalid.add(env);continue
            sample.update(policy_state=np.clip(snapshot['observation'][env],-5,5),
                trajectory_target=np.zeros((24,2,11,3),np.float32),trajectory_valid=np.zeros((24,2,11),bool),
                scene_target=np.zeros((1,24,32,3),np.float32),scene_valid=np.zeros((1,24,32),bool),
                action_target=np.zeros((8,24),np.float32))
            samples.append(sample);indices.append(env)
        action=torch.zeros(len(self.env_objects),24,device='cuda')
        if samples:
            batch={k:v.cuda() for k,v in collate(samples).items()}
            prediction=self.model(batch)['action'][:,0]
            if not torch.isfinite(prediction).all():raise FloatingPointError('nonfinite policy command')
            action[indices]=prediction.clamp(-1,1)
        return action
