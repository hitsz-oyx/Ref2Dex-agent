"""Independent frozen WAM feature source; never replaces the common BC reference."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from shadow_finetune import paths
from .fast_bc import FastController


class FeatureController(FastController):
    def __init__(self,checkpoint,env_objects,cache):
        packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.schema=packet['schema'];self.source_step=packet.get('step')
        if self.schema=='ref2dex.ptv3-shared-wam.v3':
            sys.path.insert(0,str(paths.TASK.parent/'ptv3-shared-wam-v3/src'))
            from ptv3_shared_wam_v3.model import SharedWAMV3
            encoder=SharedWAMV3(packet['stats'],packet['config']);weights=packet['model']
        elif self.schema in ('ref2dex.ptv3-shared-wam.v2',paths.SCHEMA):
            from ptv3_shared_wam_v2.model import SharedWAMV2
            encoder=SharedWAMV2(packet['stats'],packet['config']);weights=packet['model']
            if self.schema==paths.SCHEMA:weights={key[len('forecaster.'):]:value for key,value in weights.items() if key.startswith('forecaster.')}
        else:raise ValueError('Own v2/v3 WAM feature checkpoint required')
        encoder.load_state_dict(weights,strict=True)
        self.model=nn.Module();self.model.add_module('forecaster',encoder)
        self.model=self.model.cuda().eval().requires_grad_(False)
        self.history=[];self.invalid=set();self.env_objects=env_objects
        self.expose_wam_features=True;self.features_only=True
        meta=json.loads((Path(cache)/'manifest.json').read_text());lookup={row['object_code']:row['geometry'] for row in meta['records']}
        self.geometry=[]
        for row in env_objects:
            with np.load(lookup[row['object_code']],allow_pickle=False) as data:
                points=data['surface'];normals=data['normals'];length=np.linalg.norm(normals,axis=-1);keep=length>1e-6
                self.geometry.append(dict(surface=points[keep],normals=normals[keep]/length[keep,None]))
