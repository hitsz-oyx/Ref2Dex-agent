"""New native Shadow action head; pretrained WAM parameters remain load-compatible."""
import torch
from torch import nn
from . import paths
from ptv3_shared_wam_v2.model import SharedWAMV2
from ptv3_shared_wam_v2.objectives import forecast_loss


class ShadowActor(nn.Module):
    def __init__(self,shared_width=768,width=128):
        super().__init__()
        for key,shape in [('state',(300,)),('action',(8,24))]:
            self.register_buffer(key+'_mean',torch.zeros(shape));self.register_buffer(key+'_scale',torch.ones(shape))
        self.state=nn.Sequential(nn.Linear(300,width),nn.GELU(),nn.Linear(width,width))
        self.shared=nn.Sequential(nn.Linear(shared_width,width),nn.GELU(),nn.Linear(width,width))
        self.queries=nn.Parameter(torch.randn(8,width)*.02)
        layer=nn.TransformerDecoderLayer(width,4,width*4,dropout=0.,activation='gelu',batch_first=True)
        self.decoder=nn.TransformerDecoder(layer,2);self.head=nn.Linear(width,24)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,state,shared,valid):
        token=self.state((state-self.state_mean)/self.state_scale)[:,None]
        memory=torch.cat((token,self.shared(torch.where(valid[...,None],shared,0.))),1)
        mask=torch.cat((torch.ones(len(state),1,dtype=torch.bool,device=state.device),valid),1)
        z=self.decoder(self.queries[None].expand(len(state),-1,-1),memory,memory_key_padding_mask=~mask)
        return self.head(z)*self.action_scale+self.action_mean


class ShadowPolicy(nn.Module):
    def __init__(self,stats,config,geometry=None):
        super().__init__();self.forecaster=SharedWAMV2(stats,config,geometry)
        self.actor=ShadowActor(config['shared_width'])

    def forward(self,batch):
        p=self.forecaster(batch,return_shared=True)
        p['action']=self.actor(batch['policy_state'],p.pop('shared_tokens'),p.pop('shared_valid'))
        return p

    def groups(self):
        groups=self.forecaster.parameter_groups(5e-6,1e-5)
        return groups+[dict(name='shadow_actor',params=list(self.actor.parameters()),lr=5e-4)]


def loss(prediction,batch,config,model):
    total,terms=forecast_loss(prediction,batch,config,model.forecaster.normalizer)
    error=(prediction['action']-batch['action_target'])/model.actor.action_scale
    action=error.abs().mean()
    return total+action,dict(terms,action=action,root=error[...,:6].abs().mean(),finger=error[...,6:].abs().mean())
