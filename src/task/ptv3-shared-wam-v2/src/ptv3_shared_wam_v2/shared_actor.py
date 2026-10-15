"""Matched query actor for predicted trajectory and/or shared-token interfaces."""
import torch
from torch import nn
from torch.nn import functional as F

INTERFACES=('pred_tau','shared','shared_tau')


class SharedQueryActor(nn.Module):
    def __init__(self,interface,shared_width=768,width=128,action_dim=15):
        super().__init__()
        if interface not in INTERFACES or shared_width<33 or action_dim not in (12,15):raise ValueError('actor interface/width/action dimension')
        self.interface=interface;self.shared_width=shared_width
        for key,shape in [('state',(87,)),('tau',(24,33)),('action',(8,action_dim))]:
            self.register_buffer(key+'_mean',torch.zeros(shape))
            self.register_buffer(key+'_scale',torch.ones(shape))
        self.state_encoder=nn.Sequential(nn.Linear(87,width),nn.GELU(),nn.Linear(width,width))
        self.memory_encoder=nn.Sequential(nn.Linear(shared_width,width),nn.GELU(),nn.Linear(width,width))
        self.step_embedding=nn.Parameter(torch.randn(24,width)*.02)
        self.type_embedding=nn.Parameter(torch.randn(3,width)*.02)
        self.action_queries=nn.Parameter(torch.randn(8,width)*.02)
        layer=nn.TransformerDecoderLayer(width,4,width*4,dropout=0.,activation='gelu',batch_first=True)
        self.decoder=nn.TransformerDecoder(layer,2)
        self.head=nn.Linear(width,action_dim)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,state,tau=None,shared=None,shared_valid=None):
        if state.shape!=(len(state),87):raise ValueError('actor requires current87D state')
        memory=[self.state_encoder((state-self.state_mean)/self.state_scale)[:,None]+self.type_embedding[0]]
        masks=[torch.ones(len(state),1,dtype=torch.bool,device=state.device)]
        if self.interface in ('pred_tau','shared_tau'):
            if tau is None or tau.shape!=(len(state),24,33):raise ValueError('predicted native tau required')
            normalized=(tau-self.tau_mean)/self.tau_scale
            memory.append(self.memory_encoder(F.pad(normalized,(0,self.shared_width-33)))+
                          self.step_embedding+self.type_embedding[1])
            masks.append(torch.ones(len(state),24,dtype=torch.bool,device=state.device))
        if self.interface in ('shared','shared_tau'):
            if (shared is None or shared_valid is None or shared.shape[:2]!=shared_valid.shape
                    or shared.shape[0]!=len(state) or shared.shape[-1]!=self.shared_width):
                raise ValueError('shared tokens and valid mask required')
            # Invalid payloads never enter projections (even NaNs in padded slots).
            memory.append(self.memory_encoder(torch.where(shared_valid[...,None],shared,0.))+
                          self.type_embedding[2])
            masks.append(shared_valid)
        memory=torch.cat(memory,1);valid=torch.cat(masks,1)
        queries=self.action_queries[None].expand(len(state),-1,-1)
        out=self.decoder(queries,memory,memory_key_padding_mask=~valid)
        return self.head(out)*self.action_scale+self.action_mean
