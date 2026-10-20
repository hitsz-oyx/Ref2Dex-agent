"""Self-trained current300D Shadow reference; no official policy weights loaded."""
import torch
from torch import nn

SCHEMA='ref2dex.shadow-current-bc-mlp.v1'
SCHEDULED_SCHEMA='ref2dex.shadow-scheduled-current-bc-mlp.v1'
SCHEDULED_DIM=565  # current300 + world hand tau8x11x3 + observed-history validity


class CurrentMLP(nn.Module):
    def __init__(self,input_dim=300):
        super().__init__()
        if input_dim not in (300,SCHEDULED_DIM):raise ValueError('current300 or scheduled565 required')
        self.input_dim=input_dim
        for name,size in [('x',input_dim),('y',24)]:
            self.register_buffer(name+'_mean',torch.zeros(size))
            self.register_buffer(name+'_scale',torch.ones(size))
        widths=(input_dim,1024,1024,512,512,24)
        layers=[]
        for index,(lo,hi) in enumerate(zip(widths,widths[1:])):
            layers.append(nn.Linear(lo,hi))
            if index<4:layers.append(nn.ELU())
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,observation):
        if observation.shape[-1]!=self.input_dim:raise ValueError('BC observation dimension mismatch')
        observed=(observation.clamp(-5,5) if self.input_dim==300 else
                  torch.cat((observation[...,:300].clamp(-5,5),observation[...,300:]),-1))
        x=(observed-self.x_mean)/self.x_scale
        return self.net(x)*self.y_scale+self.y_mean

    def initialize_current(self,packet):
        """Model-only expansion of our full-data BC, with zero new input weights."""
        if self.input_dim!=SCHEDULED_DIM or packet.get('schema')!=SCHEMA:
            raise ValueError('scheduled model and self-trained current BC parent required')
        state=self.state_dict()
        for key,value in packet['model'].items():
            if key=='net.0.weight':
                state[key].zero_();state[key][:,:300].copy_(value)
            elif key in ('x_mean','x_scale'):
                state[key][:300].copy_(value)
            else:state[key].copy_(value)
        self.load_state_dict(state,strict=True)


class MLPController:
    def __init__(self,checkpoint,env_objects,teacher,cache):
        packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if packet.get('schema')!=SCHEMA:raise ValueError('Self-trained current300 BC checkpoint required')
        self.model=CurrentMLP().cuda().eval();self.model.load_state_dict(packet['model'],strict=True)
        self.hidden_model=self.model.net[:-1].eval()
        self.invalid=set()

    @torch.no_grad()
    def act(self,snapshot,observation,tick):
        normalized=(observation.clamp(-5,5)-self.model.x_mean)/self.model.x_scale
        self.last_features=self.hidden_model(normalized).detach()
        return (self.model.net[-1](self.last_features)*self.model.y_scale+self.model.y_mean).clamp(-1,1)
