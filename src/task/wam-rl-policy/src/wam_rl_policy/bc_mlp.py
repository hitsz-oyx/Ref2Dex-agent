"""Self-trained current300D Shadow reference; no official policy weights loaded."""
import torch
from torch import nn

SCHEMA='ref2dex.shadow-current-bc-mlp.v1'


class CurrentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        for name,size in [('x',300),('y',24)]:
            self.register_buffer(name+'_mean',torch.zeros(size))
            self.register_buffer(name+'_scale',torch.ones(size))
        widths=(300,1024,1024,512,512,24)
        layers=[]
        for index,(lo,hi) in enumerate(zip(widths,widths[1:])):
            layers.append(nn.Linear(lo,hi))
            if index<4:layers.append(nn.ELU())
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,observation):
        x=(observation.clamp(-5,5)-self.x_mean)/self.x_scale
        return self.net(x)*self.y_scale+self.y_mean


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
