"""Direct state-outcome catalogue control, without motor-target physics input."""
import torch
from torch import nn


class NativeStateOutcomePolicy(nn.Module):
    def __init__(self,physical_dim,native_dim):
        super().__init__();self.history=nn.GRU(69,64,batch_first=True)
        self.physical=nn.Sequential(nn.Linear(physical_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.native=nn.Sequential(nn.Linear(native_dim,64),nn.SiLU(),nn.LayerNorm(64))
        self.head=nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,24))

    def forward(self,history,physical,native,prior):
        _,h=self.history(history);x=torch.cat((h[-1],self.physical(physical),self.native(native)),-1)
        return self.head(x).reshape(-1,8,3)+prior[:,None,[5,3,4]]
