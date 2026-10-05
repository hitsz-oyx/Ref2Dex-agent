"""Anatomical point-flow actions and centered factual consequence innovations."""
from hashlib import sha256
import torch
from torch import nn


def finger_summary(flow):
    if flow.ndim!=4 or flow.shape[-2:]!=(120,3):
        raise ValueError('expected [windows,15,120,3] with24points/finger')
    grouped=flow.reshape(*flow.shape[:2],5,24,3)
    mean=grouped.mean(-2).flatten(2)
    rms=grouped.square().mean(-2).sqrt().flatten(2)
    return torch.nn.functional.pad(torch.cat((mean,rms),-1)/.02,(0,2))


class RelativeFingerHead(nn.Module):
    def __init__(self,h_width,centered):
        super().__init__();self.centered=centered
        self.net=nn.Sequential(nn.Linear(h_width+32,64),nn.Tanh(),
                               nn.Linear(64,32),nn.Tanh(),nn.Linear(32,26))

    def forward(self,h,action):
        if action.shape!=(len(h),15,32):raise ValueError('recipient15candidate actions required')
        x=torch.cat((h[:,None].expand(-1,15,-1),action),-1)
        value=self.net(x.flatten(0,1)).reshape(len(h),15,26)
        return value-value.mean(1,keepdim=True) if self.centered else value


@torch.no_grad()
def evaluate_all(model,h,action):
    return torch.cat([model(h[ids],action[ids]) for ids in torch.arange(len(h),device=h.device).split(64)])


def fit_head(h,action,arms,target,fit_ids,centered,updates=300):
    torch.manual_seed(245);model=RelativeFingerHead(h.shape[1],centered).to(h.device)
    initial=sha256(torch.cat([v.detach().flatten() for v in model.parameters()]).cpu().numpy().tobytes()).hexdigest()
    generator=torch.Generator(device=h.device).manual_seed(246)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]
    for _ in range(updates):
        ids=fit_ids[torch.randint(len(fit_ids),(64,),generator=generator,device=h.device)]
        value=model(h[ids],action[ids])[torch.arange(len(ids),device=h.device),arms[ids]]
        loss=(value-target[ids]).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite relative-finger loss')
        optimizer.zero_grad();loss.backward()
        gradient=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(gradient):raise ValueError('nonfinite relative-finger gradient')
        optimizer.step();losses.append(float(loss.detach()))
    model.eval();prediction=evaluate_all(model,h,action)
    if not torch.isfinite(prediction).all():raise ValueError('nonfinite relative-finger output')
    factual=prediction[fit_ids,arms[fit_ids]]
    return model,prediction,dict(initial_hash=initial,parameters=sum(p.numel() for p in model.parameters()),
        updates=updates,batch=64,losses=losses,final_source_residual_mse=float((factual-target[fit_ids]).square().mean()))
