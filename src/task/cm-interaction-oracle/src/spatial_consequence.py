"""Nominal spatial OI-CmV2 adapter; compact consequence/prognosis heads.

The hand sweep is an interpolation to a decision-time PD endpoint, not an
observed or predicted execution trajectory. Original V13 modules are reused.
"""
from types import SimpleNamespace
import torch
from torch import nn
from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model

WIDTH = 32
UPDATES = 300
BATCH = 64
SEED = 245


class SpatialConsequence(nn.Module):
    def __init__(self, h_width, heads):
        super().__init__()
        self.spatial = ObjectInteractionCmv2V13Model(SimpleNamespace(
            hidden_width=WIDTH, num_tokens=4, use_residual=False,
            interaction_mode='swept', feature_scale_m=.02, frame_dt_s=8/30,
            knn_k=8, interaction_radius_m=.02))
        self.head = nn.Sequential(nn.Linear(h_width+WIDTH+32+26,64),nn.Tanh(),
                                  nn.Linear(64,32),nn.Tanh(),nn.Linear(32,heads))

    def forward(self, h, action, physical, geometry, arms, ids):
        batch = dict(obj_points=geometry['obj_points'].expand(len(ids),-1,-1),
            obj_normals=geometry['obj_normals'].expand(len(ids),-1,-1),
            hand_points=geometry['points'][ids], hand_normals=geometry['normals'][ids],
            hand_flow=geometry['nominal_flow'][ids,arms[ids]],
            delta_time_s=torch.full((len(ids),),8/30,device=h.device))
        fused = self.spatial(batch)['fused_feature']
        return self.head(torch.cat((h[ids],fused,action[ids],physical[ids]),-1))


def evaluate(model, h, action, physical, geometry, arms, ids):
    result=[]
    with torch.no_grad():
        for chunk in ids.split(BATCH):
            result.append(model(h,action,physical,geometry,arms,chunk))
    return torch.cat(result)


def fit(h, action, physical, geometry, arms, target, fit_ids, updates=UPDATES):
    # Same init and exact same sampled rows across matched variants.
    torch.manual_seed(SEED)
    model=SpatialConsequence(h.shape[1],target.shape[1]).to(h.device)
    initial=torch.cat([v.detach().flatten() for v in model.parameters()]).cpu()
    from hashlib import sha256
    initial_hash=sha256(initial.numpy().tobytes()).hexdigest()
    generator=torch.Generator(device=h.device).manual_seed(SEED+1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]
    for step in range(updates):
        ids=fit_ids[torch.randint(len(fit_ids),(BATCH,),generator=generator,device=h.device)]
        loss=(model(h,action,physical,geometry,arms,ids)-target[ids]).square().mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
        optimizer.zero_grad();loss.backward()
        gradient=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(gradient): raise ValueError('nonfinite gradient')
        optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    prediction=evaluate(model,h,action,physical,geometry,arms,torch.arange(len(h),device=h.device))
    if not torch.isfinite(prediction).all(): raise ValueError('nonfinite predictions')
    return model,prediction,dict(initial_hash=initial_hash,updates=updates,batch=BATCH,
        parameters=sum(p.numel() for p in model.parameters()),losses=losses,
        final_train_mse=float((prediction[fit_ids]-target[fit_ids]).square().mean()))


def action_slots(variant, geometry, intended_arms, onehot):
    rows=torch.arange(len(intended_arms),device=intended_arms.device)
    blank=torch.zeros(len(rows),32,device=rows.device)
    if variant=='Arm':
        return torch.nn.functional.pad(onehot,(0,18)),torch.zeros_like(intended_arms)
    if variant=='Joint':
        # Physical radian perturbations relative to source PD target, fixed
        # .32 rad convention, no test-fitted whitening.
        return torch.nn.functional.pad(geometry['joint'][rows,intended_arms]/.32,(0,14)),torch.zeros_like(intended_arms)
    return blank,intended_arms if variant in ('Flow','Shuffled') else torch.zeros_like(intended_arms)
