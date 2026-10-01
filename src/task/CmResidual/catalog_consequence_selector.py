"""Frozen catalog physics and direct-state-outcome policies, common abstain rules."""
import torch
from .plan_consequence_model import StateOptionConsequenceModel,StateOptionOutcomeModel


class FrozenCatalogSelector:
    def __init__(self,checkpoint,device):
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if p['schema']!='ref2dex.catalog_consequence_controls.v1':raise ValueError('catalog control bundle required')
        self.device=device;self.normalization={k:v.to(device) for k,v in p['normalization'].items()}
        self.rest_mean=p['task_rest_mean'].to(device);self.rest_std=p['task_rest_std'].to(device)
        self.calibration={k:{name:v.to(device) for name,v in values.items()} for k,values in p['calibration'].items()}
        self.margins=p['margins_mm'];self.fixed=p['fit_best_fixed_option'];self.models={}
        for mode,states in p['models'].items():
            self.models[mode]=[]
            for state in states:
                model=(StateOptionOutcomeModel(p['native_observation_dim']) if mode=='state_policy' else StateOptionConsequenceModel(p['native_observation_dim'])).to(device)
                model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);self.models[mode].append(model)

    @torch.no_grad()
    def choose(self,mode,history,native_observation,rest_z):
        norm=self.normalization;h=(history-norm['history_mean'])/norm['history_std'];o=(native_observation-norm['native_mean'])/norm['native_std'];r=(rest_z-self.rest_mean)/self.rest_std
        ood=(h.abs()>8).any((-1,-2))|(o.abs()>8).any(-1)|(r.abs()>8)
        h=h.clamp(-8,8);o=o.clamp(-8,8);r=r.clamp(-8,8)
        prediction=torch.stack([model(h,o,r) if mode=='state_policy' else model(h,o) for model in self.models[mode]])
        point=prediction.mean(0);cal=self.calibration[mode];cols=slice(1,3) if mode=='state_policy' else slice(30,32)
        member_probability=torch.sigmoid(prediction[:,:,:,cols]*cal['slope']+cal['bias']);prob=torch.sigmoid(point[:,:,cols]*cal['slope']+cal['bias'])
        if mode=='state_policy':scores=prediction[:,:,:,0]*10;mean=point[:,:,0]*10
        else:
            z=history[:,-1,38];start=(z-rest_z).clamp_min(0)
            heights=(z[None,:,None,None]+prediction[:,:,:,:10]*.01-rest_z[None,:,None,None])[:,:,:,-3:].amin(-1).clamp_min(0)
            scores=(heights*member_probability[:,:,:,0]-start[None,:,None])*1000
            point_height=(z[:,None,None]+point[:,:,:10]*.01-rest_z[:,None,None])[:,:,-3:].amin(-1).clamp_min(0)
            mean=(point_height*prob[:,:,0]-start[:,None])*1000
        std=(scores-scores[:,:,4:5]).std(0,unbiased=False)
        gain=mean-mean[:,4:5]>self.margins[mode]+std
        risk=member_probability[:,:,:,1].amax(0)<=prob[:,4:5,1]+.02
        contact=prob[:,:,0]>=prob[:,4:5,0]-.05
        valid=gain&risk&contact&~ood[:,None]
        best=mean.masked_fill(~valid,-torch.inf).argmax(-1);choice=torch.where(valid.any(-1),best,torch.full_like(best,4))
        return choice,dict(ood=ood,score_mm=mean,relative_std_mm=std,valid=valid,retention=prob[:,:,0],release=prob[:,:,1],gain= gain,risk=risk,contact=contact)
