"""Frozen calibrated physical program selector; no critic or external prior."""
import torch
from .plan_consequence_model import PlanConsequenceModel,StateOptionConsequenceModel,plan_features


class FrozenPlanSelector:
    def __init__(self,checkpoint,device):
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if p['schema']!='ref2dex.plan_consequence.v1':raise ValueError('actual H10 model required')
        self.device=device;self.normalization={k:v.to(device) for k,v in p['normalization'].items()}
        self.models={};self.calibration={k:{name:value.to(device) for name,value in v.items()} for k,v in p['calibration'].items()}
        self.margins=p['margins_mm'];self.fixed=p['fit_best_fixed_option']
        for mode,states in p['models'].items():
            self.models[mode]=[]
            for state in states:
                model=(StateOptionConsequenceModel(p['native_observation_dim']) if mode=='state_heads' else PlanConsequenceModel(p['native_observation_dim'],mode)).to(device)
                model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);self.models[mode].append(model)

    @torch.no_grad()
    def choose(self,mode,history,native_observation,candidate_actions,rotation_anchor,rest_z):
        n=len(history);normal=self.normalization
        h=(history-normal['history_mean'])/normal['history_std'];o=(native_observation-normal['native_mean'])/normal['native_std']
        ood=(h.abs()>8).any((-1,-2))|(o.abs()>8).any(-1)
        h=h.clamp(-8,8);o=o.clamp(-8,8);program=plan_features(candidate_actions,rotation_anchor/torch.pi)
        values=[]
        for model in self.models[mode]:
            if mode=='state_heads':value=model(h,o)
            else:value=model(h[:,None].expand(-1,8,-1,-1).reshape(n*8,10,69),o[:,None].expand(-1,8,-1).reshape(n*8,-1),program.reshape(n*8,23),program[:,4,None].expand(-1,8,-1).reshape(n*8,23)).reshape(n,8,32)
            values.append(value)
        prediction=torch.stack(values);c=self.calibration[mode]
        probability=torch.sigmoid(prediction[:,:,:,30:32]*c['slope']+c['bias'])
        z=history[:,-1,38];height=(z[None,:,None,None]+prediction[:,:,:,:10]*.01-rest_z[None,:,None,None])[:,:,:,-3:].amin(-1).clamp_min(0)
        scores=(height*probability[:,:,:,0]-(z-rest_z).clamp_min(0)[None,:,None])*1000
        mean=scores.mean(0);difference=scores-scores[:,:,4:5];uncertainty=difference.std(0,unbiased=False)
        retention=probability[:,:,:,0].mean(0);release=probability[:,:,:,1]
        valid=(mean-mean[:,4:5]>self.margins[mode]+uncertainty)&(release.max(0).values<=release.mean(0)[:,4:5]+.02)&(retention>=retention[:,4:5]-.05)&~ood[:,None]
        score=mean.masked_fill(~valid,-torch.inf);best=score.argmax(-1);chosen=torch.where(valid.any(-1),best,torch.full_like(best,4))
        return chosen,dict(ood=ood,score_mm=mean,relative_std_mm=uncertainty,valid=valid,retention=retention,release_mean=release.mean(0))
