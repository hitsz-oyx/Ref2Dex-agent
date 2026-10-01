"""Frozen one-step motor-conditioned scorer with a matched direct-state policy."""
import math
import torch
from .native_pd_consequence import NativePDConsequenceModel
from .native_pd_policy_controls import NativeStateOutcomePolicy
from .executable_contact_options import INDEPENDENT,COUPLINGS


def native_pd_targets(raw,position,offset,scale):
    value=raw.clone();value[...,6:]=(1+value[...,6:])/2
    target=offset+scale*value;target[...,:6]+=position[...,:6]
    for dst,src,ratio in COUPLINGS:target[...,dst]=target[...,src]*ratio
    return target


def physical_inputs(history,hand_force,object_force,mass,gravity,clearance,rest):
    state=history[:,-1,:49];raw=torch.cat((hand_force.flatten(-2),object_force),-1)/(mass[:,None]*gravity);raw=raw.sign()*raw.abs().log1p()
    physical=torch.cat((raw,clearance[:,None],(state[:,38]-rest).clamp_min(0)[:,None]),-1)
    velocity=state[:,45]/30;pair=history[:,-1,49:51].bool().all(-1)
    bits=torch.stack((pair,pair&(clearance>=.002),clearance<.002),-1)
    prior=torch.cat((velocity[:,None].expand(-1,2)/.002,(bits.float()*2-1)*math.log(99),velocity[:,None]/.01),-1)
    return physical,prior


class FrozenNativePDSelector:
    def __init__(self,checkpoint,device):
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if p['schema']!='ref2dex.native_pd_controls.v1':raise ValueError('native motor-conditioned bundle required')
        self.device=device;self.fixed=p['fixed'];self.norm={k:v.to(device) for k,v in p['normalization'].items()};self.native_mean=p['native_mean'].to(device);self.native_std=p['native_std'].to(device);self.margins=p['margins_mm'];self.cal={k:{name:v.to(device) for name,v in d.items()} for k,d in p['calibration'].items()};self.models={}
        for mode,states in {**p['models'],'state_policy':p['state_policy_models']}.items():
            self.models[mode]=[]
            for state in states:
                model=(NativeStateOutcomePolicy(p['physical_dim'],p['native_dim']) if mode=='state_policy' else NativePDConsequenceModel(p['physical_dim'],mode)).to(device)
                model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);self.models[mode].append(model)

    @torch.no_grad()
    def choose(self,mode,history,native,rest,candidate_pd,hand_force,object_force,mass,gravity,clearance):
        physical,prior=physical_inputs(history,hand_force,object_force,mass,gravity,clearance,rest)
        norm=self.norm;h=(history-norm['history_mean'])/norm['history_std'];f=(physical-norm['physical_mean'])/norm['physical_std'];o=(native-self.native_mean)/self.native_std
        goal=(candidate_pd-history[:,-1,None,:18])[:,:,list(INDEPENDENT)];g=(goal-norm['goal_mean'])/norm['goal_std']
        ood=(h.abs()>8).any((-1,-2))|(f.abs()>8).any(-1)|(o.abs()>8).any(-1);candidate_ood=(g.abs()>8).any(-1)
        h=h.clamp(-8,8);f=f.clamp(-8,8);o=o.clamp(-8,8);g=g.clamp(-8,8);pred=[]
        for model in self.models[mode]:
            if mode=='state_policy':value=model(h,f,o,prior)
            else:
                _,hidden=model.history(h);state=torch.cat((hidden[-1],model.physical(f)),-1)
                value=(model.head(torch.cat((state[:,None].expand(-1,8,-1),g),-1))+prior[:,None])[:,:,[5,3,4]]
            pred.append(value)
        pred=torch.stack(pred);point=pred.mean(0);cal=self.cal[mode];member_probability=torch.sigmoid(pred[:,:,:,1:]*cal['slope']+cal['bias']);prob=torch.sigmoid(point[:,:,1:]*cal['slope']+cal['bias'])
        score=point[:,:,0]*10;member_score=pred[:,:,:,0]*10;std=(member_score-member_score[:,:,4:5]).std(0,unbiased=False)
        gain=score-score[:,4:5]>self.margins[mode]+std;risk=member_probability[:,:,:,1].amax(0)<=prob[:,4:5,1]+.02;contact=prob[:,:,0]>=prob[:,4:5,0]-.05
        valid=gain&risk&contact&~ood[:,None]&~candidate_ood;best=score.masked_fill(~valid,-torch.inf).argmax(-1);choice=torch.where(valid.any(-1),best,torch.full_like(best,4))
        return choice,dict(ood=ood,candidate_ood=candidate_ood,score_mm=score,relative_std_mm=std,retention=prob[:,:,0],release=prob[:,:,1],gain=gain,risk=risk,contact=contact,valid=valid)
