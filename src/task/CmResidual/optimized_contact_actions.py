"""Generate bounded feedback-law coefficients using frozen Cm consequences."""
import math
import torch
from .contact_geometry_actions import BLOCKS
from .contact_geometry_consequence import GeometryConsequenceModel,node_inputs,relative_geometry
from .native_pd_selector import physical_inputs,native_pd_targets
from .orientation_anchored_options import orientation_anchored_action


def live_inputs(history,native,hand_force,object_force,mass,gravity,clearance,rest):
    """No future tensors are accepted by this interface."""
    state=history[:,-1,:49];n=len(state)
    physical,_=physical_inputs(history,hand_force,object_force,mass,gravity,clearance,rest)
    nodes,_=node_inputs(state,native,state[:,:18],state.new_zeros(n,6,6))
    pos,vel=relative_geometry(native);q=state[:,39:43];x,y,z,w=q.unbind(-1)
    R=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                   2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                   2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(n,3,3)
    omega=torch.einsum('nij,nj->ni',R.transpose(-1,-2),state[:,46:49])
    dt=1/30;bits=history[:,-1,49:51];logits=(bits*2-1)*math.log(9)
    local=torch.cat(((vel-torch.cross(omega[:,None].expand_as(pos),pos,dim=-1))*dt/.005,
                      torch.zeros_like(vel)),-1).flatten(-2)
    step=torch.cat((torch.zeros_like(state[:,43:46]),state[:,43:46]*dt/.005,
                    state[:,45,None]*dt/.002,logits),-1)
    dz=torch.minimum(state[:,45]*dt*8,state[:,45]*dt*10)
    height=(state[:,38]-rest).clamp_min(0)
    support=bits.bool().all(-1)&(clearance+dz>=.002)
    cv_height=(height+dz).clamp_min(0)*support
    loss=(clearance>=.002)&(clearance+torch.minimum(state[:,45]*dt,state[:,45]*dt*10)<.002)
    macro=torch.cat((dz[:,None]/.01,dz[:,None]/.002,logits,
                     ((loss.float()*2-1)*math.log(9))[:,None],
                     (((support&(cv_height>=.03)).float()*2-1)*math.log(9))[:,None],cv_height[:,None]/.01),-1)
    return dict(history=history,physical=physical,node_state=nodes,prior=torch.cat((local,step,macro),-1))


def mixed_command(bank,weights,anchor,position,offset,scale):
    command=bank[:,1].clone()
    for block,(start,stop) in enumerate(BLOCKS):
        command[:,start:stop]=torch.einsum('ne,ned->nd',weights[:,block],bank[:,:,start:stop])
    return orientation_anchored_action(command,anchor,position,offset,scale)


class FrozenConsequenceActionGenerator:
    STEPS=32
    LR=.15
    RISK_PENALTY_MM=100.

    def __init__(self,checkpoint,device):
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if p['schema']!='ref2dex.contact_geometry_consequence.v1':
            raise ValueError('audited geometry consequence bundle required')
        self.device=device;self.norm={k:v.to(device) for k,v in p['normalization'].items()};self.models={}
        for mode in ('cm','direct_score','shuffled'):
            self.models[mode]=[]
            for state in p['models'][mode]:
                model=GeometryConsequenceModel(p['physical_dim'],mode).to(device)
                model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
                self.models[mode].append(model)

    @torch.no_grad()
    def context(self,mode,inputs):
        norm=self.norm
        h=((inputs['history']-norm['history_mean'])/norm['history_std']).clamp(-8,8)
        physical=((inputs['physical']-norm['physical_mean'])/norm['physical_std']).clamp(-8,8)
        node=((inputs['node_state']-norm['node_state_mean'])/norm['node_state_std']).clamp(-8,8)
        law=((torch.ones(len(h),1,device=h.device)-norm['law_mean'])/norm['law_std']).clamp(-8,8)
        contexts=[]
        for model in self.models[mode]:
            _,hidden=model.history(h)
            contexts.append((hidden[-1].detach(),model.physical(physical).detach()))
        return dict(node=node,law=law,state=inputs['history'][:,-1,:49],prior=inputs['prior'],
                    contexts=contexts)

    def predict(self,mode,weights,bank,anchor,position,offset,scale,context):
        action=mixed_command(bank,weights,anchor,position,offset,scale)
        targets=native_pd_targets(action,position,offset,scale)
        # Only the goal/weights vary during optimization; state and history stay fixed.
        node_goal=context['state'].new_zeros(len(bank),6,6)
        channels=((0,1,2,3,4,5),(6,7),(8,9),(10,11),(12,13),(14,15,16,17))
        difference=targets-position
        for node,indices in enumerate(channels):
            node_goal[:,node,:len(indices)]=difference[:,list(indices)]
        na=torch.cat((node_goal,weights),-1)
        na=((na-self.norm['node_action_mean'])/self.norm['node_action_std']).clamp(-8,8)
        scores=[];risks=[]
        for model,(hidden,physical) in zip(self.models[mode],context['contexts']):
            nodes=model.nodes(torch.cat((context['node'],na),-1))
            value=model.trunk(torch.cat((hidden,physical,nodes.mean(-2),context['law']),-1))
            output=model.global_head(value)
            scores.append((context['prior'][:,51]+output[:,15])*10)
            risks.append(torch.sigmoid(context['prior'][:,49]+(0 if mode=='direct_score' else output[:,13])))
        return torch.stack(scores),torch.stack(risks),action,targets

    def optimize(self,mode,bank,anchor,position,offset,scale,inputs):
        context=self.context(mode,inputs);n=len(bank)
        reference=bank.new_zeros(n,6,6);reference[:,:,1]=1
        with torch.no_grad():
            ref_score,ref_risk,_,_=self.predict(mode,reference,bank,anchor,position,offset,scale,context)
        init=.95*reference+.05/6
        with torch.enable_grad():
            logits=init.log().detach().requires_grad_(True)
            optimizer=torch.optim.Adam([logits],lr=self.LR)
            best=reference.clone();best_value=bank.new_full((n,),-1.645*.001)
            trace=[]
            for iteration in range(self.STEPS+1):
                weights=logits.softmax(-1)
                scores,risks,_,_=self.predict(mode,weights,bank,anchor,position,offset,scale,context)
                difference=scores-ref_score
                mean=difference.mean(0)
                std=((difference-mean[None]).square().mean(0)+1e-6).sqrt()
                risk_difference=risks.mean(0)-ref_risk.mean(0)
                objective=mean-1.645*std-self.RISK_PENALTY_MM*(risk_difference-.05).clamp_min(0)
                if not torch.isfinite(objective).all():raise ValueError('nonfinite planning objective')
                improved=objective.detach()>best_value
                best[improved]=weights.detach()[improved];best_value[improved]=objective.detach()[improved]
                trace.append(torch.stack((mean.detach(),std.detach(),risk_difference.detach(),objective.detach()),-1))
                if iteration==self.STEPS:break
                optimizer.zero_grad(set_to_none=True)
                (-objective.mean()).backward()
                if logits.grad is None or not torch.isfinite(logits.grad).all():raise ValueError('planning gradient unavailable')
                torch.nn.utils.clip_grad_norm_([logits],1.);optimizer.step()
        with torch.no_grad():
            scores,risks,action,targets=self.predict(mode,best,bank,anchor,position,offset,scale,context)
        return best,dict(trace=torch.stack(trace),predicted_score_mm=scores,predicted_loss=risks,
                         initial_command=action,initial_pd_targets=targets,
                         changed_reference=(best-reference).abs().amax((-1,-2))>1e-5,
                         optimize_steps=self.STEPS,model_mode=mode)
