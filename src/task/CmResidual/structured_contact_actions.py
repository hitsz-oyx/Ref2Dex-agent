"""Experimental frozen joint-consequence generation of executable coefficients."""
import torch
from .structured_contact_consequence import StructuredContactConsequence,current_features
from .contact_geometry_consequence import node_channels
from .native_pd_selector import native_pd_targets
from .optimized_contact_actions import mixed_command


def live_inputs(history,native,hand_force,object_force,mass,gravity,clearance,rest):
    return dict(history=history,native=native,hand_force=hand_force,object_force=object_force,
                mass=mass,gravity=gravity,clearance=clearance,rest=rest)


class FrozenStructuredActionGenerator:
    STEPS=32
    LR=.15
    PENALTY_MM=100.
    LOSS_ALLOWANCE=.02
    SUPPORT_ALLOWANCE=.05

    def __init__(self,checkpoint,device):
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if p['schema']!='ref2dex.structured_contact_consequence.v1':raise ValueError('structured bundle required')
        self.device=device;self.norm={k:v.to(device) for k,v in p['normalization'].items()};self.models={}
        for mode,states in p['models'].items():
            self.models[mode]=[]
            for weights in states:
                model=StructuredContactConsequence(p['physical_dim'],mode).to(device)
                model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False)
                self.models[mode].append(model)

    @torch.no_grad()
    def context(self,mode,bank,anchor,position,offset,scale,inputs):
        ref=bank.new_zeros(len(bank),6,6);ref[:,:,1]=1
        cup=mixed_command(bank,ref,anchor,position,offset,scale)
        data=current_features(**inputs,bank=bank,cup_action=cup,candidate_action=cup,weights=ref,
            law=bank.new_ones(len(bank)),offset=offset,scale=scale)
        unbounded={k:(v-self.norm[k+'_mean'])/self.norm[k+'_std'] for k,v in data.items()}
        normalized={k:v.clamp(-8,8) for k,v in unbounded.items()}
        ood=torch.zeros(len(bank),device=bank.device,dtype=torch.bool)
        for key in ('history','physical','native','node_context'):
            ood|=(unbounded[key].abs()>8).flatten(1).any(-1)
        cached=[]
        for model in self.models[mode]:
            _,h=model.history(normalized['history'])
            cached.append((h[-1],model.physical(normalized['physical']),model.native(normalized['native'])))
        static_support=[];static_loss=[]
        if mode=='direct_score':
            # Height-only control receives state-only risk priors, constant over programmes.
            for model in self.models['state_only']:
                value=model(**normalized);static_support.append(value['support_probability']);static_loss.append(value['loss_probability'])
        return dict(cached=cached,node_context=normalized['node_context'],law=normalized['law'],
            reference_weights=ref,cup_pd=native_pd_targets(cup,position,offset,scale),ood=ood,
            static_support=torch.stack(static_support) if static_support else None,
            static_loss=torch.stack(static_loss) if static_loss else None)

    def predict(self,mode,weights,bank,anchor,position,offset,scale,context):
        command=mixed_command(bank,weights,anchor,position,offset,scale)
        pd=native_pd_targets(command,position,offset,scale)
        action=torch.cat((node_channels(pd-context['cup_pd']),weights-context['reference_weights']),-1)
        unbounded=(action-self.norm['node_action_mean'])/self.norm['node_action_std']
        ood=(unbounded.abs()>8).flatten(1).any(-1)
        action=unbounded.clamp(-8,8);scores=[];loss=[];support=[]
        for model,(hidden,physical,native) in zip(self.models[mode],context['cached']):
            nodes=model.nodes(torch.cat((context['node_context'],action),-1))
            trunk=model.trunk(torch.cat((hidden,physical,native,nodes.mean(-2),context['law']),-1))
            output=model.global_head(trunk);joint=output[:,:8].softmax(-1)[:,7]
            score=joint*torch.nn.functional.softplus(output[:,8])*10
            if mode=='direct_score':score=torch.nn.functional.softplus(output[:,11])*10
            scores.append(score);loss.append(output[:,10].sigmoid());support.append(joint)
        if mode=='direct_score':
            loss=list(context['static_loss']);support=list(context['static_support'])
        return torch.stack(scores),torch.stack(loss),torch.stack(support),command,pd,ood

    def optimize(self,mode,bank,anchor,position,offset,scale,inputs):
        if mode not in ('cm','shuffled','direct_score'):raise ValueError('action-conditioned generation mode required')
        context=self.context(mode,bank,anchor,position,offset,scale,inputs)
        reference=context['reference_weights'];n=len(bank)
        with torch.no_grad():
            ref_score,ref_loss,ref_support,_,_,_=self.predict(mode,reference,bank,anchor,position,offset,scale,context)
        with torch.enable_grad():
            logits=(.95*reference+.05/6).log().detach().requires_grad_(True)
            optimizer=torch.optim.Adam([logits],lr=self.LR)
            best=reference.clone();best_value=bank.new_full((n,),-1.645*.001);trace=[]
            for iteration in range(self.STEPS+1):
                weights=logits.softmax(-1)
                score,loss,support,_,_,candidate_ood=self.predict(mode,weights,bank,anchor,position,offset,scale,context)
                gain=score-ref_score;mean=gain.mean(0);std=((gain-mean[None]).square().mean(0)+1e-6).sqrt()
                loss_difference=(loss-ref_loss).mean(0);support_difference=(support-ref_support).mean(0)
                confidence=mean-1.645*std
                value=confidence-self.PENALTY_MM*(loss_difference-self.LOSS_ALLOWANCE).clamp_min(0)
                value=value-self.PENALTY_MM*(-support_difference-self.SUPPORT_ALLOWANCE).clamp_min(0)
                feasible=(confidence>0)&(loss_difference<=self.LOSS_ALLOWANCE)&(support_difference>=-self.SUPPORT_ALLOWANCE)&~context['ood']&~candidate_ood
                if not torch.isfinite(value).all():raise ValueError('nonfinite structured planning')
                improved=feasible&(value.detach()>best_value)
                best[improved]=weights.detach()[improved];best_value[improved]=value.detach()[improved]
                trace.append(torch.stack((mean.detach(),std.detach(),loss_difference.detach(),support_difference.detach(),value.detach(),feasible.float()),-1))
                if iteration==self.STEPS:break
                optimizer.zero_grad(set_to_none=True);(-value.mean()).backward()
                if logits.grad is None or not torch.isfinite(logits.grad).all():raise ValueError('finite action gradients required')
                torch.nn.utils.clip_grad_norm_([logits],1.);optimizer.step()
        with torch.no_grad():
            scores,loss,support,command,pd,ood=self.predict(mode,best,bank,anchor,position,offset,scale,context)
        return best,dict(trace=torch.stack(trace),predicted_score_mm=scores,predicted_loss=loss,
            predicted_support=support,initial_command=command,initial_pd_targets=pd,
            changed_reference=(best-reference).abs().amax((-1,-2))>1e-5,context_ood=context['ood'],candidate_ood=ood,
            reference_score_mm=ref_score,reference_loss=ref_loss,reference_support=ref_support,
            optimize_steps=self.STEPS,model_mode=mode,
            direct_score_risk_source='state_only_static' if mode=='direct_score' else 'action_conditioned_joint_model')
