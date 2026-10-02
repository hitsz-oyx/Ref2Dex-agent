"""Equivalent unguarded decisions with batched diagnostic risk evaluation.

Risk does not enter the unguarded objective. Only its diagnostic trajectory is
batched, after exactly the original height optimizer has chosen its weights.
Guarded modes use the original implementation unchanged.
"""
import torch
from .contact_risk_guard import ContactRiskGuard


class EfficientContactRiskGuard(ContactRiskGuard):
    def optimize(self, mode, bank, anchor, position, offset, scale, inputs):
        if mode != 'unguarded':
            return super().optimize(mode,bank,anchor,position,offset,scale,inputs)
        risk_mode = 'cm' if mode == 'unguarded' else mode
        height_context = self.context('direct_score', bank, anchor, position, offset, scale, inputs)
        risk_context = self.context(risk_mode, bank, anchor, position, offset, scale, inputs)
        reference = height_context['reference_weights']
        with torch.no_grad():
            ref_score = self.predict('direct_score', reference, bank, anchor, position, offset, scale, height_context)[0]
            ref_risk = self.risk(risk_mode, reference, bank, anchor, position, offset, scale, risk_context)
        best = reference.clone()
        best_value = bank.new_full((len(bank),), -1.645*.001)
        trace, trajectory = [], []
        with torch.enable_grad():
            logits = (.95*reference+.05/6).log().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([logits], lr=self.LR)
            for step in range(self.STEPS+1):
                weights = logits.softmax(-1)
                scores, _, _, _, _, candidate_ood = self.predict('direct_score', weights, bank, anchor,
                    position, offset, scale, height_context)
                trajectory.append(weights.detach().clone())
                gain = scores-ref_score
                mean = gain.mean(0)
                std = ((gain-mean[None]).square().mean(0)+1e-6).sqrt()
                difference = bank.new_zeros(len(bank),4)
                confidence = mean-1.645*std
                value = confidence
                feasible = (confidence>0)&~height_context['ood']&~candidate_ood
                if not torch.isfinite(value).all():
                    raise ValueError('finite constrained objective required')
                improve = feasible&(value.detach()>best_value)
                best[improve] = weights.detach()[improve]
                best_value[improve] = value.detach()[improve]
                trace.append(torch.cat((mean.detach()[:,None],std.detach()[:,None],difference.detach(),
                    value.detach()[:,None],feasible.float()[:,None]), -1))
                if step == self.STEPS:
                    break
                optimizer.zero_grad(set_to_none=True)
                (-value.mean()).backward()
                if logits.grad is None or not torch.isfinite(logits.grad).all():
                    raise ValueError('finite action gradient required')
                torch.nn.utils.clip_grad_norm_([logits], 1.)
                optimizer.step()
        with torch.no_grad():
            length=len(trajectory);n=len(bank)
            tiled={k:risk_context[k].repeat(length,*([1]*(risk_context[k].ndim-1)))
                for k in ('node_context','law','reference_weights','cup_pd')}
            tiled['cached']=[tuple(v.repeat(length,1) for v in values) for values in risk_context['cached']]
            all_risks=self.risk(risk_mode,torch.stack(trajectory).reshape(length*n,6,6),
                bank.repeat(length,1,1),anchor.repeat(length,1),position.repeat(length,1),offset,scale,tiled)
            all_risks=all_risks.reshape(len(self.models[risk_mode]),length,n,4)
            risk_difference=(all_risks-ref_risk[:,None]).mean(0)
            trace=torch.stack(trace)
            trace[:,:,2:6]=risk_difference
        with torch.no_grad():
            scores, _, _, command, pd, ood = self.predict('direct_score', best, bank, anchor,
                position, offset, scale, height_context)
            risks = self.risk(risk_mode, best, bank, anchor, position, offset, scale, risk_context)
        return best, dict(trace=trace, predicted_score_mm=scores, predicted_risk=risks,
            reference_score_mm=ref_score, reference_risk=ref_risk, initial_command=command,
            initial_pd_targets=pd, context_ood=height_context['ood']|risk_context['ood'], candidate_ood=ood,
            changed_reference=(best-reference).abs().amax((-1,-2))>1e-5,
            risk_columns=['terminal_joint_probability','any_contact_loss_probability','geometric_loss_probability','support_probability'],
            mode=mode, height_source='direct_score', risk_source=risk_mode, optimize_steps=self.STEPS)
