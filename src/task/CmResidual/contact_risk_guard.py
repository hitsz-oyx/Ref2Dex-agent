"""Frozen contact consequences constrain a separate trained height scorer."""
import torch
from .structured_contact_actions import FrozenStructuredActionGenerator
from .support_preserving_consequence import SupportPreservingContactConsequence
from .contact_geometry_consequence import node_channels
from .native_pd_selector import native_pd_targets
from .optimized_contact_actions import mixed_command


class ContactRiskGuard(FrozenStructuredActionGenerator):
    JOINT_ALLOWANCE = .02
    CONTACT_LOSS_ALLOWANCE = .02

    def __init__(self, checkpoint, device):
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if bundle['schema'] != 'ref2dex.support_preserving_contact_consequence.v1':
            raise ValueError('audited contact-loss bundle required')
        self.device = device
        self.norm = {k:v.to(device) for k,v in bundle['normalization'].items()}
        self.models = {}
        for mode, states in bundle['models'].items():
            self.models[mode] = []
            for state in states:
                model = SupportPreservingContactConsequence(bundle['physical_dim'], mode).to(device)
                model.load_state_dict(state, strict=True)
                model.eval().requires_grad_(False)
                self.models[mode].append(model)

    def risk(self, mode, weights, bank, anchor, position, offset, scale, context):
        command = mixed_command(bank, weights, anchor, position, offset, scale)
        pd = native_pd_targets(command, position, offset, scale)
        action = torch.cat((node_channels(pd-context['cup_pd']), weights-context['reference_weights']), -1)
        action = ((action-self.norm['node_action_mean'])/self.norm['node_action_std']).clamp(-8, 8)
        law = context['law']
        if mode == 'state_only':
            action, law = torch.zeros_like(action), torch.zeros_like(law)
        values = []
        for model, (hidden, physical, native) in zip(self.models[mode], context['cached']):
            nodes = model.nodes(torch.cat((context['node_context'], action), -1))
            trunk = model.trunk(torch.cat((hidden, physical, native, nodes.mean(-2), law), -1))
            output = model.global_head(trunk)
            events = output[:, :8].softmax(-1)
            joint = events[:, 6:8].sum(-1)
            contact_loss = 1-joint*(1-output[:, 19].sigmoid())
            values.append(torch.stack((joint, contact_loss, output[:, 10].sigmoid(), events[:, 7]), -1))
        return torch.stack(values)

    def optimize(self, mode, bank, anchor, position, offset, scale, inputs):
        if mode not in ('unguarded', 'cm', 'shuffled', 'state_only'):
            raise ValueError('unknown risk control')
        risk_mode = 'cm' if mode == 'unguarded' else mode
        height_context = self.context('direct_score', bank, anchor, position, offset, scale, inputs)
        risk_context = self.context(risk_mode, bank, anchor, position, offset, scale, inputs)
        reference = height_context['reference_weights']
        with torch.no_grad():
            ref_score = self.predict('direct_score', reference, bank, anchor, position, offset, scale, height_context)[0]
            ref_risk = self.risk(risk_mode, reference, bank, anchor, position, offset, scale, risk_context)
        best = reference.clone()
        best_value = bank.new_full((len(bank),), -1.645*.001)
        trace = []
        with torch.enable_grad():
            logits = (.95*reference+.05/6).log().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([logits], lr=self.LR)
            for step in range(self.STEPS+1):
                weights = logits.softmax(-1)
                scores, _, _, _, _, candidate_ood = self.predict('direct_score', weights, bank, anchor,
                    position, offset, scale, height_context)
                risks = self.risk(risk_mode, weights, bank, anchor, position, offset, scale, risk_context)
                gain = scores-ref_score
                mean = gain.mean(0)
                std = ((gain-mean[None]).square().mean(0)+1e-6).sqrt()
                difference = (risks-ref_risk).mean(0)
                confidence = mean-1.645*std
                value = confidence
                feasible = (confidence>0)&~height_context['ood']&~candidate_ood
                if mode != 'unguarded':
                    violations = torch.stack(((-difference[:,0]-self.JOINT_ALLOWANCE).clamp_min(0),
                        (difference[:,1]-self.CONTACT_LOSS_ALLOWANCE).clamp_min(0),
                        (difference[:,2]-self.LOSS_ALLOWANCE).clamp_min(0),
                        (-difference[:,3]-self.SUPPORT_ALLOWANCE).clamp_min(0)), -1)
                    value = value-self.PENALTY_MM*violations.sum(-1)
                    feasible &= (violations==0).all(-1)&~risk_context['ood']
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
            scores, _, _, command, pd, ood = self.predict('direct_score', best, bank, anchor,
                position, offset, scale, height_context)
            risks = self.risk(risk_mode, best, bank, anchor, position, offset, scale, risk_context)
        return best, dict(trace=torch.stack(trace), predicted_score_mm=scores, predicted_risk=risks,
            reference_score_mm=ref_score, reference_risk=ref_risk, initial_command=command,
            initial_pd_targets=pd, context_ood=height_context['ood']|risk_context['ood'], candidate_ood=ood,
            changed_reference=(best-reference).abs().amax((-1,-2))>1e-5,
            risk_columns=['terminal_joint_probability','any_contact_loss_probability','geometric_loss_probability','support_probability'],
            mode=mode, height_source='direct_score', risk_source=risk_mode, optimize_steps=self.STEPS)
