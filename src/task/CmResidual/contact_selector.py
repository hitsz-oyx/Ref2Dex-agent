"""Frozen physical-consequence selector used for actual-state intervention data."""
from pathlib import Path
import torch

from .contact_ranker import physical_history,ConsequenceNetwork,all_predictions,policy_choice


class FrozenContactSelector:
    def __init__(self,checkpoint,device):
        self.saved=torch.load(Path(checkpoint),map_location=device,weights_only=False)
        if self.saved['schema']!='ref2dex.contact_ranker.v1':raise ValueError('ranker schema')
        self.device=device;self.models=[]
        # Construction must not perturb the actor/PointNet sampling RNG.
        with torch.random.fork_rng(devices=[]):
            for state in self.saved['models']['cm']:
                model=ConsequenceNetwork().to(device);model.load_state_dict(state)
                model.eval().requires_grad_(False);self.models.append(model)

    @torch.no_grad()
    def predict(self,history,candidate,rest,motion,start,trigger):
        s=self.saved
        h=(physical_history(history,rest)-s['history_mean'])/s['history_scale']
        actions=(candidate-s['action_mean'])/s['action_scale']
        context=torch.cat((candidate[:,4],torch.nn.functional.one_hot(motion.long(),3).float(),
                           ((start+trigger).float()/600)[:,None]),-1)
        context=(context-s['context_mean'])/s['context_scale']
        predictions=[]
        for model in self.models:
            raw=all_predictions(model,h,actions,context)
            predictions.append(torch.stack(((raw[:,:,0]*s['lift_scale']+s['lift_mean']).clamp_min(0),
                                             raw[:,:,1].sigmoid(),raw[:,:,2].sigmoid()),-1))
        eligible=history[:,-1,38]-rest>=.03
        choice,mean,lower=policy_choice(torch.stack(predictions),s['calibration']['cm']['margin_mm'],
                                       drop_supported=s['drop_supported'],eligible=eligible)
        rows=torch.arange(len(choice),device=choice.device)
        return dict(proposed_arm=choice,predicted_gain_mm=mean[rows,choice,0]-mean[:,4,0],
                    lower_gain_mm=lower[rows,choice],predicted_contact=mean[rows,choice,1],
                    predicted_drop=mean[rows,choice,2])
