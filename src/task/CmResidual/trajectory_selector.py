"""Frozen physical trajectory policies for prospective common-action controls."""
import torch
from .contact_ranker import physical_history
from .contact_trajectory import TrajectoryNetwork,all_trajectories,decode_trajectories,retained_choice

# Dexplore_Inspire overwrites6coupled finger targets. A constant2step plan
# depends on these12 independent normalized commands, including6wrist DOFs.
EFFECTIVE_CHANNELS=(0,1,2,3,4,5,6,8,10,12,14,15)


def effective_commands(candidate):
    return candidate[...,list(EFFECTIVE_CHANNELS)]


def recommendation_probability(candidate,policy_proposals,assignment):
    command=effective_commands(candidate)
    proposed=command.gather(1,policy_proposals[:,:,None].expand(-1,-1,len(EFFECTIVE_CHANNELS)))
    actual=command[torch.arange(len(command),device=command.device),assignment]
    return (proposed==actual[:,None]).all(-1).float().mean(-1)


class FrozenTrajectorySelectors:
    policy_names=('cm','state_only','action_shuffled','always_base','best_fixed')

    def __init__(self,checkpoint,device):
        self.saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if self.saved['schema']!='ref2dex.contact_trajectory.v1' or not self.saved['setup_gate']['passed']:
            raise ValueError('supported frozen probability setup required')
        if self.saved['score_key']!='supported_change_mm':raise ValueError('signed support objective required')
        self.models=[];self.pool={}
        with torch.random.fork_rng(devices=[]):
            for variant,states in self.saved['models'].items():
                self.pool[variant]=[]
                for state in states:
                    model=TrajectoryNetwork(state_only=variant=='state_only').to(device);model.load_state_dict(state)
                    model.eval().requires_grad_(False);self.models.append(model);self.pool[variant].append(model)

    @torch.no_grad()
    def predict(self,history,candidate,rest,motion,start,trigger):
        s=self.saved
        h=(physical_history(history,rest)-s['history_mean'])/s['history_scale']
        a=(candidate-s['action_mean'])/s['action_scale']
        context=torch.cat((candidate[:,4],torch.nn.functional.one_hot(motion.long(),3).float(),
                           ((start+trigger).float()/600)[:,None]),-1)
        context=(context-s['context_mean'])/s['context_scale']
        if not all(torch.isfinite(v).all() for v in (h,a,context)):raise ValueError('nonfinite pre-action input')
        initial=(history[:,-1,38]-rest)*1000
        choices={};predictions={};lower={}
        for variant in ['cm','state_only','action_shuffled']:
            raw=torch.stack([all_trajectories(m,h,a,context) for m in self.pool[variant]])
            if not torch.isfinite(raw).all():raise ValueError('nonfinite physical trajectory prediction')
            pred=decode_trajectories(raw,s['height_mean'],s['height_scale'],initial[None,:,None],s['probability_calibration'][variant])
            choice,low=retained_choice(pred,s['calibration'][variant]['margin_mm'],s['release_supported'],score_key=s['score_key'])
            choices[variant]=choice;predictions[variant]={k:v.mean(0) for k,v in pred.items()};lower[variant]=low
        choice=choices['cm'];rows=torch.arange(len(choice),device=choice.device);cm=predictions['cm']
        pool=torch.stack([choices['cm'],choices['state_only'],choices['action_shuffled'],
                          torch.full_like(choice,4),torch.full_like(choice,s['best_fixed'])],-1)
        return dict(proposed_arm=choice,policy_proposals=pool,
                    predicted_gain_mm=cm['supported_change_mm'][rows,choice]-cm['supported_change_mm'][:,4],
                    lower_gain_mm=lower['cm'][rows,choice],predicted_contact=cm['joint_contact'][rows,choice],
                    predicted_release=cm['release'][rows,choice])
