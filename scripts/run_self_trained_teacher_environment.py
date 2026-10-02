"""Qualified repository self-trained actor in the unchanged native reset/physics scaffold."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    s=(ROOT/'scripts/run_continuous_critic_environment_v2.py').read_text().replace('768','192')
    old="    from isaacgym import gymapi,gymtorch"
    new="""    def policy_groups(motion,seed):
        g=torch.Generator(device='cpu').manual_seed(seed+16000)
        result=torch.full((192,),-1,dtype=torch.long)
        for m in range(3):
            ids=(motion.cpu()==m).nonzero().flatten()
            if len(ids)!=64:raise ValueError('192env balanced qualification')
            result[ids]=torch.arange(4).repeat_interleave(16)[torch.randperm(64,generator=g)]
        return result
    from isaacgym import gymapi,gymtorch"""
    changes={old:new,
        "            raise RuntimeError('learned policy invocation forbidden in static diagnostic')":"            return super().get_action(*args,**kwargs)",
        "if [int((task.data_id==m).sum()) for m in range(3)]!=[256]*3":"if [int((task.data_id==m).sum()) for m in range(3)]!=[64]*3",
        "            scale12=task._pd_action_scale[list(INDEPENDENT)].abs()":"            if self.is_rnn or not self.normalize_input or not self.clip_actions:raise ValueError('frozen feedforward/RMS/clip expert interface')\n            if self.obs_shape!=(1442,):raise ValueError('native expert observation dimension')\n            self.states=None\n            scale12=task._pd_action_scale[list(INDEPENDENT)].abs()",
        "'normalized_context','request_noise')}":"'normalized_context','request_noise','teacher_observation','teacher_action','teacher_observation_valid','teacher_preprogress')}",
        "                for group,variant in enumerate(('cm','state_only','none'),1):":"                teacher_observation=torch.zeros((192,1442),device=device);teacher_action=torch.zeros((192,18),device=device)\n                teacher_observation_valid=torch.full((192,),tick>0,device=device,dtype=torch.bool);teacher_preprogress=task.progress_buf.clone()\n                if tick>0:\n                    task._compute_observations(ids)\n                    teacher_observation=task.obs_buf.clone()\n                    teacher_action=self.get_action({'obs':teacher_observation},True)\n                    teacher_action[:,[7,9,11,13,16,17]]=0\n                    expert_goal=task._action_to_pd_targets(teacher_action.clone())\n                    goal[assignment>=2]=expert_goal[assignment>=2]\n                for group,variant in ():",
        "normalized_context=normalized_context,request_noise=request_noise)":"normalized_context=normalized_context,request_noise=request_noise,teacher_observation=teacher_observation,teacher_action=teacher_action,teacher_observation_valid=teacher_observation_valid,teacher_preprogress=teacher_preprogress)",
        "no_source_actor_calls=True,learned_policy_calls=202":"no_source_actor_calls=False,source_actor_self_trained=True,source_actor_calls=201,official_actor_used=False,source_actor_checkpoint_sha256=args.checkpoint_sha256,source_actor_first_tick=1,learned_policy_calls=202",
        "engineering_only=False,no_policy_utility_claim=True":"engineering_only=False,unused_continuous_heads=True,no_policy_utility_claim=True",
    }
    for old,new in changes.items():
        if s.count(old)!=1:raise ValueError('teacher scaffold marker drift '+old)
        s=s.replace(old,new)
    # Frozen old residual heads are not evaluated or initialized: metadata checkpoint only.
    begin=s.index("            for group,variant in enumerate(('cm','state_only','none'),1):")
    end=s.index('            if self.is_rnn',begin)
    s=s[:begin]+s[end:]
    return s

def main():
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
if __name__=='__main__':main()
