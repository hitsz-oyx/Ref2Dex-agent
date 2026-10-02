"""Retained independent reset, PD, mesh and causal state checks; independent P0 forward."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    s=(ROOT/'scripts/audit_continuous_critic_panel.py').read_text().replace('768','192')
    changes={
        "a.panel not in list(range(547,567))+[568,569]":"a.panel not in (587,588)",
        "m['experiment_id']!='P-20261002-continuous-critic-policy'":"m['experiment_id']!='P-20261002-self-trained-teacher-qualification'",
        "or not r['no_source_actor_calls']":"or r['no_source_actor_calls'] or not r['source_actor_self_trained'] or r['official_actor_used'] or r['source_actor_calls']!=201 or r['source_actor_checkpoint_sha256']!=m['teacher_sha256']",
        "np.ones(4,dtype=int)*64":"np.ones(4,dtype=int)*16",
        "torch.arange(4).repeat_interleave(64)[torch.randperm(256,generator=private)]":"torch.arange(4).repeat_interleave(16)[torch.randperm(64,generator=private)]",
        "r['deterministic']!=(a.panel in (568,569))":"r['deterministic'] or not r['unused_continuous_heads']",
    }
    for old,new in changes.items():
        if s.count(old)!=1:raise ValueError('teacher audit marker drift '+old)
        s=s.replace(old,new)
    b=s.index('        goal=base.copy();requested=');e=s.index('        # Raw-state reconstruction',b)
    s=s[:b]+'''        goal=base.copy()
        expert=v('teacher_action').copy();expert[...,6:]=(1+expert[...,6:])/2
        expert_pd=n('pd_offset')+n('pd_scale')*expert;expert_pd[...,:6]+=q[...,:6]
        for parent,children in COUPLING.items():
            for child,ratio in children:expert_pd[...,child]=expert_pd[...,parent]*ratio
        active=arm>=2;goal[1:,active]=expert_pd[1:,active]
        maximum['target']=float(np.abs(goal-v('target')).max())
        if maximum['target']>1e-5:raise ValueError('teacher action/current PD targets')
        if not np.array_equal(v('teacher_preprogress'),np.broadcast_to(np.arange(202)[:,None],(202,192))) or not np.array_equal(v('teacher_observation_valid'),np.broadcast_to((np.arange(202)>0)[:,None],(202,192))):raise ValueError('expert pre-state causal clock')
        if v('teacher_observation').shape!=(202,192,1442) or v('teacher_observation')[0].any() or v('teacher_action')[0].any():raise ValueError('explicit common firsttick')
        for key in ('request','request_mean','request_noise','request_logprob','critic_value','aux_prediction','executed_action12'):
            if v(key).any():raise ValueError('unused continuous model trace '+key)
''' + s[e:]
    b=s.index('        z=v(\'normalized_context\').astype(np.float64)');e=s.index('        physical_target=np.concatenate(',b)
    s=s[:b]+'''        z=v('normalized_context').astype(np.float64).reshape(-1,70)
        for index in range(0,len(base_checkpoint['model'])//2):
            key='network.'+str(index*2)
            z=layer(z,base_checkpoint['model'],key)
            z=np.tanh(z) if index==len(base_checkpoint['model'])//2-1 else np.maximum(z,0)
        z=z.reshape(202,192,18);z[...,[7,9,11,13,16,17]]=0
        maximum['mean']=float(np.abs(z-v('model_residual')).max())
        if maximum['mean']>2e-5:raise ValueError('independent P0 fullforward')
''' + s[e:]
    s=s.replace('no_policy_update_inside_rollout=True,','expert_replay_separate_gpu=True,full1442_feature_derivation_not_independently_replayed=True,no_policy_update_inside_rollout=True,')
    return s

def main():
    exec(compile(build_source(),str(ROOT/'scripts/audit_continuous_critic_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_continuous_critic_panel.py')})
if __name__=='__main__':main()
