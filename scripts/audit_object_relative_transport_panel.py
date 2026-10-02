"""Full native/P0/mesh replay with independent relative transport and preserved targets."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def build_source():
    from scripts.audit_self_trained_teacher_panel import build_source as original
    s=original().replace('(587,588)','(591,592)').replace('P-20261002-self-trained-teacher-qualification','P-20261002-object-relative-transport')
    s=s.replace("or r['no_source_actor_calls'] or not r['source_actor_self_trained'] or r['official_actor_used'] or r['source_actor_calls']!=201", "or not r['no_source_actor_calls'] or r['official_actor_used'] or r['source_actor_calls']!=0 or not r['construction_actor_only'] or not r['transport_contract']")
    b=s.index('        goal=base.copy()');e=s.index("        for key in ('request'",b)
    s=s[:b]+'''        goal=base.copy();delta=np.zeros_like(goal)
        reference_object=n('native_reference_object')[motion[None],planned_progress]
        dynamic=arm==2;static=arm==3
        delta[:,dynamic,:2]=np.clip(obj[:,dynamic,:2]-reference_object[:,dynamic,:2],-.02,.02)
        delta[:,static,:2]=n('placement_offsets')[static][None]
        if not np.array_equal(delta,v('transport_delta')):raise ValueError('independent causal relative transport delta')
        goal[...,:2]+=delta[...,:2]
        goal[...,:2]=np.maximum(np.minimum(goal[...,:2],q[...,:2]+np.abs(n('pd_scale')[:2])),q[...,:2]-np.abs(n('pd_scale')[:2]))
        maximum['target']=float(np.abs(goal-v('target')).max())
        if maximum['target']>1e-5:raise ValueError('native relative transport projection')
        if not np.array_equal(v('target')[...,2:],v('base_target')[...,2:]):raise ValueError('same-state lift/rotation/fingers preserved exactly')
''' + s[e:]
    s=s.replace('expert_replay_separate_gpu=True,full1442_feature_derivation_not_independently_replayed=True,','relative_transport_independently_reconstructed=True,same_state_lift_rotation_finger_targets_preserved=True,no_source_actor_forward=True,')
    return s

def main():
    exec(compile(build_source(),str(ROOT/'scripts/audit_continuous_critic_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_continuous_critic_panel.py')})
if __name__=='__main__':main()
