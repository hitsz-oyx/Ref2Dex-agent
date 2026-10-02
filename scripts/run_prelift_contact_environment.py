"""One fixed early preparation option, then unchanged P0 feedback."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_statistical_option_environment import build_source as original
    source=original()
    changes={
        'decision_steps=lift_starts[motion]-8':'decision_steps=lift_starts[motion]-32',
        'active=(tick>=decision_steps[selected])&(tick<stops[motion[selected]]+30)':'active=(tick>=decision_steps[selected])&(tick<decision_steps[selected]+24)',
        'option_held_until_task_deadline=True':'option_held_until_task_deadline=False,prelift_option_opportunity=True,option_control_steps=24,option_returns_to_p0_after24=True',
        'option_draw_seed=args.eval_seed+18000,paired_initial_offsets=False':'option_draw_seed=args.eval_seed+18000,option_control_steps=24,paired_initial_offsets=False',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
