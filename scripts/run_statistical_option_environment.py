"""Random options under independently randomized placements; no replay assumption."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_paired_option_environment import build_source as original
    source=original()
    changes={
        'offsets,option_cluster,option_raw12=matched_inputs(motion,assignment,args.eval_seed);before_placement=task._target_states.clone()':
        '_,option_cluster,option_raw12=matched_inputs(motion,assignment,args.eval_seed);offsets=placement_offsets(768,args.eval_seed).to(device);before_placement=task._target_states.clone()',
        'paired_initial_offsets=True':'paired_initial_offsets=False,independent_placements=True',
        'paired_joint_option=True':'statistical_option_collection=True',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source


def main():
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),
         {'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
