"""Identical P0-mean Gaussian behavior in three randomized arms, no antithetics."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_statistical_option_environment import build_source as original
    source=original()
    old='_,option_cluster,option_raw12=matched_inputs(motion,assignment,args.eval_seed);offsets=placement_offsets(768,args.eval_seed).to(device);before_placement=task._target_states.clone()'
    new='''_,option_cluster,_=matched_inputs(motion,assignment,args.eval_seed)
            option_generator=torch.Generator(device='cpu').manual_seed(args.eval_seed+18000)
            option_raw12=torch.randn((768,12),generator=option_generator).to(device)
            option_raw12[assignment==0]=0
            offsets=placement_offsets(768,args.eval_seed).to(device);before_placement=task._target_states.clone()'''
    assert source.count(old)==1;source=source.replace(old,new)
    assert source.count('statistical_option_collection=True')==1
    source=source.replace('statistical_option_collection=True','statistical_option_collection=True,independent_gaussian_options=True,gaussian_option_sigma=1.')
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
