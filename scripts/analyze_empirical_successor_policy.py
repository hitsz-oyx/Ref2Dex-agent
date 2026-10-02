"""Exactly unchanged actual-policy gates on fresh623/624; JSON-safe scalars."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    source=(ROOT/'scripts/analyze_option_model_policy.py').read_text()
    assert source.count('(611,612)')==2;source=source.replace('(611,612)','(623,624)')
    source=source.replace('pretraining_trajectories=1536,evaluation_trajectories=1536','pretraining_trajectories_reused=1536,evaluation_trajectories=1536,new_physical_optimizer_steps=3000,new_actor_optimizer_steps=2000,direct_q_optimizer_steps_reused=1000')
    source=source.replace('json.dumps(result,indent=2)','json.dumps(result,indent=2,default=lambda v:v.item())').replace('json.dumps(result)','json.dumps(result,default=lambda v:v.item())')
    return source


if __name__=='__main__':
    exec(compile(build_source(),str(ROOT/'scripts/analyze_option_model_policy.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/analyze_option_model_policy.py')})
