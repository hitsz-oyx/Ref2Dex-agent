"""Fresh half-episode physical data or complete reward episodes with fixed counts."""
import argparse,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source(kind):
    assert kind in ('short','common','extra');n=768;ticks=101 if kind=='short' else 202
    from scripts.run_statistical_option_environment import build_source as original
    source=original();marker='    from isaacgym import gymapi,gymtorch';assert source.count(marker)==1;source=source.replace(marker,'    from src.task.CmResidual.budgeted_physical_q import policy_groups,matched_inputs\n'+marker)
    marker='physical_metadata=dict(';assert source.count(marker)==1;source=source.replace(marker,"physical_metadata=dict(control_dt=float(task.dt),simulation_dt=float(task.sim_params.dt),control_frequency_inverse=int(task.control_freq_inv),simulation_substeps=int(task.sim_params.substeps),")
    source=source.replace('statistical_option_collection=True',f"statistical_option_collection=True,budget_panel_kind='{kind}',no_terminal_task_labels={kind=='short'},truncated_before_task_completion={kind=='short'}")
    if kind=='short':
        replacements={
            'for tick in range(202)':'for tick in range(101)',
            "if (phase_steps!=90).any():raise ValueError('incomplete native holding phase')":"if not torch.equal(phase_steps,(101-anchors[motion]+1).clamp(0,90)):raise ValueError('complete truncated physical prefix')",
            'physical_steps_each=202,phase_steps_each=90':'physical_steps_each=101,phase_steps_each=None,recorded_phase_steps=phase_steps.cpu().tolist()',
            "learned_policy_calls=202 if args.mode=='policy' else 0":"learned_policy_calls=101 if args.mode=='policy' else 0",
        }
        for old,new in replacements.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source


if __name__=='__main__':
    sys.path.insert(0,str(ROOT));p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--budget-panel-kind',choices=('short','common','extra'),required=True);a,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    exec(compile(build_source(a.budget_panel_kind),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
