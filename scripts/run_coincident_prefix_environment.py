"""P0-only short prefix in a co-located, collision-isolated engineering scene."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_paired_option_environment import build_source as original
    source = original()
    changes = {
        'physical_metadata=dict(native_body_names=':
        'physical_metadata=dict(environment_origins=[[task.gym.get_env_origin(env).x,task.gym.get_env_origin(env).y,task.gym.get_env_origin(env).z] for env in task.envs],native_body_names=',
        'for tick in range(202):':'for tick in range(56):',
        'raw=torch.where(active[:,None],option_raw12[selected],torch.zeros_like(option_raw12[selected]))':
        'raw=torch.zeros_like(option_raw12[selected])',
        '                    trace[key].append(value.cpu().clone())':
        '''                    trace[key].append(value.cpu().clone())
                if tick==55:
                    torch.save({k:torch.stack(v) for k,v in trace.items()},args.run_dir/'trace.pt')
                    result=dict(run_status='COMPLETED',engineering_only=True,physics_steps_each=56,options_executed=False,source_actor_calls=0,initial_sha256=sha(args.run_dir/'initial.pt'),trace_sha256=sha(args.run_dir/'trace.pt'),physical_metadata_sha256=sha(args.run_dir/'physical_metadata.json'))
                    (args.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\\n')
                    print(json.dumps(result),flush=True)
                    return''',
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
