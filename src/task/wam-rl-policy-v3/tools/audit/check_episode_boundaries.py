#!/usr/bin/env python3
"""Bounded native integration smoke through the real official PPO entry point.

Accepts closed_loop_scheduled_shadow arguments; injects one env-0 termination.
No success-rate claim: checks reset/sample ownership, histories and real updates.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]


def main():
    def argument(name):
        return sys.argv[sys.argv.index(name)+1]
    if not 2<=int(argument('--ppo-updates'))<=4 or not 2<=int(argument('--envs'))<=32:
        raise ValueError('Smoke requires 2–4 updates and 2–32 environments')
    if int(argument('--seconds'))>300 or '--official-ppo' not in sys.argv:
        raise ValueError('Official bounded smoke required')
    from isaacgym import gymapi  # import native bindings before torch
    import torch
    import wam_rl_policy_v3.official_bridge as bridge
    original_run=bridge.run

    def audited_run(task,snapshot,sim_step,actor,runtime,args):
        task.random_time=False
        trace=[];physical=np.zeros(args.envs,int)
        forced_tick=16 if args.rank==1 and os.environ.get('REF2DEX_AUDIT_STAGGERED')=='1' else 8
        current=dict(active=False,step=0,forced=False,canonical=None)
        original_pre=task.pre_physics_step
        original_reward=task.compute_reward
        original_step=bridge.ScheduledOfficialEnv.step

        def pre(actions):
            if current['active']:
                physical[task.reset_buf.cpu().numpy().astype(bool)]+=1
                current['canonical']=actions.clone()
            return original_pre(actions)

        def reward(*a,**kw):
            result=original_reward(*a,**kw)
            if current['active'] and current['step']==forced_tick:
                task.reset_buf[0]=1
                if args.rank==1 and args.envs>2:task.reset_buf[2]=1
                current['forced']=True
            return result

        def step(env,actions,id=0):
            current['active']=True
            before=actions.clone()
            result=original_step(env,actions,id)
            current['active']=False
            observation,reward,done,info=result
            torch.testing.assert_close(actions,before,rtol=0,atol=0)
            valid=info['transition_valid'].cpu().numpy()
            initial=info['reset_initialization']
            np.testing.assert_array_equal(valid,~initial)
            np.testing.assert_array_equal(info['execution_episode'],physical)
            np.testing.assert_array_equal(info['action_source_episode'][valid],physical[valid])
            if initial.any():
                assert not current['canonical'][initial].any()
                assert not reward[initial].any() and done[initial].all()
                assert not observation[initial,300:].any()
                assert all(len(env.inputs.histories[i])==1 for i in np.flatnonzero(initial))
                if current['step']==forced_tick+1:
                    assert len(env.inputs.histories[1])==7
            trace.append(dict(tick=current['step'],initialization=initial.tolist(),
                valid=valid.tolist(),observation_episode=info['action_source_episode'].tolist(),
                execution_episode=physical.tolist()))
            current['step']+=1
            return result

        task.pre_physics_step=pre;task.compute_reward=reward
        bridge.ScheduledOfficialEnv.step=step
        reports=original_run(task,snapshot,sim_step,actor,runtime,args)
        assert current['forced'] and trace[forced_tick+1]['initialization'][0]
        report=dict(status='PASS',scope='native engineering smoke; not policy evaluation',
            steps=len(trace),forced_terminal_tick=forced_tick,trace=trace,reports=reports)
        (args.output/'boundary-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        return reports

    bridge.run=audited_run
    runpy.run_path(str(TASK/'tools/run/closed_loop_scheduled_shadow.py'),run_name='__main__')


if __name__=='__main__':main()
