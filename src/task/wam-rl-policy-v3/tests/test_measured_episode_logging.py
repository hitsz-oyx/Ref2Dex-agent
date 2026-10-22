"""Conservation at the real step/reset boundary, rather than a copied reward formula."""
import json
import sys
from pathlib import Path
import pytest
import numpy as np
ROOT=Path(__file__).resolve().parents[4]


def test_actual_step_logs_reward_phase_and_interaction_conservation(monkeypatch):
    sys.path.insert(0,str(ROOT/'src/task/ptv3-shared-wam-v6/tools/audit'))
    from audit_action_reset_cpu import reset_replay
    from wam_rl_policy_v3.measured_official_env import MeasuredOfficialEnv
    original=MeasuredOfficialEnv.step;seen=[];rewards=[]
    def step(env,*args,**kwargs):
        result=original(env,*args,**kwargs);seen.append(env);rewards.append(float(result[1].sum()))
        if env.steps==2:
            assert not env.reward_components[0].any() and not env.phase_controls[0].any()
            assert env.phase_controls[1].sum()==2
        if env.steps==4:
            assert not env.reward_components[1].any() and not env.phase_controls[1].any()
            assert env.phase_controls[0].sum()==2
        for episode in result[3]['episode_results'].values():
            assert sum(episode['reward_components'].values())==pytest.approx(episode['return_sum'])
            assert sum(episode['phase_controls'].values())==episode['controls']
        return result
    monkeypatch.setattr(MeasuredOfficialEnv,'step',step)
    assert reset_replay()['status']=='PASS'
    summary=seen[-1].training_statistics()
    assert summary['transitions']==10
    assert sum(summary[k] for k in summary if k.startswith('phase_'))==10
    assert summary['reward_sum']==pytest.approx(sum(rewards),abs=1e-6)
    assert sum(summary[k] for k in summary if k.startswith('reward_component_'))==pytest.approx(summary['reward_sum'],abs=1e-6)
    assert summary['completed_episode_controls']==6


def test_episode_file_is_bounded_and_overflow_is_explicit(tmp_path):
    from wam_rl_policy_v3.episode_logging import EpisodeLog
    path=tmp_path/'episodes.jsonl';log=EpisodeLog(path,max_bytes=25)
    log.append(dict(episode_id=1));log.append(dict(episode_id=2))
    assert json.loads(path.read_text())==dict(episode_id=1)
    assert log.summary()['written']==1 and log.summary()['dropped_after_cap']==1
    assert path.stat().st_size<=25
    with pytest.raises(FileExistsError):EpisodeLog(path)
