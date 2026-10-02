"""Original independent physics checks plus decision-state mean and held requests."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    source=(ROOT/'scripts/audit_continuous_critic_panel.py').read_text()
    changes={
        "a.panel not in list(range(547,567))+[568,569]":"a.panel != 583",
        "m['experiment_id']!='P-20261002-continuous-critic-policy'":"m['experiment_id']!='ENG-20261002-coherent-native-contract'",
        "r['deterministic']!=(a.panel in (568,569))":"r['deterministic'] or panel_record['update']!=20 or r['coherent_period']!=4 or r['request_decisions_each']!=51 or r['full_request_blocks']!=50 or r['final_partial_block_ticks']!=2",
        "            stored_mean=v('request_mean')[:,selected].reshape(-1,12);":"            decision_ticks=(np.arange(202)//4)*4\n            mu=mu.reshape(202,192,12)[decision_ticks].reshape(-1,12)\n            stored_mean=v('request_mean')[:,selected].reshape(-1,12);",
        "            raw=requested[:,selected].reshape(-1,12).astype(np.float64)":"            raw=requested[:,selected].reshape(-1,12).astype(np.float64)\n            if not np.allclose(v('request_logstd')[:,selected],logstd[None,None],atol=2e-6,rtol=0):raise ValueError('cached decision logstd')\n            if not np.array_equal(requested[:,selected],requested[decision_ticks][:,selected]) or not np.array_equal(v('request_noise')[:,selected],v('request_noise')[decision_ticks][:,selected]):raise ValueError('actual four-tick held request/noise')\n            if not np.array_equal(v('request_decision_tick')[:,selected],np.broadcast_to(decision_ticks[:,None],(202,192))) or not np.array_equal(v('request_is_decision')[:,selected],np.broadcast_to((np.arange(202)%4==0)[:,None],(202,192))):raise ValueError('actual Gaussian decision clock')",
        "        physical_target=np.concatenate(":"        if np.any(v('request_decision_tick')[:,arm==0]!=-1) or v('request_is_decision')[:,arm==0].any() or v('request_logstd')[:,arm==0].any():raise ValueError('unchanged arm has no Gaussian decisions')\n        physical_target=np.concatenate(",
        'causal_inputs_request_likelihood_target_projection_and_PD_verified=True,':'coherent_request_period=4,decision_state_likelihood_replayed=True,held_request_and_noise_verified=True,full202native_clock_verified=True,causal_inputs_request_likelihood_target_projection_and_PD_verified=True,',
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('exact coherent independent audit marker drift '+old)
        source=source.replace(old,new)
    return source

def main():
    exec(compile(build_source(),str(ROOT/'scripts/audit_continuous_critic_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_continuous_critic_panel.py')})

if __name__=='__main__':main()
