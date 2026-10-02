"""Full native/P0 audit with explicit fixed options and independent initial pairing."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_source():
    source = (ROOT / 'scripts/audit_truth_successor_native_panel.py').read_text()
    changes = {
        'a.panel!=595': 'a.panel!=601',
        "m['experiment_id']!='P-20261002-truth-successor-task-value'":
        "m['experiment_id']!='P-20261002-paired-option-task-opportunity'",
        "g=torch.Generator(device='cpu').manual_seed(seed+11000);offsets=((torch.rand((768,2),generator=g)*2-1)*.01).numpy()":
        """g=torch.Generator(device='cpu').manual_seed(seed+11000);draws=((torch.rand((192,2),generator=g)*2-1)*.01).numpy()
        g=torch.Generator(device='cpu').manual_seed(seed+18000);options=torch.randn((192,12),generator=g).numpy()
        offsets=np.zeros((768,2),np.float32);clusters=np.full(768,-1,dtype=np.int64);option_raw=np.zeros((768,12),np.float32)
        for mo in range(3):
            slots=np.arange(mo*64,(mo+1)*64)
            for group in range(4):
                envs=np.flatnonzero((motion==mo)&(arm==group));assert len(envs)==64
                offsets[envs]=draws[slots];clusters[envs]=slots
                if group in (2,3):option_raw[envs]=options[slots]*(1 if group==2 else -1)
        if not np.array_equal(clusters,n('option_cluster')) or not np.array_equal(option_raw,n('option_raw12')) or initial['option_draw_seed']!=seed+18000 or not initial['paired_initial_offsets']:raise ValueError('independent private option and matched group reconstruction')""",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    begin = source.index("        for group,variant", source.index("z=v('normalized_context')"))
    end = source.index('        physical_target=', begin)
    source = source[:begin] + '''        active=(np.arange(202)[:,None]>=steps[None])&(np.arange(202)[:,None]<stops[None]+30)
        expected_requests=np.where(active[...,None],option_raw[None],np.zeros_like(option_raw)[None])
        if not np.array_equal(expected_requests,requested):raise ValueError('fixed option start/stop/antithetic draws')
        for channel in ('request_mean','request_noise','request_logprob','critic_value','aux_prediction'):
            if np.any(v(channel)):raise ValueError('unused request-policy channel must be zero '+channel)
        if not r['paired_joint_option'] or r['request_policy_used'] or r['option_sampling_std']!=1. or not r['option_held_until_task_deadline'] or r['continuous_updates']!=0 or checkpoint['updates']!=0:raise ValueError('held-option collection semantics')
        # Full P0 forward on actual saved normalized inputs, independent NumPy.
        flat=z.reshape(-1,70);state=base_checkpoint['model']
        layers=sorted(int(k.split('.')[1]) for k in state if k.endswith('.weight'))
        for j,index in enumerate(layers):
            flat=layer(flat,state,'network.'+str(index))
            flat=np.tanh(flat) if j==len(layers)-1 else np.maximum(flat,0)
        flat=flat.reshape(202,768,18);flat[...,[7,9,11,13,16,17]]=0
        p0_error=float(np.abs(flat-v('model_residual')).max())
        if p0_error>2e-5:raise ValueError(('full P0 NumPy replay',p0_error))
''' + source[end:]
    marker='report=dict(run_status='
    assert source.count(marker) == 1
    source = source.replace(marker, 'report=dict(paired_option_draw_initialization_and_execution_verified=True,full_p0_numpy_forward_error=p0_error,run_status=')
    return source


def main():
    exec(compile(build_source(), str(ROOT / 'scripts/audit_truth_successor_native_panel.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts/audit_truth_successor_native_panel.py')})


if __name__ == '__main__':
    main()
