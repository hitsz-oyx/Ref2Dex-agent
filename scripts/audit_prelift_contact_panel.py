"""Full native audit with exact finite preparation and return-to-P0 contract."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_statistical_option_panel import build_source as original
    source=original()
    changes={
        'a.panel not in (603,604)':'a.panel not in (629,630)',
        "m['experiment_id']!='P-20261002-option-model-policy'":"m['experiment_id']!='P-20261002-prelift-contact-opportunity'",
        "steps,n('lift_start')[motion]-8":"steps,n('lift_start')[motion]-32",
        'active=(np.arange(202)[:,None]>=steps[None])&(np.arange(202)[:,None]<stops[None]+30)':'active=(np.arange(202)[:,None]>=steps[None])&(np.arange(202)[:,None]<steps[None]+24)',
        "or not r['option_held_until_task_deadline']":"or r['option_held_until_task_deadline'] or not r['prelift_option_opportunity'] or not r['option_returns_to_p0_after24'] or r['option_control_steps']!=24 or initial['option_control_steps']!=24",
        "physical105=bool(valid[retained,env].all())":"physical105=bool(valid[retained,env].all()),any_joint_lift_in105=bool(valid[retained,env].any())",
        'paired_option_draw_initialization_and_execution_verified=True':'independent_zero_duplicate_and_antithetic_options_verified=True,early24tick_active_window_verified=True,return_to_p0_after_end_verified=True',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    marker="        for channel in ('request_mean'"
    assert source.count(marker)==1
    block="""        if not np.all(active.sum(0)==24):raise ValueError('exact24native-controlticks')
        inactive=~active
        p0_restore_error=float(np.abs(v('target')[inactive]-v('base_target')[inactive]).max())
        if p0_restore_error>1e-5:raise ValueError(('P0beforestart/afterend',p0_restore_error))
"""
    source=source.replace(marker,block+marker)
    source=source.replace('full_p0_numpy_forward_error=p0_error,','full_p0_numpy_forward_error=p0_error,p0_restore_maximum=p0_restore_error,')
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})
