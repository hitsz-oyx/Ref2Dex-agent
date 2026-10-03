"""Independent variable-count/prefix native audit, no labels for truncated data."""
import re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source(seed):
    assert seed in (651,652,653,654);short=seed in (651,652);extra=False
    from scripts.audit_statistical_option_panel import build_source as original
    source=original().replace('a.panel not in (603,604)','a.panel not in (651,652,653,654)').replace("m['experiment_id']!='P-20261002-option-model-policy'","m['experiment_id']!='P-20261003-budgeted-physical-critic'")
    marker="        physical=json.loads((directory/'physical_metadata.json').read_text())";assert source.count(marker)==1
    source=source.replace(marker,marker+"\n        if abs(physical['control_dt']-1/30)>1e-8 or abs(physical['control_dt']-physical['simulation_dt']*physical['control_frequency_inverse'])>1e-8:raise ValueError('actual control/simulation time accounting')")
    if extra:
        for old,new in ((768,384),(256,128),(192,96),(64,32)):source=re.sub(r'\b'+str(old)+r'\b',str(new),source)
    if short:
        source=source.replace("r['learned_policy_calls']!=202","r['learned_policy_calls']!=101")
        source=source.replace("v('progress').shape!=(202,768)","v('progress').shape!=(101,768)").replace('np.broadcast_to(np.arange(1,203)[:,None],(202,768))','np.broadcast_to(np.arange(1,102)[:,None],(101,768))').replace('np.arange(1,203)[:,None]','np.arange(1,102)[:,None]').replace('np.arange(202)','np.arange(101)').replace('reshape(202,768','reshape(101,768')
        source=source.replace("physical_steps_each", "physical_steps_each")
        start=source.index('        rise=');end=source.index('    if len(rows)!=768',start)
        source=source[:start]+"""        if not r['no_terminal_task_labels'] or not r['truncated_before_task_completion'] or r['physical_steps_each']!=101:raise ValueError('short physical episode is unlabeled')
        for env in range(768):rows.append(dict(seed=seed,environment=env,motion=int(motion[env]),arm=int(arm[env]),terminal_task_label_available=False))
"""+source[end:]
    source=source.replace('paired_option_draw_initialization_and_execution_verified=True,',f'variable_count_and_prefix_native_contract_verified=True,complete_terminal_task_labels_available={not short},')
    return source


if __name__=='__main__':
    sys.path.insert(0,str(ROOT));seed=int(sys.argv[sys.argv.index('--panel')+1]);exec(compile(build_source(seed),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})
