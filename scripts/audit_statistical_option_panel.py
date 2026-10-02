"""Native options/physics replay under independent placement, no prefix matching."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_paired_option_panel import build_source as original
    source=original()
    changes={
        'a.panel!=601':'a.panel not in (603,604)',
        "m['experiment_id']!='P-20261002-paired-option-task-opportunity'":"m['experiment_id']!='P-20261002-option-model-policy'",
        'torch.rand((192,2),generator=g)':'torch.rand((768,2),generator=g)',
        'offsets[envs]=draws[slots];clusters[envs]=slots':'offsets[envs]=draws[envs];clusters[envs]=slots',
        "or not initial['paired_initial_offsets']":"or initial['paired_initial_offsets'] or not initial['independent_placements']",
        "r['paired_joint_option']":"r['statistical_option_collection']",
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source


def main():
    exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),
         {'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
