"""New fixed cohorts; retain every existing native audit and tolerance."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / 'scripts/audit_truth_successor_native_panel.py').read_text()
    changes = {
        "a.panel!=595": "a.panel not in (597,598)",
        "m['experiment_id']!='P-20261002-truth-successor-task-value'":
        "m['experiment_id']!='P-20261002-observed-support-task-value'",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, str(ROOT / 'scripts/audit_truth_successor_native_panel.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts/audit_truth_successor_native_panel.py')})


if __name__ == '__main__':
    main()
