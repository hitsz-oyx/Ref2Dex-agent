"""Export the fixed-final-step teacher-control probe as CSV, Markdown and a plot."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    result = json.loads((run / 'result.json').read_text())
    audit = json.loads((run / 'independent-audit.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    curve = json.loads((run / 'curve.json').read_text())
    if result['status'] != 'COMPLETED' or audit['status'] != 'PASS' or manifest['engineering_only']:
        raise ValueError('audited scientific run required')
    out = run / 'summary'
    out.mkdir(exist_ok=False)
    rows = []
    for group, report in result['final']['groups'].items():
        for phase in ['all', 'startup', 'later', 'middle_phase', 'interpolation_eligible_later']:
            if phase in report:
                rows.append(dict(group=group, phase=phase, **report[phase]))
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Full-teacher control: fixed final step 24000', '',
             '| Group | Phase | Ticks | Sequences | Sequence mean mm | Tick mean mm | Tick p95 mm |',
             '|---|---|---:|---:|---:|---:|---:|']
    def number(value):
        return 'n/a' if value is None else '%.3f' % value
    for r in rows:
        lines.append('| %s | %s | %d | %d | %s | %s | %s |' % (
            r['group'], r['phase'], r['count'], r['sequences'], number(r['sequence_mean_mm']),
            number(r['mean_mm']), number(r['p95_mm'])))
    lines += ['', 'Interpolation has no startup rows. Groups differ in phase membership;',
              'use the stratified rows and per-sequence results alongside the overall table.',
              'Single-seed offline Probe; no closed-loop success claim.', '']
    (out / 'table.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ['Seen training states', 'Isolated same-sequence block', 'Unseen sequences']
    colors = ['#2b8cbe', '#41ab5d', '#e6550d']
    for group, label, color in zip(['seen', 'interpolation', 'unseen'], labels, colors):
        axes[0].plot([r['step'] for r in curve],
                     [r['groups'][group]['all']['sequence_mean_mm'] for r in curve],
                     label=label, color=color)
    axes[0].set(xlabel='Training updates', ylabel='Sequence mean wrist PD error (mm)', yscale='log')
    axes[0].axvline(12000, color='gray', linestyle=':', linewidth=1)
    axes[0].legend(fontsize=8)
    groups = result['final']['groups']
    values = [groups[g]['all']['sequence_mean_mm'] for g in ['seen', 'interpolation', 'unseen']]
    bars = axes[1].bar(['Seen', 'Same-seq heldout', 'Unseen seq'], values, color=colors)
    axes[1].bar_label(bars, labels=['%.2f' % v for v in values], padding=3)
    axes[1].set(ylabel='Sequence mean wrist PD error (mm)', title='Fixed step 24000; complete groups')
    axes[1].set_ylim(0, max(values) * 1.2)
    figure.suptitle('Full1442 teacher inputs; Inspire cup/mug family; offline Probe')
    figure.tight_layout()
    figure.savefig(out / 'generalization.png', dpi=160)
    plt.close(figure)
    (out / 'manifest.json').write_text(json.dumps(dict(
        source_run=str(run), training_git_commit=manifest['git_commit'],
        fixed_step=24000, audit='PASS', artifacts=['metrics.csv', 'table.md', 'generalization.png']), indent=2) + '\n')
    print(str(out))


if __name__ == '__main__':
    main()
