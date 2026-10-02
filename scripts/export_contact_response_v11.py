"""Preserve v10 and prepare native TeX v11 only from the complete audited probe."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
BASE = ROOT / 'src/task/CmResidual/research/contact_response/output'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latex_cell(value):
    # Recorded cells are plain text, not executable TeX. Preserve their values.
    chars = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
             '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(chars.get(c, c) for c in value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closeout', type=Path, required=True)
    args = parser.parse_args()
    closeout = args.closeout.resolve()
    result = json.loads(closeout.read_text())
    assert result['run_status'] == 'COMPLETED'
    assert result['fixed_final_only_integer_gates_independently_reconstructed']
    assert result['scientific_sources_unchanged']
    assert result['one_optimization_seed_probe_only'] and not result['journal_ready']
    for path, digest in result['source_sha256'].items():
        assert sha(path) == digest, path
    old = json.loads((PAPER / 'export_manifest-v10.json').read_text())
    for path, digest in old['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(PAPER / 'manuscript-v10.pdf') == old['output_sha256']
    destination = PAPER / 'manuscript-v11.tex'
    tabledir = PAPER / 'tables-v11'
    manifest_path = PAPER / 'source_manifest-v11.json'
    if any(p.exists() for p in [destination, tabledir, manifest_path]):
        raise FileExistsError('retain all previous revisions and attempts')
    source = (PAPER / 'manuscript-v10.tex').read_text()
    source = source.replace('revision 10', 'revision 11')
    tabledir.mkdir()
    tables = []
    for relative in re.findall(r'\\input\{(tables/[^}]+)\}', source):
        original = PAPER / relative
        lines = original.read_text().splitlines()
        escaped = []
        for line in lines:
            if '&' in line and line.rstrip().endswith('\\\\'):
                cells = [v.strip() for v in line.rstrip()[:-2].split('&')]
                line = ' & '.join(latex_cell(v) for v in cells) + r'\\'
            escaped.append(line)
        target = tabledir / original.name
        target.write_text('\n'.join(escaped) + '\n')
        tables.append(target)
        source = source.replace('\\input{' + relative + '}',
                                '\\input{tables-v11/' + target.name + '}')
    names = ['unchanged', 'cm', 'state_only', 'none']
    cohorts = [('Pooled /384', result['pooled'])]
    cohorts += [('Seed ' + k + ' /192', v) for k, v in result['by_seed'].items()]
    cohorts += [('Motion ' + k + ' /128', v) for k, v in result['by_motion'].items()]
    table = tabledir / 'continuous_policy-r1.tex'
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Cohort & Unchanged & Cm & State only & No auxiliary\\', r'\midrule']
    for label, cohort in cohorts:
        lines.append(label + ' & ' + ' & '.join(str(cohort[n]['success']) for n in names) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table.write_text('\n'.join(lines) + '\n'); tables.append(table)
    pooled = result['pooled']
    counts = ', '.join(n.replace('_', ' ') + ' ' + str(pooled[n]['success']) + '/384' for n in names)
    gain_state = 100 * (pooled['cm']['rate'] - pooled['state_only']['rate'])
    gain_none = 100 * (pooled['cm']['rate'] - pooled['none']['rate'])
    failed = [k for k, value in result['gates'].items() if not value]
    if result['label'] == 'UNPROMISING':
        decision = ('The fixed full-coordinate auxiliary-critic recipe is UNPROMISING: '
                    'it fails ' + str(len(failed)) + ' of seven prospective gates. '
                    'This exact recipe stops without coefficient, learning-rate, '
                    'seed, checkpoint or training-duration searches. It does not '
                    'refute other uses of physical interaction models.')
        abstract = ('A complete continuous-action auxiliary-critic comparison uses '
                    '15360 training and 1536 final-only evaluation trajectories. '
                    'Its pooled physical105 successes are ' + counts + '; '
                    'the prospective Cm utility gate fails. ')
    else:
        decision = ('The fixed full-coordinate auxiliary-critic recipe is PROMISING '
                    'under all seven prospective gates. This is one optimization '
                    'seed on a privileged-state synthetic-reference task, not '
                    'formal validation, an identified physical mechanism or a '
                    'new auxiliary-learning method. Independent optimization '
                    'seeds and matched mechanism controls are required next.')
        abstract = ('A complete continuous-action auxiliary-critic comparison uses '
                    '15360 training and 1536 final-only evaluation trajectories. '
                    'Its pooled physical105 successes are ' + counts + '; '
                    'all prospective Cm utility gates pass as a single-seed Probe. ')
    old_abstract = ('A new\ncontinuous-action auxiliary-critic comparison has passed engineering and one\n'
                    'training-update audit but remains incomplete, with no prospective utility result.\n')
    assert old_abstract in source
    source = source.replace(old_abstract, abstract + '\n')
    start = source.index(r'\section{Continuous policy comparison: implementation evidence only}')
    end = source.index(r'\section{Limitations and research decision}', start)
    section = r'''\section{Complete continuous policy-training comparison}
We freeze a 12-coordinate continuous PPO comparison: Cm physical auxiliary
critic, state-only auxiliary control and no-auxiliary control, all with identical
initial actor and critic weights, 20 updates and equal native interactions.
The value baseline is state-only. Cm receives current state and executed target
minus current joint position, and predicts actual next object position and
linear-velocity changes. Separate actor and critic encoders prevent direct
auxiliary gradients to the actor; joint gradient clipping can nevertheless
couple the actor update scale to critic gradients. The only reward is actual
terminal physical105, never a model prediction or surrogate reward.

Gaussian request likelihood is computed before tanh and native projection;
it is not executed-target density or entropy. Auxiliary actor-critic objectives
are established \cite{jaderberg2016}, and clipped-tail score correction
overlaps CAPG \cite{fujita2018}. This recipe is not a new general method.

Training uses seeds 547--566, 768 independent randomized native environments
per panel, 192 per arm across three motions. All three learned variants complete
3840 training episodes and 3040 PPO minibatch updates each, totaling 9120
optimizer steps. Evaluation uses only the final checkpoint, deterministic
actions and preselected seeds 568/569. The engineering seed 567 never enters
training or evaluation. The primary gate requires pooled Cm success to exceed
BOTH learned controls by five percentage points, Cm to be no worse than unchanged
pooled, and Cm to be no worse than both learned controls on EACH evaluation seed.
All seven conditions are required; no subgroup can rescue a failed gate.

\begin{table}[ht]
\centering
\input{tables-v11/continuous_policy-r1.tex}
\caption{Final-checkpoint-only physical105 success counts. Denominators in each
row apply to each arm. Motions and environment seeds share one optimization
seed; these are Probe results, not independent training replications.}
\end{table}
'''
    section += ('\nPooled counts are ' + counts + '. Cm minus state-only is '
                + f'{gain_state:.3f}' + r' percentage points; Cm minus no auxiliary is '
                + f'{gain_none:.3f}' + ' percentage points. ' + decision + '\n\n')
    section += r'''All 22 native panels are independently reconstructed from their raw
trajectories, including current-state inputs, request probabilities, executable
targets, native PD and complete-mesh physical105 labels. All actor and critic
gradients and Adam updates for 60 predetermined first minibatches are checked
independently in NumPy; the other 9060 optimizer steps are retained and counted
but not independently replayed. Full coverage and all seven final gates are
separately reconstructed with integer success counts.

The original update-17 NumPy audit fails its frozen gradient threshold at one
first-layer ReLU: float64 and GPU float32 arithmetic give opposite signs within
roundoff of zero. A saved-batch GPU replay reproduces every saved gradient,
forward output and loss exactly, without optimizer or simulation updates. A
separate NumPy audit uses the witnessed branch only after independently checking
the FP32 dot-product rounding enclosure. All ORIGINAL scalar tolerances then
pass, including maximum gradient discrepancy of 1.223e-6. The original failed
audit/parent remain unchanged, and the correction is recorded separately.
Training resumes from the exact saved update-17 Adam state; no valid optimizer
update is repeated. The audit is explicitly branch-aware, not a claim that
float32 and float64 derivatives agree at nonsmooth boundaries.

After an execution interruption, completed panels and full Adam checkpoints are
retained exactly. A subsequent native allocator failure on GPU1 is preserved,
alongside one engineering-only reproduction excluded from training and testing.
The remaining phases run on an admitted idle GPU4 of the same RTX3090 model;
this is an explicitly recorded runtime migration, not uninterrupted execution
on a single device. No valid panel or optimizer update is repeated or selected.
Initial reset packets match across the failed and recovered panel, without a
claim of cloned solver state. All controls share the device within each panel.

Physical105 means root rise and complete-mesh table clearance on all 105
specified ticks. It does not establish force closure or mechanically supported
grasp in every successful learned-policy trajectory. An earlier randomized
support-removal witness concerns a different base-policy cohort; it is not
automatically transferred to the learned policy.

'''
    source = source[:start] + section + source[end:]
    source = source.replace('The first matched frozen-feature policy-training comparison is negative;',
                            'The matched frozen-feature comparison is negative; the continuous comparison is a completed single-seed Probe;')
    source = source.replace('Matched Cm-on/off POLICY TRAINING has now been executed without utility;',
                            'Matched Cm-on/off policy training has been executed in two distinct fixed designs;')
    source = source.replace('The PDF is a ReportLab review copy; portable LaTeX source and tables are also\nretained.',
                            'Revision 11 is compiled from native LaTeX. Earlier ReportLab review copies,\ntheir exact inputs and the failed revision-10 native compilation are retained.')
    source = source.replace('Negative outcomes and unresolved policy utility are retained',
                            'All negative outcomes and the bounded continuous Probe are retained')
    # Historical prose has three plain-text subscript names outside math.
    for value in ['Z_j', 'tau_j(X)', 'lift_start']:
        source = source.replace(value, value.replace('_', r'\_'))
    destination.write_text(source)
    inputs = {str((ROOT / p).resolve()): h for p, h in old['source_sha256'].items()}
    inputs.update(result['source_sha256'])
    for path in [closeout, destination, Path(__file__), *tables]:
        inputs[str(path.resolve())] = sha(path)
    manifest = dict(run_status='SOURCE_COMPLETED', rendering='native TeX source; compilation pending',
                    source_sha256=inputs, continuous_label=result['label'],
                    continuous_counts=result['pooled'], gates=result['gates'],
                    previous_v10_inputs_unchanged=True, old_tables_values_preserved=True,
                    tables=21, figures=4, one_optimization_seed_probe_only=True,
                    journal_ready=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(source=str(destination), label=result['label'], inputs=len(inputs))))


if __name__ == '__main__':
    main()
