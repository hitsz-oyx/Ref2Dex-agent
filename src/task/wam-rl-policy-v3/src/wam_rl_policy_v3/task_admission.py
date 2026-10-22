"""One explicit executable-task admission policy for PPO and native sampling.

Annotations remain immutable. UNKNOWN stays available for corpus inspection,
but has no grasp-task success semantics and is not an executable RL task.
"""
import json
from pathlib import Path
from .task_spec import TaskSpec,TerminalMode

POLICY='ref2dex.executable-task-admission.reject-unknown.v1'


def load_executable_tasks(manifest):
    path=Path(manifest).resolve();audit=json.loads(path.read_text())
    if audit.get('schema')!='ref2dex.teacher-measured-rollout-admission.v2' or audit.get('status')!='COMPLETED':
        raise ValueError('corrected completed corpus required')
    accepted=[];rejected=[];eligible=0;seen=set()
    for row in audit['records']:
        if row['split']!='train' or row['engineering']!='PASS' or row['training_prefix_commands']<=0:continue
        if row['sequence'] in seen:raise ValueError('duplicate eligible sequence: '+row['sequence'])
        seen.add(row['sequence']);eligible+=1
        side=path.parent/row['sidecar'];spec=TaskSpec.from_packet(json.loads((side/'task.json').read_text()))
        if spec.terminal_mode==TerminalMode.UNKNOWN:
            rejected.append(dict(sequence=row['sequence'],terminal_mode=spec.terminal_mode.value,
                reason='UNKNOWN has no defined task success branch',task_path=str(side/'task.json')))
            continue
        spec.require_success_path()
        accepted.append((row,spec,side))
    report=dict(schema=POLICY,source_manifest=str(path),eligible_count=eligible,admitted_count=len(accepted),
        rejected_count=len(rejected),rejected=rejected,admitted_sequences=[r['sequence'] for r,_,_ in accepted],
        annotation_mutated=False,comparison_scope='new admission changes the training distribution; never replace the old run denominator silently')
    return accepted,report


def admitted_native_records(records,manifest):
    """The host and policy must sample exactly the same executable TRAIN set."""
    entries,report=load_executable_tasks(manifest);allowed={r['sequence'] for r,_,_ in entries}
    selected=[row for row in records if row['sequence'] in allowed]
    if {r['sequence'] for r in selected}!=allowed:raise ValueError('native geometry does not cover executable admitted TRAIN')
    return selected,report


def require_native_membership(names,report):
    if not isinstance(names,list) or not names:raise ValueError('native must declare sampled task membership before PPO')
    unexpected=set(names)-set(report['admitted_sequences'])
    if unexpected:
        raise ValueError(f'native samples {len(unexpected)} tasks outside executable admission: {sorted(unexpected)[:8]}')
