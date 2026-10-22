"""Rebuild all 501 complete declarations, separating raw invalidity and dropout."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.task_annotations import task_from_teacher_archive,task_with_safe_prefix,planned_contact_events
from wam_rl_policy_v3.task_spec import TaskSpec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    base=ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1';manifest=json.loads((base/'manifest.json').read_text())
    legacy=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3';rows=[]
    for row in manifest['records']:
        side=base/row['sidecar'];source=Path(row['source'])
        raw=TaskSpec.from_packet(json.loads((side/'raw-task.json').read_text()))
        saved=TaskSpec.from_packet(json.loads((side/'task.json').read_text()))
        old=TaskSpec.from_packet(json.loads((legacy/'sequences'/row['sequence']/'task.json').read_text()))
        receipt=json.loads((source.parents[1]/'manifest.json').read_text())
        with np.load(source/'canonical.npz') as z:points=z['points'].copy()
        with np.load(source/'trajectory.npz') as z:
            data={k:z[k].copy() for k in ('obj','hand','frame_tick','body_contact_forces','contact_force_valid','reference_current')}
        rebuilt=task_from_teacher_archive(data,receipt['observation_contract']['contact_body_ids'],object_points=points)
        cuts=[0,int(row['training_prefix_commands']),len(data['obj'])-1]
        unchanged=all(task_with_safe_prefix(rebuilt,int(cut))[0].packet()==rebuilt.packet() for cut in cuts)
        first_contact=next((e.source_frame for e in raw.events if e.kind.value=='CONTACT'),None)
        ref_events,ref_mode=planned_contact_events(data['reference_current'])
        explanation=None
        if old.terminal_mode.value=='UNKNOWN':
            explanation=('prefix ends before first complete CONTACT interval' if first_contact is not None and row['training_prefix_commands']<first_contact+2
                else 'truncated contact evidence fails the original three-frame/merge interval rule; full events retained below')
        rows.append(dict(sequence=row['sequence'],complete_source_length=len(data['obj']),prefix_commands=row['training_prefix_commands'],
            old_terminal_mode=old.terminal_mode.value,complete_terminal_mode=raw.terminal_mode.value,rebuilt_terminal_mode=rebuilt.terminal_mode.value,
            complete_task_equals_saved=raw.packet()==saved.packet(),complete_recording_rebuild_equals_raw=rebuilt.packet()==raw.packet(),
            prefix_does_not_change_task=unchanged,annotation_invalid=rebuilt.annotation_invalid,raw_invalid_reasons=list(rebuilt.annotation_invalid_reasons),
            raw_force_missing_after_reset=int((~data['contact_force_valid'][1:]).sum()),active_dropout_applied=False,
            source_events=[dict(kind=e.kind.value,source_frame=e.source_frame,seconds=e.source_frame/raw.source_fps) for e in raw.events],
            legacy_prefix_events=[dict(kind=e.kind.value,source_frame=e.source_frame) for e in old.events],
            old_unknown_explanation=explanation,reference_terminal_mode=ref_mode.value,raw_reference_terminal_conflict=raw.terminal_mode!=ref_mode,
            terminal_intent_independently_verified=False))
    old_unknown=[r for r in rows if r['old_terminal_mode']=='UNKNOWN']
    report=dict(status='PASS' if len(rows)==501 and len(old_unknown)==64 and all(r['complete_task_equals_saved'] and r['complete_recording_rebuild_equals_raw'] and r['prefix_does_not_change_task'] for r in rows) else 'FAIL',
        records=rows,old_unknown_reclassifications=old_unknown,summary=dict(records=len(rows),old_unknown=len(old_unknown),
            old_unknown_to_release=sum(r['complete_terminal_mode']=='RELEASE' for r in old_unknown),old_unknown_to_hold=sum(r['complete_terminal_mode']=='HOLD' for r in old_unknown),
            annotation_invalid=sum(r['annotation_invalid'] for r in rows),defined_declarations=sum(not r['annotation_invalid'] for r in rows),
            raw_reference_terminal_conflicts=sum(r['raw_reference_terminal_conflict'] for r in rows),
            independently_certified_full_task_positive_controls=0),
        scope='raw measurement validity is not a certificate of planned intent or executable full-task quality; no samples deleted, no new full-task rate claimed')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],summary=report['summary'])))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
