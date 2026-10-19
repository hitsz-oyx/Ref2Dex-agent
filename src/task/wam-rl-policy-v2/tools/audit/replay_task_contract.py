"""Replay full archive membership through the same measured task controller.

No teacher reference enters measured feedback. Missing DOF force cannot be
reconstructed here; release support remains unknown. This is an interface audit,
not an evaluation of a new policy or a placement success certificate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v2.task_annotations import dexplore_task_spec,ANNOTATION_SCHEMA
from wam_rl_policy_v2.task_runtime import TaskController
from wam_rl_policy_v2.robot_adapter import InspireAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit report required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.cache/'manifest.json').read_text());rows=[];geometry={};started=time.monotonic()
    membership=meta['sequences'][:a.limit] if a.limit else meta['sequences']
    for row in membership:
        gid=row['geometry']
        if gid not in geometry:
            g=Path(meta['records'][gid]['geometry']);g=g if g.is_absolute() else a.cache/g
            with np.load(g) as z:geometry[gid]=(z['query'].copy(),z['surface'].copy())
        source=Path(row['source']);batch=json.loads((source.parents[1]/'manifest.json').read_text())
        contact_ids=batch['observation_contract']['contact_body_ids']
        with np.load(source/'trajectory.npz') as z:
            spec=dexplore_task_spec(z['reference_obj'],z['reference_current'])
            query,surface=geometry[gid]
            controller=TaskController(spec,query,interaction_points=surface);phases=Counter();transitions=Counter();probes=0
            last=None;steps=0;missing_force='dof_force' not in z.files
            for t in range(len(z['obj'])):
                contact_valid=bool(z['contact_force_valid'][t]) if 'contact_force_valid' in z.files else t>0
                result=controller.observe(z['obj'][t],InspireAdapter.canonical_hand(z['hand'][t]),z['velocity'][t],t/30,
                    contact_forces=z['body_contact_forces'][t,contact_ids],force_valid=contact_valid,supported=None)
                d=result['decision'];last=d;steps+=1;phases[d.phase.name]+=1;transitions.update(d.transitions)
                probes+=int(d.probe_active)
                if result['task'].shape!=(10,) or not np.isfinite(result['task']).all():raise ValueError('invalid task10')
                if d.terminated:break
            rows.append(dict(sequence=row['sequence'],split=row['cache_split'],source=str(source),
                original_frames=spec.source_length,path_frames=len(spec.object_path),
                first_source_frame=int(spec.source_frames[0]),last_source_frame=int(spec.source_frames[-1]),
                events=[dict(frame=e.source_frame,type=e.kind.value) for e in spec.events],terminal_mode=spec.terminal_mode.value,
                observed_frames=steps,phase_frames=dict(phases),transitions=dict(transitions),probe_frames=probes,
                path_complete=last.path_complete,measured_task_complete=last.task_complete,
                failure_reason=last.failure_reason,confirmed_progress=last.path_progress,
                missing_dof_force=missing_force,support_unknown=True))
        if len(rows)%50==0:print(json.dumps(dict(sequences=len(rows),elapsed_s=time.monotonic()-started)),flush=True)
    report=dict(status='COMPLETED',engineering_only=True,annotation_schema=ANNOTATION_SCHEMA,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,
        input=str(a.cache.resolve()),interaction_surface_points=512,elapsed_s=time.monotonic()-started,sequences=len(rows),
        splits=dict(Counter(r['split'] for r in rows)),terminal_modes=dict(Counter(r['terminal_mode'] for r in rows)),
        multiple_contact_sequences=sum(sum(e['type']=='CONTACT' for e in r['events'])>1 for r in rows),
        phase_frames=dict(sum((Counter(r['phase_frames']) for r in rows),Counter())),
        failures=dict(Counter(r['failure_reason'] or 'archive_end_or_success' for r in rows)),
        terminated=sum(r['failure_reason'] is not None or r['measured_task_complete'] for r in rows),
        task_complete=sum(r['measured_task_complete'] for r in rows),
        missing_dof_force=sum(r['missing_dof_force'] for r in rows),rows=rows,
        limits='Known event annotations are intent proxies; no support certificate, new policy, learned shape feature or WAM event-token evaluation. Full raw sequences replayed; old task-active mask ignored.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
