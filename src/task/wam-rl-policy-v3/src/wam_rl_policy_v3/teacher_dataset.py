"""Audited measured episodes for the force/shape/task10 Inspire executor."""
import json
from pathlib import Path
import numpy as np
from .inspire_shadow import ACTIVE,current_state,measured_from_legacy,object_state
from .task_spec import TaskSpec
from .task_annotations import GOAL_SOURCE,task_with_safe_prefix
from .task_runtime import TaskController
from .robot_adapter import InspireAdapter
from .teacher_quality import QUALITY_SCHEMA,TABLE_SUPPORT_CONTRACT


class TeacherRolloutCorpus:
    def __init__(self,audit_manifest,split):
        self.manifest_path=Path(audit_manifest);self.root=self.manifest_path.parent
        self.manifest=json.loads(self.manifest_path.read_text())
        if (self.manifest.get('schema')!=QUALITY_SCHEMA or self.manifest.get('status')!='COMPLETED'
                or not self.manifest.get('training_allowed') or self.manifest.get('goal_source')!=GOAL_SOURCE
                or self.manifest.get('table_support_contract')!=TABLE_SUPPORT_CONTRACT):
            raise ValueError('complete measured teacher-rollout admission with corrected table support required')
        if split not in ('train','dev'):raise ValueError('TRAIN/DEV split required')
        self.records=[r for r in self.manifest['records'] if ('dev' if r['split']=='val' else r['split'])==split
            and r['engineering']=='PASS' and r['training_prefix_commands']>0]

    def __len__(self):return len(self.records)

    def episode(self,index):
        row=self.records[index];source=Path(row['source']);n=row['training_prefix_commands'];sidecar=self.root/row['sidecar']
        spec=TaskSpec.from_packet(json.loads((sidecar/'task.json').read_text()))
        if spec.source_role!=GOAL_SOURCE:raise ValueError('same-execution declared task required')
        raw_path=sidecar/'raw-task.json'
        if raw_path.exists():
            declared=TaskSpec.from_packet(json.loads(raw_path.read_text()))
            if spec.packet()!=declared.packet():raise ValueError('task differs from complete raw-task declaration; migrate truncated prefix sidecars')
        elif spec.source_frames[-1]!=spec.source_length-1:
            raise ValueError('complete declared raw-task required for a truncated or censored task')
        metadata_path=sidecar/'task-availability.json'
        stored=json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        _,availability=task_with_safe_prefix(spec,n,unsafe_future=stored.get('unsafe_future') if stored else None)
        if stored is not None and stored!=availability:raise ValueError('declared task/prefix availability metadata mismatch')
        receipt=json.loads((source.parents[1]/'manifest.json').read_text());ids=receipt['observation_contract']['contact_body_ids']
        with np.load(source/'canonical.npz') as geometry:points=geometry['points'].copy()
        with np.load(sidecar/'admission.npz') as masks:support=masks['supported'][:n].copy()
        controller=TaskController(spec,points)
        with np.load(source/'trajectory.npz') as z:
            if len(z['obj'])!=spec.source_length:raise ValueError('declared task source_length differs from complete recorded source')
            ticks=z['frame_tick'][:n].copy()
            if ticks.shape!=(n,) or not np.array_equal(ticks,np.arange(n)):
                raise ValueError('same-execution consecutive native ticks required')
            action=z['action'][:n,ACTIVE].copy();previous=np.zeros_like(action);previous[1:]=action[:-1]
            base=current_state(z['q'][:n],z['dq'][:n],measured_from_legacy(z['native_current_obs'][:n]),
                object_state(z['obj'][:n],z['velocity'][:n]),np.broadcast_to(spec.object_path[-1,:3,3],(n,3)),previous)
            dof_valid=z['dof_force_valid'][:n];contact_valid=z['contact_force_valid'][:n]
            force=np.where(dof_valid[:,None],z['dof_force'][:n],0)
            contact=np.where(contact_valid[:,None,None],z['body_contact_forces'][:n,ids],0)
            measured=np.concatenate((base,force,contact.reshape(n,15),dof_valid[:,None],contact_valid[:,None]),-1).astype(np.float32)
            task=np.zeros((n,10),np.float32);active=np.zeros(n,bool)
            local_goals=np.zeros((n,4,4),np.float64)
            for t in range(n):
                result=controller.observe(z['obj'][t],InspireAdapter.canonical_hand(z['hand'][t]),z['velocity'][t],t/30,
                    contact_forces=contact[t],force_valid=bool(contact_valid[t]),supported=bool(support[t]))
                task[t]=result['task'];active[t]=not result['decision'].terminated
                local_goals[t]=result['decision'].local_goal_pose
                if result['decision'].terminated:break
        if measured.shape!=(n,347) or not np.isfinite(measured).all():raise ValueError('finite measured347 required')
        return dict(sequence=row['sequence'],measured=measured,task=task,active=active,action=action,
            object_points=points.astype(np.float32),task_spec=spec.packet(),goal_source=GOAL_SOURCE,
            local_goal_pose=local_goals,ticks=ticks,task_prefix_availability=availability,
            task_censored=availability['task_censored'],complete_positive_eligible=False)
