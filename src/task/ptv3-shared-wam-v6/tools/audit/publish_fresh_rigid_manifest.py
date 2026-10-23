#!/usr/bin/env python3
"""Bind fresh candidate tuple ordinals to actual audit order and reset sources."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def publish(out):
    read=lambda p:json.loads(p.read_text())
    independent=read(out/'independent-audit.json')
    assert independent['status']=='PASS' and independent['certified_goal_rollouts']==96
    records=[json.loads(line) for line in (out/'rollout-verdicts.jsonl').read_text().splitlines()]
    tuples=np.load(out/'certified-candidate-tuples.npy',allow_pickle=False)
    assert len(records)==96 and len({r['job']['rollout_id'] for r in records})==96
    expected=[];rows=[]
    plan=read(out/'plan.json');assets={r['object_code']:r for r in plan['objects']}
    for ordinal,record in enumerate(records):
        job=record['job'];folder=out/job['rollout_id'];asset=assets[job['object_code']]
        assert record['goal_certification']=='CERTIFIED_LEGAL'
        assert record['candidate_windows']==record['admitted_windows']==274
        expected.extend([ordinal,0,tick] for tick in range(3,277))
        rows.append(dict(sequence_ordinal=ordinal,rollout_id=job['rollout_id'],source_id=4,
            sequence=str(folder/'sequence'),contact=str(folder/'contact'),events=str(folder/'events'),
            geometry=str(out/asset['registry']),geometry_sha256=asset['registry_sha256'],
            reset_binding=str(folder/'reset-binding.json'),goal_instruction=str(folder/'goal-channel/instruction.json'),
            admission=str(folder/'admission.json'),source_role=record['source_role'],
            available_before_reset=record['available_before_reset'],frames=301,fps=30,
            split='unassigned candidate; official TRAIN assets only',goal_certification=record['goal_certification']))
    assert np.array_equal(tuples,np.asarray(expected,dtype=np.int64))
    manifest=dict(schema='ref2dex.fresh-rigid-candidates.v1',run_id=out.name,
        training_allowed=False,production_recipe_modified=False,source_id=4,
        tuple_columns=['sequence_ordinal','object_slot','tick30'],
        ordinal_scope='row order in rollout-verdicts.jsonl; NOT plan.jobs order or original6k sequence IDs',
        tuple_index=str(out/'certified-candidate-tuples.npy'),candidate_windows=len(tuples),sequences=rows,
        goal_consumer='ptv3_shared_wam_v6.fresh_rigid.reset_bound_sample',
        goal_policy='reset-declared task_completion pose; never default sample_window own-future endpoint',
        event_policy='offline proximity/event labels only; no deployment planned-event certificate',
        certification_receipt=str(out/'independent-audit.json'))
    if shutil.disk_usage(out).free<30*2**30:raise RuntimeError('30GiB free-space floor')
    with (out/'candidate-manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PASS',sequences=len(rows),candidate_windows=len(tuples),tuple_binding='exact audit row order')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    publish(p.parse_args().output.resolve())
