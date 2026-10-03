"""Completion evidence checks; control validation does not prove scientific truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.workflow_runtime import WorkflowError


def completion_evidence(status: dict[str, Any], decision: dict[str, Any], workspace: Path) -> dict[str, Any]:
    ids = decision.get('evidence_task_ids')
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        raise WorkflowError('completion requires formal Validation evidence and independent eval review')
    deliveries = {item['task_id']: item for item in status['deliveries'] if item['role'] != 'root'}
    if any(item['acceptance'] is None for item in deliveries.values()):
        raise WorkflowError('completion has unfinished or unadjudicated worker deliveries')
    evidence = []
    for ident in ids:
        item = deliveries.get(ident)
        if not item or item['contract'].get('kind') != 'validation' or item['acceptance'].get('decision') != 'accepted' or item['goal_version'] != status['control']['goal_version']:
            raise WorkflowError('completion requires accepted current-goal formal Validation evidence')
        evidence.append(item)
    reviews = [item for item in evidence if item['role'] == 'agent_eval']
    sources = [item for item in evidence if item['role'] != 'agent_eval']
    if not reviews or not sources:
        raise WorkflowError('completion requires formal Validation and independent eval review')
    try:
        review = json.loads(reviews[-1]['execution']['output'])
        if review['schema'] != 'ref2dex.validation-evidence.v1' or review['goal_version'] != status['control']['goal_version']:
            raise ValueError('review identity mismatch')
        if not all(item['task_id'] in review['reviewed_task_ids'] for item in sources):
            raise ValueError('review does not cover evidence')
        checks = review['checks']
        for objective in ('self_trained_grasp', 'cm_policy_utility'):
            check = checks[objective]
            if check['verdict'] != 'SUPPORTED' or check['pre_registered'] is not True or not str(check['scope']).strip():
                raise ValueError('mission objective is not formally supported')
            seeds = check['seeds']
            if not isinstance(seeds, list) or len(set(seeds)) < 2 or not all(type(seed) is int for seed in seeds):
                raise ValueError('formal review needs multiple seeds')
            card = (workspace / check['card']).resolve()
            card.relative_to((workspace / 'docs/experiments/validations').resolve())
            if not card.is_file() or not check['validation_id'].startswith('VAL-') or check['validation_id'] not in card.read_text():
                raise ValueError('validation card missing or identity mismatch')
        if checks['self_trained_grasp']['self_trained'] is not True:
            raise ValueError('baseline is not self-trained')
        cm = checks['cm_policy_utility']
        if cm['matched_control'] is not True or set(cm['arms']) != {'Cm-on', 'Cm-off'}:
            raise ValueError('Cm comparison is not matched')
    except (KeyError, ValueError, TypeError, AttributeError, OSError) as exc:
        raise WorkflowError('completion review lacks valid mission evidence: ' + str(exc)) from exc
    if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        raise WorkflowError('completion requires an acceptance reason')
    return {'reason': decision['reason'], 'goal_version': status['control']['goal_version'],
            'evidence_task_ids': ids, 'checks': checks}
