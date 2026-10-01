"""Durable per-task continuation policy; execution state stays in the backend."""
from __future__ import annotations

import json
import time
from typing import Any

from scripts.workflow_bindings import protect_workspace, WorkspaceConflict
from scripts.workflow_policy import worker_prompt
from scripts.workflow_runtime import provider_failure, WorkflowError, NativeGoalHeld, ProviderWaiting


def save(db: Any, ident: str, value: dict[str, Any]) -> None:
    db.execute('INSERT OR REPLACE INTO recoveries VALUES (?,?)', (ident, json.dumps(value)))


def attach(workflow: Any, db: Any, item: dict[str, Any], recovery: dict[str, Any], execution_id: str) -> None:
    db.execute('UPDATE deliveries SET execution_id=?, store=COALESCE(?,store) WHERE id=?',
               (execution_id, recovery.get('binding_store'), item['task_id']))
    db.execute('UPDATE execution_attempts SET execution_id=? WHERE task_id=? AND attempt=?',
               (execution_id, item['task_id'], recovery['attempt']))
    recovery.update(phase='observing', source_execution_id=execution_id)
    if item['role'] == 'root' and recovery.get('observation_fingerprint'):
        contract = {**item['contract'], 'observation_fingerprint': recovery['observation_fingerprint']}
        db.execute('UPDATE deliveries SET contract=? WHERE id=?', (json.dumps(contract, sort_keys=True), item['task_id']))
    save(db, item['task_id'], recovery)
    workflow.set(db, 'idle_fingerprint', None)


def recover(workflow: Any, status: dict[str, Any]) -> None:
    for item in status['deliveries']:
        if item['acceptance'] is not None or (item['role'] == 'root' and item['applied']):
            continue
        with workflow.transaction() as db:
            row = db.execute('SELECT value FROM recoveries WHERE task_id=?', (item['task_id'],)).fetchone()
        recovery = json.loads(row['value']) if row else None
        runtime = workflow.runtime(item['role'], (recovery or {}).get('binding_store', item['binding_store']))
        if recovery and recovery['phase'] in ('launching', 'uncertain'):
            found = runtime.find(recovery['name'])
            with workflow.transaction() as db:
                if len(found) == 1:
                    attach(workflow, db, item, recovery, found[0])
                else:
                    recovery['phase'] = 'uncertain'
                    save(db, item['task_id'], recovery)
            continue
        if status['mode'] != 'active':
            continue
        execution = item['execution']
        failure = provider_failure(execution)
        if execution.get('active') is True or execution.get('status') == 'unknown':
            continue
        if recovery and recovery['phase'] == 'stopped':
            continue
        if failure not in ('capacity', 'connection', 'usage_limit'):
            if recovery and execution.get('status') == 'succeeded':
                recovery.update(phase='finished', gateway_attempts=0)
                with workflow.transaction() as db:
                    save(db, item['task_id'], recovery)
            continue
        version = status['control']['instruction_version']
        if item['goal_version'] != status['control']['goal_version'] or (
                item['role'] == 'root' and item['contract'].get('instruction_version') != version):
            recovery = recovery or {'task_id': item['task_id'], 'source_execution_id': item['execution_id'],
                                    'kind': failure, 'due': None, 'attempt': 0, 'gateway_attempts': 0}
            recovery.update(phase='stopped', reason='superseded control or goal')
            with workflow.transaction() as db:
                save(db, item['task_id'], recovery)
            continue
        if recovery is None or recovery['source_execution_id'] != item['execution_id'] or recovery['phase'] != 'waiting':
            previous = recovery or {}
            # Capacity is a wait, not a gateway attempt. Success clears the streak.
            recovery = {'task_id': item['task_id'], 'source_execution_id': item['execution_id'],
                        'kind': failure, 'phase': 'waiting', 'due': time.time() + (60 if failure == 'capacity' else workflow.config.get('recovery_backoff_seconds', 1)),
                        'attempt': previous.get('attempt', 0), 'gateway_attempts': previous.get('gateway_attempts', 0)}
            if failure == 'connection' and recovery['gateway_attempts'] >= 3:
                recovery.update(phase='stopped', reason='three gateway recovery attempts exhausted')
            with workflow.transaction() as db:
                workflow.check_control(db, version)
                db.execute('INSERT OR IGNORE INTO execution_attempts(task_id,attempt,execution_id,name,store) VALUES (?,?,?,?,?)',
                           (item['task_id'], 0, item['execution_id'], item['task_id'], item['binding_store']))
                save(db, item['task_id'], recovery)
            continue
        if time.time() < recovery['due']:
            continue
        if item['role'] != 'root' and workflow.gpu_usage(item['task_id']) + item['contract'].get('gpu', 0) > workflow.budget['max_gpu']:
            recovery['reason'] = 'waiting for authorized GPU capacity'
            with workflow.transaction() as db:
                save(db, item['task_id'], recovery)
            continue
        if failure == 'usage_limit':
            try:
                binding = workflow.select_binding(item['role'], item['contract'].get('provider'), item['task_id'], item['contract'].get('kind'))
            except ProviderWaiting:
                recovery['reason'] = 'waiting for a permitted verified account'
                with workflow.transaction() as db:
                    save(db, item['task_id'], recovery)
                continue
            runtime = workflow.runtime(item['role'], binding['store'])
        try:
            protect_workspace(runtime.binding)
        except WorkspaceConflict as exc:
            recovery.update(phase='stopped', reason=str(exc))
            with workflow.transaction() as db:
                save(db, item['task_id'], recovery)
            continue
        prompt = workflow.root_prompt(status) if item['role'] == 'root' else worker_prompt(item['contract'], workflow.roles[item['role']])
        prompt += '\nContinue the SAME authorized task after a provider interruption. Inspect saved artifacts and owned processes first; do not duplicate experiments or rerun completed work. Preserve existing changes. Do not unpause a user-paused Goal or reset its budget.'
        try:
            source_runtime = workflow.runtime(item['role'], item['binding_store'])
            execution = source_runtime.continuation_source(execution)
            if runtime.store != source_runtime.store:
                # Cross-account handoff must never resume an old account's thread.
                prompt += '\nAccount handoff. Previous execution: ' + json.dumps(item['execution'], ensure_ascii=False)
                execution = {**execution, 'provider': {}}
        except NativeGoalHeld as exc:
            recovery.update(phase='stopped', reason=str(exc))
            with workflow.transaction() as db:
                save(db, item['task_id'], recovery)
            continue
        with workflow.transaction() as db:
            workflow.check_control(db, version)
            workflow.check_budget(db, item['role'])
            timeout = item['contract']['timeout_seconds']
            if 'wall_time_seconds' in workflow.budget:
                timeout = min(timeout, int(workflow.budget['wall_time_seconds'] - (time.time() - workflow.get(db, 'started'))))
            if timeout <= 0:
                raise WorkflowError('no authorized time remains for recovery')
            recovery.update(phase='launching', attempt=recovery['attempt'] + 1, timeout_seconds=timeout, binding_store=runtime.binding['store'])
            if item['role'] == 'root':
                recovery['observation_fingerprint'] = workflow.fingerprint(status)
            if failure == 'connection':
                recovery['gateway_attempts'] += 1
            recovery['name'] = 'ref2dex:' + workflow.get(db, 'workflow_id') + ':' + item['task_id'] + ':recovery-' + str(recovery['attempt'])
            db.execute('INSERT INTO execution_attempts(task_id,attempt,execution_id,name,store) VALUES (?,?,?,?,?)',
                       (item['task_id'], recovery['attempt'], None, recovery['name'], runtime.binding['store']))
            save(db, item['task_id'], recovery)
        # Persist intent before the side effect; a missing response is reconciled
        # by unique name and never causes another continuation to be launched.
        with workflow.transaction() as db:
            if workflow.get(db, 'mode') != 'active' or workflow.get(db, 'instruction_version') != version:
                recovery.update(phase='waiting', attempt=recovery['attempt'] - 1)
                if failure == 'connection':
                    recovery['gateway_attempts'] -= 1
                db.execute('DELETE FROM execution_attempts WHERE task_id=? AND execution_id IS NULL', (item['task_id'],))
                save(db, item['task_id'], recovery)
                continue
            execution_id = runtime.continue_task(execution, recovery['name'], prompt, recovery['timeout_seconds'])
            attach(workflow, db, item, recovery, execution_id)
