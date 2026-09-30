"""Durable research intent and delivery acceptance around an external executor."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workflow_policy import root_prompt as build_root_prompt, worker_prompt
from scripts.workflow_bindings import validate_bindings, protect_workspace, WorkspaceConflict
from scripts.research_evidence import completion_evidence
from scripts.workflow_recovery import recover
from scripts.workflow_runtime import Runtime, TERMINAL, WorkflowError, BudgetError, ResearchPaused, ProviderWaiting, provider_failure


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Workflow:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text())
        self.directory = Path(self.config['state_dir']).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.bindings = self.config['bindings']
        document = yaml.safe_load(Path(self.config['roles_file']).read_text())
        self.roles = {item['agent_key']: item for item in document['roles']}
        self.budget = self.config['campaign']
        for key in ('max_dispatches', 'max_root_turns', 'wall_time_seconds'):
            if key in self.budget and (type(self.budget[key]) is not int or self.budget[key] <= 0):
                raise WorkflowError('explicit campaign limits must be positive integers')
        if type(self.budget.get('max_gpu')) is not int or not 0 <= self.budget['max_gpu'] <= 4:
            raise WorkflowError('campaign max_gpu must be between zero and four')
        attempts = self.config.get('max_recovery_attempts', 3)
        if type(attempts) is not int or not 1 <= attempts <= 10:
            raise WorkflowError('recovery attempts must be between one and ten')
        for key, default in (('poll_seconds', 2), ('recovery_backoff_seconds', 1)):
            value = self.config.get(key, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 60:
                raise WorkflowError('poll and recovery intervals must be positive and at most sixty seconds')
        command = self.config.get('command')
        if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
            raise WorkflowError('command must be an argv array')
        if not Path(self.config['state_dir']).is_absolute() or not Path(self.config['roles_file']).is_absolute():
            raise WorkflowError('state_dir and roles_file must be absolute')
        if 'root' not in self.bindings or self.bindings['root'].get('runtime') != 'codex':
            raise WorkflowError('root must use the Codex runtime')
        self.candidates = validate_bindings(self.config, self.roles, self.budget)
        self.db_path = self.directory / 'research.sqlite'
        with self.transaction() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS deliveries (
                  id TEXT PRIMARY KEY, role TEXT NOT NULL, contract TEXT NOT NULL,
                  name TEXT NOT NULL UNIQUE, execution_id TEXT, acceptance TEXT, applied INTEGER NOT NULL DEFAULT 0);
            ''')
            db.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, goal_version INTEGER NOT NULL, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS instructions (request_id TEXT PRIMARY KEY, version INTEGER NOT NULL UNIQUE, text TEXT NOT NULL, new_goal INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS recoveries (task_id TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS execution_attempts (task_id TEXT NOT NULL, attempt INTEGER NOT NULL, execution_id TEXT, name TEXT NOT NULL, PRIMARY KEY(task_id,attempt))')
            columns = {row[1] for row in db.execute('PRAGMA table_info(deliveries)')}
            if 'store' not in columns:
                db.execute('ALTER TABLE deliveries ADD COLUMN store TEXT')
            if 'goal_version' not in columns:
                db.execute('ALTER TABLE deliveries ADD COLUMN goal_version INTEGER NOT NULL DEFAULT 0')
            defaults: dict[str, Any] = {'shutdown_requested': False, 'instruction_version': 0, 'processed_version': 0, 'goal_version': 0, 'completion': None, 'blocked': None, 'blocked_fingerprint': None, 'active_bindings': {}, 'provider_waits': {}, 'provider_ready_after': {}, 'mode': 'paused', 'workflow_id': uuid.uuid4().hex, 'started': None,
                        'failures': 0, 'retry_at': 0, 'idle_fingerprint': None, 'pid': None, 'last_error': None}
            for key, value in defaults.items():
                db.execute('INSERT OR IGNORE INTO control VALUES (?, ?)', (key, encoded(value)))

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=40)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def get(self, db: sqlite3.Connection, key: str) -> Any:
        return json.loads(db.execute('SELECT value FROM control WHERE key=?', (key,)).fetchone()[0])

    def set(self, db: sqlite3.Connection, key: str, value: Any) -> None:
        db.execute('UPDATE control SET value=? WHERE key=?', (encoded(value), key))

    def runtime(self, role: str, store: str | None = None) -> Runtime:
        candidates = self.candidates[role]
        if store is None:
            with self.transaction() as db:
                store = self.get(db, 'active_bindings').get(role, candidates[0]['store'])
        for binding in candidates:
            if binding['store'] == store:
                return Runtime(self.config, binding)
        raise WorkflowError('historical execution binding is no longer configured')

    def select_binding(self, role: str, provider: str | None, ident: str, kind: str | None = None) -> dict[str, Any]:
        with self.transaction() as db:
            active = self.get(db, 'active_bindings').get(role, self.candidates[role][0]['store'])
            history = [dict(row) for row in db.execute('SELECT rowid AS sequence, * FROM deliveries WHERE role=? AND execution_id IS NOT NULL ORDER BY rowid DESC', (role,))]
            waits = self.get(db, 'provider_waits')
            ready_after = self.get(db, 'provider_ready_after')
        eligible = [item for item in self.candidates[role] if provider is None or item['provider'] == provider]
        if not eligible:
            raise WorkflowError('requested provider is not a configured role binding')
        if provider is None:
            eligible.sort(key=lambda item: item['store'] != active)
        for binding in eligible:
            if kind in ('probe', 'validation') and binding.get('engineering_only'):
                if provider is not None:
                    raise WorkflowError('engineering-only fallback cannot run research experiments')
                continue
            latest = next((row for row in history if (row['store'] or self.candidates[role][0]['store']) == binding['store'] and row['sequence'] > ready_after.get(binding['store'], 0)), None)
            observations = []
            if latest:
                try:
                    execution = self.runtime(role, binding['store']).read(latest['execution_id'])
                except WorkflowError:
                    if not latest['acceptance']:
                        raise
                    execution = {'status': 'unknown'}
                failure = provider_failure(execution)
                if failure == 'unavailable':
                    observations.append(latest['sequence'])
            with self.transaction() as db:
                # A readiness notification during the read invalidates older observations.
                waits = self.get(db, 'provider_waits')
                ready_after = self.get(db, 'provider_ready_after')
                if observations and min(observations) > ready_after.get(binding['store'], 0):
                    waits[binding['store']] = {'role': role, 'provider': binding['provider'], 'reason': 'provider unavailable or quota exhausted'}
                    self.set(db, 'provider_waits', waits)
                if binding['store'] in waits:
                    continue
                selected = self.get(db, 'active_bindings')
                if binding['store'] != active:
                    self.save_decision(db, 'provider-' + ident, {'action': 'provider_switch', 'role': role,
                        'from_store': active, 'provider': binding['provider'], 'reason': 'configured primary unavailable'})
                selected[role] = binding['store']
                self.set(db, 'active_bindings', selected)
            return binding
        raise ProviderWaiting('no available permitted provider for ' + role)

    def provider_ready(self, role: str, provider: str) -> dict[str, Any]:
        binding = next((x for x in self.candidates.get(role, []) if x['provider'] == provider), None)
        if binding is None:
            raise WorkflowError('unknown provider binding')
        with self.transaction() as db:
            waits = self.get(db, 'provider_waits')
            waits.pop(binding['store'], None)
            self.set(db, 'provider_waits', waits)
            # A readiness notification invalidates earlier failed observations.
            ready = self.get(db, 'provider_ready_after')
            ready[binding['store']] = db.execute('SELECT COALESCE(MAX(rowid),0) FROM deliveries').fetchone()[0]
            db.execute('INSERT OR REPLACE INTO control VALUES (?,?)', ('provider_ready_after', encoded(ready)))
            self.set(db, 'idle_fingerprint', None)
        return {'role': role, 'provider': provider, 'ready': True}


    def control(self, mode: str, legacy_disabled: bool = False) -> dict[str, Any]:
        with self.transaction() as db:
            if mode == 'active':
                if self.get(db, 'completion') is not None:
                    raise WorkflowError('completed goal requires an explicit new-goal instruction')
                if not legacy_disabled:
                    raise WorkflowError('resume requires confirmation that legacy dispatch owners are disabled')
                # A resume acknowledges prior failures, but does not reset spending.
                if self.get(db, 'started') is None:
                    self.set(db, 'started', time.time())
                self.set(db, 'failures', 0)
                self.set(db, 'retry_at', 0)
                self.set(db, 'shutdown_requested', False)
                self.set(db, 'idle_fingerprint', None)
            self.set(db, 'instruction_version', self.get(db, 'instruction_version') + 1)
            self.set(db, 'mode', mode)
        return {'mode': mode}

    def instruct(self, request_id: str, text: str, new_goal: bool = False) -> dict[str, Any]:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', request_id) or not text.strip() or len(text) > 64000:
            raise WorkflowError('instruction requires a stable request ID and bounded nonempty text')
        with self.transaction() as db:
            old = db.execute('SELECT * FROM instructions WHERE request_id=?', (request_id,)).fetchone()
            if old:
                if old['text'] != text or bool(old['new_goal']) != new_goal:
                    raise WorkflowError('request ID already belongs to another instruction')
                return {'version': old['version'], 'reused': True}
            version = self.get(db, 'instruction_version') + 1
            db.execute('INSERT INTO instructions VALUES (?,?,?,?)', (request_id, version, text, int(new_goal)))
            self.set(db, 'instruction_version', version)
            self.set(db, 'idle_fingerprint', None)
            if new_goal:
                self.set(db, 'goal_version', self.get(db, 'goal_version') + 1)
                self.set(db, 'completion', None)
                if self.get(db, 'mode') == 'completed':
                    self.set(db, 'mode', 'paused')
            if self.get(db, 'mode') == 'blocked':
                self.set(db, 'mode', 'active')
                self.set(db, 'blocked', None)
        return {'version': version, 'reused': False}

    def check_control(self, db: sqlite3.Connection, expected_version: int | None) -> None:
        if self.get(db, 'mode') != 'active' or (expected_version is not None and self.get(db, 'instruction_version') != expected_version):
            raise ResearchPaused('control changed before applying decision')

    def save_decision(self, db: sqlite3.Connection, ident: str, value: dict[str, Any], goal_version: int | None = None) -> None:
        goal = self.get(db, 'goal_version') if goal_version is None else goal_version
        db.execute('INSERT INTO decisions VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value WHERE decisions.goal_version=excluded.goal_version',
                   (ident, goal, encoded(value)))

    def decision_record(self) -> tuple[list[dict[str, Any]], str]:
        with self.transaction() as db:
            current_goal = self.get(db, 'goal_version')
            mode = self.get(db, 'mode')
            scope = db.execute('SELECT text FROM instructions WHERE new_goal=1 ORDER BY version DESC LIMIT 1').fetchone()
            scope_text = scope['text'] if scope else 'docs/MISSION.md（当前已授权范围）'
            decisions = [{'id': row['id'], 'goal_version': row['goal_version'], **json.loads(row['value'])}
                         for row in db.execute('SELECT * FROM decisions ORDER BY rowid')]
        path = self.directory / 'AUTONOMOUS_DECISIONS.md'
        with (self.directory / 'decision-record.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            goals = {item['goal_version'] for item in decisions} | {current_goal}
            for goal in sorted(goals):
                lines = ['# 自主决策记录', '', '目标版本：' + str(goal), '',
                         *(['推进状态：' + mode, '', '研究范围：' + scope_text, ''] if goal == current_goal else []),
                         '重要研究选择；实验细节以引用证据为准。', '']
                for item in decisions:
                    if item['goal_version'] == goal:
                        lines.extend(['## ' + item['id'], '', '```json', json.dumps(item, ensure_ascii=False, indent=2), '```', ''])
                content = '\n'.join(lines)
                output = path if goal == current_goal else self.directory / 'decision-history' / ('goal-' + str(goal) + '.md')
                output.parent.mkdir(exist_ok=True)
                if not output.exists() or output.read_text() != content:
                    temporary = output.with_suffix('.tmp')
                    temporary.write_text(content)
                    temporary.replace(output)
                if goal == current_goal and mode == 'completed':
                    archive = self.directory / 'decision-history' / ('completed-goal-' + str(goal) + '.md')
                    archive.parent.mkdir(exist_ok=True)
                    if not archive.exists():
                        archive.write_text(content)
        return decisions, str(path)

    def check_budget(self, db: sqlite3.Connection, role: str) -> None:
        if self.get(db, 'mode') != 'active':
            raise ResearchPaused('research is paused or stopped')
        started = self.get(db, 'started')
        if started is None or ('wall_time_seconds' in self.budget and time.time() - started >= self.budget['wall_time_seconds']):
            raise BudgetError('campaign time budget exhausted')
        count = db.execute('SELECT COUNT(*) FROM deliveries WHERE ' +
                           ('role="root"' if role == 'root' else 'role!="root"')).fetchone()[0]
        count += db.execute('SELECT COUNT(*) FROM execution_attempts a JOIN deliveries d ON a.task_id=d.id WHERE a.attempt>0 AND ' +
                            ('d.role="root"' if role == 'root' else 'd.role!="root"')).fetchone()[0]
        limit = self.budget.get('max_root_turns' if role == 'root' else 'max_dispatches')
        if limit is not None and count >= limit:
            raise BudgetError('campaign dispatch budget exhausted')

    def validate_contract(self, task: dict[str, Any]) -> None:
        if not isinstance(task, dict):
            raise WorkflowError('task contract must be an object')
        ident = task.get('task_id')
        if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', ident):
            raise WorkflowError('invalid task_id')
        role = task.get('role')
        if role not in self.bindings or role == 'root':
            raise WorkflowError('task must target a bound worker role')
        for key in ('objective', 'decision_test'):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise WorkflowError('task requires objective and decision_test')
        for key in ('stop_conditions', 'deliverables'):
            if not isinstance(task.get(key), list) or not task[key] or any(
                    not isinstance(item, str) or not item.strip() for item in task[key]):
                raise WorkflowError('task requires stop conditions and deliverables')
        kind = task.get('kind')
        if kind not in ('engineering', 'probe', 'validation'):
            raise WorkflowError('task kind must be engineering, probe or validation')
        if kind != 'engineering' and self.bindings[role].get('engineering_only'):
            raise WorkflowError('engineering-only runtime cannot run research experiments')
        if kind != 'engineering' and self.roles[role]['resources'].get('experiments') is False:
            raise WorkflowError('logical role is not authorized to run experiments')
        gpu = task.get('gpu')
        permission = self.roles[role]['resources']['gpu']
        if type(gpu) is not int or gpu < 0 or gpu > self.budget['max_gpu'] or (permission == 0 and gpu):
            raise WorkflowError('task GPU request exceeds role or campaign permission')
        timeout = task.get('timeout_seconds')
        if type(timeout) is not int or timeout <= 0 or ('wall_time_seconds' in self.budget and timeout > self.budget['wall_time_seconds']):
            raise WorkflowError('task requires a bounded timeout within any explicit campaign budget')

    def gpu_usage(self, excluded_task: str) -> int:
        with self.transaction() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM deliveries WHERE role!="root" AND acceptance IS NULL AND id!=?', (excluded_task,))]
            recoveries = {row['task_id']: json.loads(row['value']) for row in db.execute('SELECT * FROM recoveries')}
        used = 0
        for row in rows:
            recovery = recoveries.get(row['id']) or {}
            if row['execution_id'] is None:
                raise WorkflowError('unresolved launch intent blocks new dispatch')
            execution = self.runtime(row['role'], row['store'] or self.candidates[row['role']][0]['store']).read(row['execution_id'])
            if recovery.get('phase') in ('launching', 'uncertain') or execution.get('active') is True or execution.get('status') not in TERMINAL:
                used += json.loads(row['contract']).get('gpu', 0)
        return used

    def dispatch(self, task: dict[str, Any], *, root_prompt: str | None = None, expected_version: int | None = None) -> dict[str, Any]:
        if root_prompt is None:
            self.validate_contract(task)
        role = task['role']
        ident = task['task_id']
        with self.transaction() as db:
            existing = db.execute('SELECT * FROM deliveries WHERE id=?', (ident,)).fetchone()
            if existing and not existing['execution_id']:
                raise WorkflowError('unresolved launch intent; reconcile instead of redispatch')
            if existing and existing['contract'] == encoded(task) and existing['execution_id']:
                return {'task_id': ident, 'execution_id': existing['execution_id'], 'reused': True}
        binding = self.select_binding(role, task.get('provider'), ident, task.get('kind'))
        if root_prompt is None and task['kind'] != 'engineering' and binding.get('engineering_only'):
            raise WorkflowError('engineering-only fallback cannot run research experiments')
        protect_workspace(binding)
        used_gpu = self.gpu_usage(ident) if role != 'root' else 0
        # Hold pause/dispatch serialization through the bounded launch call. A
        # pause returning to the user guarantees no subsequent launch races it.
        with self.transaction() as db:
            old = db.execute('SELECT * FROM deliveries WHERE id=?', (ident,)).fetchone()
            if old:
                if old['contract'] != encoded(task):
                    raise WorkflowError('task_id already belongs to a different contract')
                if old['execution_id']:
                    return {'task_id': ident, 'execution_id': old['execution_id'], 'reused': True}
                raise WorkflowError('unresolved launch intent; reconcile instead of redispatch')
            self.check_control(db, expected_version)
            self.check_budget(db, role)
            if role != 'root':
                busy = db.execute('SELECT id FROM deliveries WHERE role=? AND acceptance IS NULL', (role,)).fetchone()
                if busy:
                    raise WorkflowError('role has an unaccepted or unfinished delivery')
                if used_gpu + task.get('gpu', 0) > self.budget['max_gpu']:
                    raise WorkflowError('concurrent GPU budget exceeded')
            name = 'ref2dex:' + self.get(db, 'workflow_id') + ':' + ident
            db.execute('INSERT INTO deliveries(id,role,contract,name,goal_version,store) VALUES (?,?,?,?,?,?)',
                       (ident, role, encoded(task), name, self.get(db, 'goal_version'), binding['store']))
        # Intent is durable BEFORE the external side effect. A separate dispatch
        # gate keeps control changes serialized without rolling back the intent.
        with self.transaction() as db:
            if self.get(db, 'mode') != 'active' or (expected_version is not None and self.get(db, 'instruction_version') != expected_version):
                # No call was made. Removing this intent is safe.
                db.execute('DELETE FROM deliveries WHERE id=? AND execution_id IS NULL', (ident,))
                db.commit()  # Keep the known-no-launch removal despite the control exception.
                raise ResearchPaused('research paused before launch')
            prompt = root_prompt or worker_prompt(task, self.roles[role])
            execution_id = Runtime(self.config, binding).launch(name, prompt, task['timeout_seconds'])
            db.execute('UPDATE deliveries SET execution_id=? WHERE id=?', (execution_id, ident))
        return {'task_id': ident, 'execution_id': execution_id}

    def reconcile(self) -> dict[str, Any]:
        unresolved = []
        with self.transaction() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM deliveries WHERE execution_id IS NULL')]
        found_ids = []
        for row in rows:
            found = self.runtime(row['role'], row['store'] or self.candidates[row['role']][0]['store']).find(row['name'])
            if len(found) == 1:
                found_ids.append((found[0], row['id']))
            else:
                unresolved.append(row['id'])
        with self.transaction() as db:
            for execution_id, ident in found_ids:
                db.execute('UPDATE deliveries SET execution_id=? WHERE id=? AND execution_id IS NULL',
                           (execution_id, ident))
            if unresolved and self.get(db, 'mode') != 'paused':
                self.set(db, 'mode', 'attention')
        return {'unresolved': unresolved}

    def status(self) -> dict[str, Any]:
        with self.transaction() as db:
            control = {row['key']: json.loads(row['value']) for row in db.execute('SELECT * FROM control')}
            rows = [dict(row) for row in db.execute('SELECT * FROM deliveries ORDER BY rowid')]
            instructions = [dict(row) for row in db.execute('SELECT * FROM instructions ORDER BY version')]
            recoveries = [json.loads(row['value']) for row in db.execute('SELECT value FROM recoveries')]
            attempts = [dict(row) for row in db.execute('SELECT * FROM execution_attempts ORDER BY attempt')]
        deliveries = []
        for row in rows:
            execution: dict[str, Any] = {'status': 'unknown'}
            if row['execution_id']:
                try:
                    execution = self.runtime(row['role'], row['store'] or self.candidates[row['role']][0]['store']).read(row['execution_id'])
                except WorkflowError as exc:
                    execution = {'status': 'unknown', 'error': str(exc)}
            deliveries.append({'task_id': row['id'], 'role': row['role'],
                               'contract': json.loads(row['contract']), 'binding_store': row['store'], 'execution_id': row['execution_id'],
                               'acceptance': json.loads(row['acceptance']) if row['acceptance'] else None,
                               'attempts': [item for item in attempts if item['task_id'] == row['id']],
                               'recovery': next((item for item in recoveries if item['task_id'] == row['id']), None),
                               'applied': bool(row['applied']), 'goal_version': row['goal_version'], 'execution': execution})
        decisions, record = self.decision_record()
        running = self.lock_active('supervisor.lock')
        if not running:
            control['pid'] = None
        return {'mode': control['mode'], 'control': control, 'deliveries': deliveries,
                'campaign': self.budget, 'instructions': instructions, 'recoveries': recoveries, 'decisions': decisions, 'decision_record': record, 'supervisor_running': running,
                'guardian_running': self.lock_active('guardian.lock')}

    def lock_active(self, filename: str) -> bool:
        with (self.directory / filename).open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
            return False

    def accept(self, ident: str, decision: str, reason: str, expected_version: int | None = None) -> dict[str, Any]:
        if decision not in ('accepted', 'rejected') or not reason.strip():
            raise WorkflowError('acceptance requires a decision and evidence reason')
        with self.transaction() as db:
            row = db.execute('SELECT * FROM deliveries WHERE id=? AND role!="root"', (ident,)).fetchone()
            recovery = db.execute('SELECT value FROM recoveries WHERE task_id=?', (ident,)).fetchone()
            if recovery and json.loads(recovery['value'])['phase'] in ('waiting', 'launching', 'uncertain'):
                raise WorkflowError('delivery has pending recovery; cannot adjudicate yet')
        if row is None or not row['execution_id']:
            raise WorkflowError('unknown or unresolved worker delivery')
        execution = self.runtime(row['role'], row['store'] or self.candidates[row['role']][0]['store']).read(row['execution_id'])
        if provider_failure(execution) in ('capacity', 'connection') and (
                not recovery or json.loads(recovery['value'])['phase'] != 'stopped'):
            raise WorkflowError('delivery is awaiting recovery observation; cannot adjudicate yet')
        if execution.get('status') not in TERMINAL:
            raise WorkflowError('cannot accept an unfinished delivery')
        if decision == 'accepted' and execution.get('status') != 'succeeded':
            raise WorkflowError('failed execution cannot be accepted')
        with self.transaction() as db:
            if expected_version is not None:
                self.check_control(db, expected_version)
            value = {'decision': decision, 'reason': reason}
            current = db.execute('SELECT acceptance FROM deliveries WHERE id=?', (ident,)).fetchone()
            if current['acceptance'] and json.loads(current['acceptance']) != value:
                raise WorkflowError('delivery already adjudicated differently')
            db.execute('UPDATE deliveries SET acceptance=? WHERE id=?', (encoded(value), ident))
            self.set(db, 'idle_fingerprint', None)
        return {'task_id': ident, 'acceptance': value}


    @contextlib.contextmanager
    def owner(self) -> Iterator[None]:
        with (self.directory / 'supervisor.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkflowError('another supervisor owns dispatch') from exc
            with self.transaction() as db:
                self.set(db, 'pid', os.getpid())
            try:
                yield
            finally:
                with self.transaction() as db:
                    self.set(db, 'pid', None)
                fcntl.flock(stream, fcntl.LOCK_UN)

    def fingerprint(self, status: dict[str, Any]) -> str:
        # Model calls follow real research/runtime changes, not clock heartbeats.
        import hashlib
        workers = [{'task_id': item['task_id'], 'acceptance': item['acceptance'],
                    'status': item['execution'].get('status'),
                    'recovery': item.get('recovery'),
                    'output': item['execution'].get('output')} for item in status['deliveries']
                   if item['role'] != 'root']
        context = []
        workspace = Path(self.bindings['root']['workspace'])
        for name in ('MISSION.md', 'STATE.md', 'CAMPAIGN.md', 'RESEARCH_QUEUE.yaml'):
            path = workspace / 'docs' / name
            context.append(path.read_text() if path.exists() else '')
        return hashlib.sha256(encoded([workers, context, status['control']['instruction_version'], status['control']['provider_waits']]).encode()).hexdigest()

    def root_prompt(self, status: dict[str, Any]) -> str:
        return build_root_prompt(status, self.roles, self.candidates)

    def tick(self) -> dict[str, Any]:
        reconciliation = self.reconcile()
        if reconciliation['unresolved']:
            with self.transaction() as db:
                mode = self.get(db, 'mode')
            return {'state': mode, **reconciliation}
        status = self.status()
        recover(self, status)
        status = self.status()
        if status['mode'] == 'blocked' and self.fingerprint(status) != status['control']['blocked_fingerprint']:
            with self.transaction() as db:
                if self.get(db, 'mode') == 'blocked':
                    self.set(db, 'mode', 'active')
            status = self.status()
        if status['mode'] != 'active':
            return {'state': status['mode']}
        if time.time() < status['control']['retry_at']:
            return {'state': 'backoff'}
        for item in status['deliveries']:
            if (item['role'] == 'root' and item['applied']) or item['acceptance'] is not None:
                continue
            if item['execution'].get('status') == 'unknown':
                raise WorkflowError('execution state unavailable; do not infer completion')
        pending = [item for item in status['deliveries'] if item['role'] == 'root' and not item['applied']]
        if pending:
            item = pending[0]
            version = item['contract'].get('instruction_version', -1)
            if version != status['control']['instruction_version']:
                if item['execution'].get('status') not in TERMINAL:
                    return {'state': 'waiting_stale_root'}
                with self.transaction() as db:
                    db.execute('UPDATE deliveries SET applied=1 WHERE id=?', (item['task_id'],))
                return {'state': 'stale_decision'}
            recovery = item.get('recovery')
            if recovery and recovery['phase'] == 'stopped':
                with self.transaction() as db:
                    self.check_control(db, version)
                    self.set(db, 'mode', 'attention')
                    self.set(db, 'last_error', 'root recovery stopped: ' + recovery.get('reason', ''))
                return {'state': 'attention', 'recovery': recovery}
            if recovery and (recovery['phase'] in ('waiting', 'launching', 'uncertain', 'stopped') or (
                    recovery['phase'] == 'observing' and provider_failure(item['execution']) in ('capacity', 'connection'))):
                return {'state': 'waiting_recovery', 'recovery': recovery}
            execution = item['execution']
            if execution.get('status') not in TERMINAL:
                return {'state': 'waiting_root'}
            consumed = True
            try:
                if provider_failure(execution) == 'unavailable':
                    with self.transaction() as db:
                        self.check_control(db, version)
                    self.select_binding('root', None, item['task_id'])
                    with self.transaction() as db:
                        self.check_control(db, version)
                        self.set(db, 'failures', 0)
                        self.set(db, 'retry_at', time.time() + self.config.get('recovery_backoff_seconds', 1))
                    return {'state': 'retrying_provider'}
                if execution.get('status') != 'succeeded' or execution.get('outputTruncated'):
                    raise WorkflowError('root execution failed or decision output was truncated')
                decision = json.loads(execution.get('output', ''))
                if not isinstance(decision, dict):
                    raise WorkflowError('root decision must be an object')
                action = decision.get('action')
                major = decision.get('major_decision')
                if major is not None and (not isinstance(major, dict) or not all(isinstance(major.get(k), str) and major[k].strip() for k in ('decision', 'evidence', 'reason', 'cost_and_stop', 'outcome'))):
                    raise WorkflowError('major_decision requires decision, evidence, reason, cost_and_stop, outcome')
                if major is not None and 'decision_id' in major and (not isinstance(major['decision_id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', major['decision_id'])):
                    raise WorkflowError('major_decision decision_id must be a stable identifier')
                if action == 'dispatch':
                    result = self.dispatch(decision['task'], expected_version=version)
                elif action in ('accept', 'reject'):
                    result = self.accept(decision['task_id'],
                                         'accepted' if action == 'accept' else 'rejected', decision['reason'], expected_version=version)
                elif action == 'complete':
                    result = completion_evidence(status, decision, Path(self.bindings['root']['workspace']))
                    with self.transaction() as db:
                        self.check_control(db, version)
                        self.set(db, 'completion', result)
                        self.set(db, 'mode', 'completed')
                        self.save_decision(db, item['task_id'], {'action': 'complete', **result})
                elif action == 'blocked':
                    if not all(isinstance(decision.get(key), str) and decision[key].strip() for key in ('reason', 'review', 'resume_when')) or not isinstance(decision.get('alternatives'), list) or len(decision['alternatives']) < 2 or any(not isinstance(x, str) or not x.strip() for x in decision['alternatives']):
                        raise WorkflowError('blocked requires higher-level review, alternatives and resume conditions')
                    with self.transaction() as db:
                        self.check_control(db, version)
                        self.set(db, 'mode', 'blocked')
                        self.set(db, 'blocked', decision)
                        self.set(db, 'blocked_fingerprint', item['contract'].get('observation_fingerprint', self.fingerprint(status)))
                        self.save_decision(db, item['task_id'], decision)
                    result = decision
                elif action == 'idle' and isinstance(decision.get('reason'), str) and decision['reason'].strip():
                    with self.transaction() as db:
                        self.check_control(db, version)
                        self.set(db, 'idle_fingerprint', item['contract'].get('observation_fingerprint', self.fingerprint(status)))
                    result = {'reason': decision['reason']}
                else:
                    raise WorkflowError('invalid root decision')
            except WorkspaceConflict as exc:
                with self.transaction() as db:
                    self.save_decision(db, item['task_id'], {'action': 'workspace_conflict', 'reason': str(exc), 'outcome': 'independent work preserved; choose other work or wait'})
                return {'state': 'workspace_conflict', 'reason': str(exc)}
            except ProviderWaiting as exc:
                return {'state': 'waiting_provider', 'reason': str(exc)}
            except ResearchPaused:
                consumed = False
                return {'state': 'paused'}
            except (ValueError, KeyError, TypeError) as exc:
                raise WorkflowError('root decision violates the task contract') from exc
            finally:
                # A failed decision is consumed; an existing worker side effect
                # remains associated with its stable ID across retries.
                if consumed:
                    with self.transaction() as db:
                        db.execute('UPDATE deliveries SET applied=1 WHERE id=?', (item['task_id'],))
            with self.transaction() as db:
                self.set(db, 'failures', 0)
                self.set(db, 'last_error', None)
                self.set(db, 'processed_version', version)
                if major is not None:
                    self.save_decision(db, 'goal-' + str(item['goal_version']) + ':' + major['decision_id'] if 'decision_id' in major else item['task_id'], major, item['goal_version'])
            return {'state': 'decision_applied', 'action': action, 'result': result}
        if self.fingerprint(status) == status['control']['idle_fingerprint']:
            return {'state': 'idle'}
        with self.transaction() as db:
            started = self.get(db, 'started')
        remaining = 300
        if 'wall_time_seconds' in self.budget:
            remaining = self.budget['wall_time_seconds'] - int(time.time() - started)
        if remaining <= 0:
            raise BudgetError('campaign time budget exhausted')
        task = {'task_id': 'root-' + uuid.uuid4().hex, 'role': 'root',
                'timeout_seconds': min(300, remaining), 'instruction_version': status['control']['instruction_version'],
                'goal_version': status['control']['goal_version'], 'observation_fingerprint': self.fingerprint(status)}
        result = self.dispatch(task, root_prompt=self.root_prompt(status), expected_version=status['control']['instruction_version'])
        return {'state': 'root_started', **result}

    def step(self) -> dict[str, Any]:
        try:
            return self.tick()
        except WorkspaceConflict as exc:
            return {'state': 'workspace_conflict', 'reason': str(exc)}
        except ProviderWaiting as exc:
            return {'state': 'waiting_provider', 'reason': str(exc)}
        except ResearchPaused:
            return {'state': 'paused'}
        except WorkflowError as exc:
            with self.transaction() as db:
                if self.get(db, 'mode') == 'paused':
                    return {'state': 'paused', 'error': str(exc)}
                count = self.get(db, 'failures') + 1
                self.set(db, 'failures', count)
                self.set(db, 'last_error', str(exc))
                if isinstance(exc, BudgetError):
                    self.set(db, 'mode', 'budget_limited')
                elif count >= self.config.get('max_recovery_attempts', 3):
                    self.set(db, 'mode', 'attention')
                self.set(db, 'retry_at', time.time() + self.config.get('recovery_backoff_seconds', 1) * 2 ** min(count - 1, 5))
                mode = self.get(db, 'mode')
            return {'state': mode if mode != 'active' else 'backoff', 'error': str(exc)}

    def run(self, cycles: int | None = None) -> dict[str, Any]:
        with self.owner():
            iteration = 0
            while cycles is None or iteration < cycles:
                with self.transaction() as db:
                    if self.get(db, 'shutdown_requested'):
                        return {'state': 'stopped'}
                value = self.step()
                iteration += 1
                if value['state'] in ('attention', 'budget_limited', 'completed'):
                    return value
                time.sleep(self.config.get('poll_seconds', 2))
        return {'state': 'finished', 'cycles': iteration}

    def serve(self) -> dict[str, Any]:
        # One detached guardian supervises only its own child. No system service,
        # queue database scraping, provider protocol or unknown PID signalling.
        with (self.directory / 'guardian.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkflowError('background guardian already running') from exc
            for attempt in range(self.config.get('max_recovery_attempts', 3)):
                if attempt:
                    with self.transaction() as db:
                        if self.get(db, 'mode') != 'active':
                            return {'state': 'not_restarted'}
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                       '--config', str(self.config_path), 'supervisor', 'run'])
                if proc.returncode == 2:
                    return {'state': 'already_running'}
                if proc.returncode == 0:
                    return {'state': 'supervisor_finished'}
                with self.transaction() as db:
                    if self.get(db, 'mode') != 'active':
                        return {'state': 'not_restarted'}
                time.sleep(self.config.get('recovery_backoff_seconds', 1) * 2 ** attempt)
            with self.transaction() as db:
                if self.get(db, 'mode') == 'active':
                    self.set(db, 'mode', 'attention')
                    self.set(db, 'last_error', 'supervisor recovery limit reached')
                mode = self.get(db, 'mode')
            return {'state': mode}

    def detach(self) -> dict[str, Any]:
        for filename in ('supervisor.lock', 'guardian.lock'):
            with (self.directory / filename).open('a') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise WorkflowError('another supervisor owns dispatch') from exc
                fcntl.flock(stream, fcntl.LOCK_UN)
        # The child guardian acquires its own lock; a second launch simply exits.
        with (self.directory / 'supervisor.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                     '--config', str(self.config_path), 'supervisor', 'serve'],
                                    start_new_session=True, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=log, close_fds=True)
        return {'state': 'guardian_starting', 'pid': proc.pid, 'log': str(self.directory / 'supervisor.log')}


def pause_existing(config_path: Path, stop: bool = False) -> dict[str, Any] | None:
    """Pause is a control operation even when runtime validation is broken."""
    config = json.loads(config_path.read_text())
    directory = Path(config['state_dir'])
    if not directory.is_absolute():
        raise WorkflowError('state_dir must be absolute')
    database = directory / 'research.sqlite'
    if not database.exists():
        return None
    with sqlite3.connect(database, timeout=40) as db:
        db.execute('BEGIN IMMEDIATE')
        changed = db.execute('UPDATE control SET value=? WHERE key="mode"', (encoded('paused'),)).rowcount
        if stop:
            db.execute('INSERT OR REPLACE INTO control VALUES (?,?)', ('shutdown_requested', encoded(True)))
        current = db.execute('SELECT value FROM control WHERE key="instruction_version"').fetchone()
        version = json.loads(current[0]) + 1 if current else 1
        db.execute('INSERT OR REPLACE INTO control VALUES (?,?)', ('instruction_version', encoded(version)))
        if changed != 1:
            raise WorkflowError('existing control state is invalid')
    return {'mode': 'paused', **({'state': 'stop_requested'} if stop else {})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    supervisor = sub.add_parser('supervisor')
    supervisor.add_argument('action', choices=['pause', 'resume', 'status', 'tick', 'run', 'serve', 'stop'])
    supervisor.add_argument('--legacy-dispatch-disabled', action='store_true')
    supervisor.add_argument('--detach', action='store_true')
    supervisor.add_argument('--cycles', type=int)
    dispatch = sub.add_parser('dispatch')
    dispatch.add_argument('--task', type=Path, required=True)
    sub.add_parser('reconcile')
    decisions = sub.add_parser('decisions')
    decisions.add_argument('--export', dest='export_path', type=Path)
    ready = sub.add_parser('provider-ready')
    ready.add_argument('--role', required=True)
    ready.add_argument('--provider', required=True)
    instruction = sub.add_parser('instruct')
    instruction.add_argument('--request-id', required=True)
    instruction.add_argument('--text', required=True)
    instruction.add_argument('--new-goal', action='store_true')
    accept = sub.add_parser('accept')
    accept.add_argument('task_id')
    accept.add_argument('--decision', choices=['accepted', 'rejected'], required=True)
    accept.add_argument('--reason', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'supervisor' and args.action in ('pause', 'stop'):
            paused = pause_existing(args.config, args.action == 'stop')
            if paused is not None:
                print(encoded(paused))
                return 0
        workflow = Workflow(args.config)
        if args.command == 'decisions':
            records, path = workflow.decision_record()
            if args.export_path:
                with args.export_path.open('x') as stream:
                    stream.write(Path(path).read_text())
            value = {'decisions': records, 'record': str(args.export_path) if args.export_path else path}
        elif args.command == 'provider-ready':
            value = workflow.provider_ready(args.role, args.provider)
        elif args.command == 'instruct':
            value = workflow.instruct(args.request_id, args.text, args.new_goal)
        elif args.command == 'supervisor':
            if args.action == 'status':
                value = workflow.status()
            elif args.action == 'stop':
                workflow.control('paused')
                with workflow.transaction() as db:
                    workflow.set(db, 'shutdown_requested', True)
                value = {'state': 'stop_requested', 'mode': 'paused'}
            elif args.action in ('pause', 'resume'):
                value = workflow.control('active' if args.action == 'resume' else 'paused', args.legacy_dispatch_disabled)
            elif args.action == 'tick':
                with workflow.owner():
                    value = workflow.step()
            elif args.action == 'run':
                if args.cycles is not None and args.cycles <= 0:
                    raise WorkflowError('cycles must be positive')
                value = workflow.detach() if args.detach else workflow.run(args.cycles)
            else:
                value = workflow.serve()
        else:
            with workflow.owner():
                if args.command == 'dispatch':
                    value = workflow.dispatch(json.loads(args.task.read_text()))
                elif args.command == 'reconcile':
                    value = workflow.reconcile()
                else:
                    value = workflow.accept(args.task_id, args.decision, args.reason)
        print(encoded(value))
        return 0
    except (WorkflowError, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        print(encoded({'error': str(exc)}))
        return 2 if isinstance(exc, WorkflowError) and ('another supervisor' in str(exc) or 'guardian already' in str(exc)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
