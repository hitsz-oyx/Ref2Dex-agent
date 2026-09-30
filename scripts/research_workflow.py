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

try:
    from scripts.workflow_runtime import Runtime, TERMINAL, WorkflowError
except ModuleNotFoundError:
    from workflow_runtime import Runtime, TERMINAL, WorkflowError


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
            if type(self.budget.get(key)) is not int or self.budget[key] <= 0:
                raise WorkflowError('campaign requires positive finite dispatch, root-turn and time bounds')
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
        stores = []
        for role, binding in self.bindings.items():
            if role not in self.roles:
                raise WorkflowError('unknown logical role')
            for key in ('provider', 'runtime', 'workspace', 'store'):
                if not isinstance(binding.get(key), str) or not binding[key]:
                    raise WorkflowError('binding requires provider, runtime, workspace and isolated store')
            if binding['runtime'].startswith('codex') and not binding.get('codex_home'):
                raise WorkflowError('Codex roles require an explicit CODEX_HOME')
            for key in ('workspace', 'store', 'codex_home', 'runtime_config'):
                if binding.get(key) and not Path(binding[key]).is_absolute():
                    raise WorkflowError('binding paths must be absolute')
            if not Path(binding['workspace']).is_dir():
                raise WorkflowError('role workspace does not exist')
            stores.append(str(Path(binding['store']).resolve()))
        if len(set(stores)) != len(stores):
            raise WorkflowError('roles must use independent backend stores')
        for binding in self.bindings.values():
            Runtime(self.config, binding)
        self.db_path = self.directory / 'research.sqlite'
        with self.transaction() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS deliveries (
                  id TEXT PRIMARY KEY, role TEXT NOT NULL, contract TEXT NOT NULL,
                  name TEXT NOT NULL UNIQUE, execution_id TEXT, acceptance TEXT, applied INTEGER NOT NULL DEFAULT 0);
            ''')
            defaults = {'mode': 'paused', 'workflow_id': uuid.uuid4().hex, 'started': None,
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

    def runtime(self, role: str) -> Runtime:
        return Runtime(self.config, self.bindings[role])

    def control(self, mode: str, legacy_disabled: bool = False) -> dict[str, Any]:
        with self.transaction() as db:
            if mode == 'active':
                if not legacy_disabled:
                    raise WorkflowError('resume requires confirmation that legacy dispatch owners are disabled')
                # A resume acknowledges prior failures, but does not reset spending.
                if self.get(db, 'started') is None:
                    self.set(db, 'started', time.time())
                self.set(db, 'failures', 0)
                self.set(db, 'retry_at', 0)
                self.set(db, 'idle_fingerprint', None)
            self.set(db, 'mode', mode)
        return {'mode': mode}

    def check_budget(self, db: sqlite3.Connection, role: str) -> None:
        if self.get(db, 'mode') != 'active':
            raise WorkflowError('research is paused or stopped')
        started = self.get(db, 'started')
        if started is None or time.time() - started >= self.budget['wall_time_seconds']:
            raise WorkflowError('campaign time budget exhausted')
        count = db.execute('SELECT COUNT(*) FROM deliveries WHERE ' +
                           ('role="root"' if role == 'root' else 'role!="root"')).fetchone()[0]
        limit = self.budget['max_root_turns' if role == 'root' else 'max_dispatches']
        if count >= limit:
            raise WorkflowError('campaign dispatch budget exhausted')

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
        gpu = task.get('gpu')
        permission = self.roles[role]['resources']['gpu']
        if type(gpu) is not int or gpu < 0 or gpu > self.budget['max_gpu'] or (permission == 0 and gpu):
            raise WorkflowError('task GPU request exceeds role or campaign permission')
        timeout = task.get('timeout_seconds')
        if type(timeout) is not int or not 0 < timeout <= self.budget['wall_time_seconds']:
            raise WorkflowError('task requires a bounded timeout within campaign budget')

    def dispatch(self, task: dict[str, Any], *, root_prompt: str | None = None) -> dict[str, Any]:
        if root_prompt is None:
            self.validate_contract(task)
        role = task['role']
        ident = task['task_id']
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
            self.check_budget(db, role)
            if role != 'root':
                busy = db.execute('SELECT id FROM deliveries WHERE role=? AND acceptance IS NULL', (role,)).fetchone()
                if busy:
                    raise WorkflowError('role has an unaccepted or unfinished delivery')
                used_gpu = 0
                for row in db.execute('SELECT * FROM deliveries WHERE role!="root" AND acceptance IS NULL'):
                    if row['execution_id'] is None:
                        raise WorkflowError('unresolved launch intent blocks new dispatch')
                    execution = self.runtime(row['role']).read(row['execution_id'])
                    if execution.get('status') not in TERMINAL:
                        used_gpu += json.loads(row['contract']).get('gpu', 0)
                if used_gpu + task.get('gpu', 0) > self.budget['max_gpu']:
                    raise WorkflowError('concurrent GPU budget exceeded')
            name = 'ref2dex:' + self.get(db, 'workflow_id') + ':' + ident
            db.execute('INSERT INTO deliveries(id,role,contract,name) VALUES (?,?,?,?)',
                       (ident, role, encoded(task), name))
        # Intent is durable BEFORE the external side effect. A separate dispatch
        # gate keeps control changes serialized without rolling back the intent.
        with self.transaction() as db:
            if self.get(db, 'mode') != 'active':
                # No call was made. Removing this intent is safe.
                db.execute('DELETE FROM deliveries WHERE id=? AND execution_id IS NULL', (ident,))
                raise WorkflowError('research paused before launch')
            prompt = root_prompt or ('Execute this research contract in your assigned worktree. '
                                     'Do not dispatch other roles or alter research claims. Report evidence, '
                                     'commit, resource/process disposition and next decision.\n' + encoded(task))
            execution_id = self.runtime(role).launch(name, prompt, task['timeout_seconds'])
            db.execute('UPDATE deliveries SET execution_id=? WHERE id=?', (execution_id, ident))
        return {'task_id': ident, 'execution_id': execution_id}

    def reconcile(self) -> dict[str, Any]:
        unresolved = []
        with self.transaction() as db:
            for row in db.execute('SELECT * FROM deliveries WHERE execution_id IS NULL').fetchall():
                found = self.runtime(row['role']).find(row['name'])
                if len(found) == 1:
                    db.execute('UPDATE deliveries SET execution_id=? WHERE id=?', (found[0], row['id']))
                else:
                    unresolved.append(row['id'])
            if unresolved:
                self.set(db, 'mode', 'attention')
        return {'unresolved': unresolved}

    def status(self) -> dict[str, Any]:
        with self.transaction() as db:
            control = {row['key']: json.loads(row['value']) for row in db.execute('SELECT * FROM control')}
            rows = [dict(row) for row in db.execute('SELECT * FROM deliveries ORDER BY rowid')]
        deliveries = []
        for row in rows:
            execution: dict[str, Any] = {'status': 'unknown'}
            if row['execution_id']:
                try:
                    execution = self.runtime(row['role']).read(row['execution_id'])
                except WorkflowError as exc:
                    execution = {'status': 'unknown', 'error': str(exc)}
            deliveries.append({'task_id': row['id'], 'role': row['role'],
                               'contract': json.loads(row['contract']), 'execution_id': row['execution_id'],
                               'acceptance': json.loads(row['acceptance']) if row['acceptance'] else None,
                               'applied': bool(row['applied']), 'execution': execution})
        running = self.lock_active('supervisor.lock')
        if not running:
            control['pid'] = None
        return {'mode': control['mode'], 'control': control, 'deliveries': deliveries,
                'campaign': self.budget, 'supervisor_running': running,
                'guardian_running': self.lock_active('guardian.lock')}

    def lock_active(self, filename: str) -> bool:
        with (self.directory / filename).open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
            return False

    def accept(self, ident: str, decision: str, reason: str) -> dict[str, Any]:
        if decision not in ('accepted', 'rejected') or not reason.strip():
            raise WorkflowError('acceptance requires a decision and evidence reason')
        with self.transaction() as db:
            row = db.execute('SELECT * FROM deliveries WHERE id=? AND role!="root"', (ident,)).fetchone()
            if row is None or not row['execution_id']:
                raise WorkflowError('unknown or unresolved worker delivery')
            execution = self.runtime(row['role']).read(row['execution_id'])
            if execution.get('status') not in TERMINAL:
                raise WorkflowError('cannot accept an unfinished delivery')
            if decision == 'accepted' and execution.get('status') != 'succeeded':
                raise WorkflowError('failed execution cannot be accepted')
            value = {'decision': decision, 'reason': reason}
            if row['acceptance'] and json.loads(row['acceptance']) != value:
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
                    'output': item['execution'].get('output')} for item in status['deliveries']
                   if item['role'] != 'root']
        context = []
        workspace = Path(self.bindings['root']['workspace'])
        for name in ('MISSION.md', 'STATE.md', 'CAMPAIGN.md', 'RESEARCH_QUEUE.yaml'):
            path = workspace / 'docs' / name
            context.append(path.read_text() if path.exists() else '')
        return hashlib.sha256(encoded([workers, context]).encode()).hexdigest()

    def root_prompt(self, status: dict[str, Any]) -> str:
        return (
            'You are the sole Ref2Dex research supervisor, using Codex. Read docs/MISSION.md, '
            'docs/STATE.md, docs/CAMPAIGN.md, docs/RESEARCH_QUEUE.yaml and the current workflow '
            'contract. Do not run experiments, dispatch commands, alter local control state or '
            'start other agents yourself. The deterministic executor applies your one decision. '
            'Inspect worker evidence, Git commits and resource disposition before acceptance. '
            'A process success is not a scientific conclusion. Preserve Probe/Validation boundaries. '
            'The user requires fixed logical roles and independent accounts. Choose only useful '
            'bounded tasks; GPU/time/disk permissions in CAMPAIGN remain binding. Return ONE JSON '
            'object with no fences or prose: {"action":"dispatch","task":{task_id,role,objective,'
            'decision_test,gpu,timeout_seconds,stop_conditions:[...],deliverables:[...]}} or '
            '{"action":"accept","task_id":"...","reason":"evidence checked"} or '
            '{"action":"reject","task_id":"...","reason":"..."} or '
            '{"action":"idle","reason":"no authorized useful next step"}. Never modify '
            'the logical role permissions or expand campaign limits. Before assigning another '
            'task to a role, adjudicate its previous delivery. Failed executions may be rejected '
            'but not accepted. Use a unique task_id for each new task; reusing an ID is idempotent. '
            'Integrate only verified research changes; report relevant evidence in the reason. '
            'Current factual execution snapshot and prior acceptance:\n' + encoded({
                **status, 'deliveries': [item for item in status['deliveries'] if item['role'] != 'root']}))

    def tick(self) -> dict[str, Any]:
        reconciliation = self.reconcile()
        if reconciliation['unresolved']:
            return {'state': 'attention', **reconciliation}
        status = self.status()
        if status['mode'] != 'active':
            return {'state': status['mode']}
        if time.time() < status['control']['retry_at']:
            return {'state': 'backoff'}
        for item in status['deliveries']:
            if item['execution'].get('status') == 'unknown':
                raise WorkflowError('execution state unavailable; do not infer completion')
        pending = [item for item in status['deliveries'] if item['role'] == 'root' and not item['applied']]
        if pending:
            item = pending[0]
            execution = item['execution']
            if execution.get('status') not in TERMINAL:
                return {'state': 'waiting_root'}
            try:
                if execution.get('status') != 'succeeded' or execution.get('outputTruncated'):
                    raise WorkflowError('root execution failed or decision output was truncated')
                decision = json.loads(execution.get('output', ''))
                if not isinstance(decision, dict):
                    raise WorkflowError('root decision must be an object')
                action = decision.get('action')
                if action == 'dispatch':
                    result = self.dispatch(decision['task'])
                elif action in ('accept', 'reject'):
                    result = self.accept(decision['task_id'],
                                         'accepted' if action == 'accept' else 'rejected', decision['reason'])
                elif action == 'idle' and isinstance(decision.get('reason'), str) and decision['reason'].strip():
                    with self.transaction() as db:
                        self.set(db, 'idle_fingerprint', self.fingerprint(status))
                    result = {'reason': decision['reason']}
                else:
                    raise WorkflowError('invalid root decision')
            except (ValueError, KeyError, TypeError) as exc:
                raise WorkflowError('root decision violates the task contract') from exc
            finally:
                # A failed decision is consumed; an existing worker side effect
                # remains associated with its stable ID across retries.
                with self.transaction() as db:
                    db.execute('UPDATE deliveries SET applied=1 WHERE id=?', (item['task_id'],))
            with self.transaction() as db:
                self.set(db, 'failures', 0)
                self.set(db, 'last_error', None)
            return {'state': 'decision_applied', 'action': action, 'result': result}
        if self.fingerprint(status) == status['control']['idle_fingerprint']:
            return {'state': 'idle'}
        with self.transaction() as db:
            started = self.get(db, 'started')
        remaining = self.budget['wall_time_seconds'] - int(time.time() - started)
        if remaining <= 0:
            raise WorkflowError('campaign time budget exhausted')
        task = {'task_id': 'root-' + uuid.uuid4().hex, 'role': 'root',
                'timeout_seconds': min(300, remaining)}
        result = self.dispatch(task, root_prompt=self.root_prompt(status))
        return {'state': 'root_started', **result}

    def step(self) -> dict[str, Any]:
        try:
            return self.tick()
        except WorkflowError as exc:
            with self.transaction() as db:
                count = self.get(db, 'failures') + 1
                self.set(db, 'failures', count)
                self.set(db, 'last_error', str(exc))
                if 'budget exhausted' in str(exc):
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
                value = self.step()
                iteration += 1
                if value['state'] in ('attention', 'budget_limited'):
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
                self.set(db, 'mode', 'attention')
                self.set(db, 'last_error', 'supervisor recovery limit reached')
            return {'state': 'attention'}

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


def pause_existing(config_path: Path) -> dict[str, Any] | None:
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
        if changed != 1:
            raise WorkflowError('existing control state is invalid')
    return {'mode': 'paused'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    supervisor = sub.add_parser('supervisor')
    supervisor.add_argument('action', choices=['pause', 'resume', 'status', 'tick', 'run', 'serve'])
    supervisor.add_argument('--legacy-dispatch-disabled', action='store_true')
    supervisor.add_argument('--detach', action='store_true')
    supervisor.add_argument('--cycles', type=int)
    dispatch = sub.add_parser('dispatch')
    dispatch.add_argument('--task', type=Path, required=True)
    sub.add_parser('reconcile')
    accept = sub.add_parser('accept')
    accept.add_argument('task_id')
    accept.add_argument('--decision', choices=['accepted', 'rejected'], required=True)
    accept.add_argument('--reason', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'supervisor' and args.action == 'pause':
            paused = pause_existing(args.config)
            if paused is not None:
                print(encoded(paused))
                return 0
        workflow = Workflow(args.config)
        if args.command == 'supervisor':
            if args.action == 'status':
                value = workflow.status()
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
