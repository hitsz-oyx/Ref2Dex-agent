"""Orchestrator CLI boundary. No provider protocol or execution-state mirror."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

TERMINAL = frozenset({'succeeded', 'failed', 'cancelled', 'timed_out'})


def provider_failure(execution: dict[str, Any]) -> str | None:
    if execution.get('status') != 'failed':
        return None
    error = str(execution.get('error', '')) + ' ' + str(execution.get('output', ''))
    if re.search(r'Selected model is at capacity', error, re.I):
        return 'capacity'
    if re.search(r'quota (?:exhausted|exceeded)|insufficient_quota|rate.limit|provider unavailable', error, re.I):
        return 'unavailable'
    if re.search(r'connection (?:refused|reset|failed)|connect(?:ion)? timeout|provider (?:connection|network) error|HTTP (?:502|503|504)', error, re.I):
        return 'connection'
    return None


class WorkflowError(RuntimeError):
    pass


class BudgetError(WorkflowError):
    pass


class ProviderWaiting(WorkflowError):
    pass


class ResearchPaused(WorkflowError):
    pass


class NativeGoalHeld(WorkflowError):
    pass


def runtime_environment(binding: dict[str, Any]) -> dict[str, str]:
    allowed = {'PATH', 'HOME', 'XDG_CONFIG_HOME', 'TMPDIR', 'LANG', 'TZ', 'LD_LIBRARY_PATH',
               'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONPATH',
               'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'no_proxy',
               'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE'}
    env = {key: value for key, value in os.environ.items() if key in allowed or key.startswith('LC_')}
    env.update(binding.get('env', {}))
    if binding.get('logical_role'):
        env['REF2DEX_ROLE'] = binding['logical_role']
    if binding.get('codex_home'):
        env['CODEX_HOME'] = binding['codex_home']
    return env


def identity_sources(binding: dict[str, Any]) -> dict[str, Any]:
    env = runtime_environment(binding)
    home = Path(env.get('HOME', str(Path.home())))
    xdg = Path(env.get('XDG_CONFIG_HOME', str(home / '.config')))
    workspace = Path(binding['workspace'])
    sources = [xdg / 'orchestrator/config.json', home / '.orchestrator/config.json',
               workspace / 'orchestrator.config.json', workspace / '.orchestrator/config.json',
               workspace / '.codex/config.toml']
    account = None
    if binding.get('codex_home'):
        codex_home = Path(binding['codex_home'])
        sources.append(codex_home / 'config.toml')
        sources.extend(sorted(codex_home.glob('*.config.toml')))
        auth = codex_home / 'auth.json'
        if auth.exists():
            try:
                value = json.loads(auth.read_text())
                tokens = value.get('tokens') or {}
                # OAuth token refresh should not invalidate an unchanged account.
                account = {'mode': value.get('auth_mode'), 'account_id': tokens.get('account_id'),
                           'api_key': value.get('OPENAI_API_KEY')}
                if not account['account_id'] and not account['api_key']:
                    account = {key: item for key, item in value.items() if key != 'last_refresh'}
            except (ValueError, AttributeError) as exc:
                raise WorkflowError('Codex account metadata is invalid') from exc
    if binding.get('runtime_config'):
        sources.append(Path(binding['runtime_config']))
    return {'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                      for path in sources},
            'account': hashlib.sha256(json.dumps(account, sort_keys=True).encode()).hexdigest()}


class Runtime:
    def __init__(self, config: dict[str, Any], binding: dict[str, Any]):
        self.command = config['command']
        self.binding = binding
        self.store = Path(binding['store']).resolve()
        self.workspace = Path(binding['workspace']).resolve()
        identity = {key: binding.get(key) for key in
                    ('provider', 'runtime', 'codex_home', 'workspace', 'env', 'runtime_config', 'logical_role')}
        identity['effective_config'] = identity_sources(binding)
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.store.mkdir(parents=True, exist_ok=True)
        marker = self.store / '.ref2dex-identity'
        lock = self.store / '.ref2dex-identity.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if marker.exists():
                if marker.read_text() != self.identity:
                    raise WorkflowError('backend identity changed; bind a new isolated store')
            else:
                if any(path != lock for path in self.store.iterdir()):
                    raise WorkflowError('unsealed nonempty store cannot be adopted; bind a new isolated store')
                temporary = self.store / '.ref2dex-identity.tmp'
                temporary.write_text(self.identity)
                temporary.replace(marker)

    def invoke(self, *args: str) -> Any:
        # All credentials and routing additions must be explicit in the local binding.
        env = runtime_environment(self.binding)
        command = [*self.command, '--workspace', str(self.workspace),
                   '--orchestrator-dir', str(self.store)]
        if self.binding.get('runtime_config'):
            command += ['--config', self.binding['runtime_config']]
        try:
            proc = subprocess.run([*command, *args, '--json'], cwd=self.workspace,
                                  env=env, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkflowError('backend unavailable or command timed out; reconcile before retry') from exc
        if proc.returncode:
            raise WorkflowError('backend command failed; inspect the account-scoped backend logs')
        try:
            value = json.loads(proc.stdout)
        except ValueError as exc:
            raise WorkflowError('backend returned invalid JSON; reconcile before retry') from exc
        if isinstance(value, dict) and value.get('ok') is False:
            raise WorkflowError('backend rejected command')
        return value

    def launch(self, name: str, prompt: str, timeout: int) -> str:
        value = self.invoke('launch', self.binding['runtime'], '--name', name,
                            '--timeout-ms', str(timeout * 1000), '--max-output-bytes', '200000', prompt)
        execution = value.get('task', value)
        ident = execution.get('taskId')
        if not isinstance(ident, str) or not ident:
            raise WorkflowError('launch response lacks taskId; reconcile before retry')
        return ident

    def read(self, execution_id: str) -> dict[str, Any]:
        value = self.invoke('read', execution_id, '--max-bytes', '200000')
        # Full read JSON is a task summary, not the compact multi-read view.
        if not isinstance(value, dict) or value.get('taskId') != execution_id:
            raise WorkflowError('backend returned the wrong execution identity')
        if value.get('state') in ('stale', 'orphaned', 'lost'):
            value['status'] = 'unknown'
        return {key: value[key] for key in ('taskId', 'name', 'status', 'state', 'active',
                    'output', 'outputTruncated', 'exitCode', 'account', 'error', 'provider', 'resume', 'resumedFrom', 'goal', 'session') if key in value}

    def continue_task(self, execution: dict[str, Any], name: str, prompt: str, timeout: int) -> str:
        provider = execution.get('provider') or {}
        if self.binding['runtime'] in ('codex', 'codex-app-server') and provider.get('threadId'):
            value = self.invoke('resume', execution['taskId'], '--name', name,
                                '--timeout-ms', str(timeout * 1000), '--max-output-bytes', '200000', prompt)
            task = value.get('task', value)
            ident = task.get('taskId')
            if not isinstance(ident, str) or not ident:
                raise WorkflowError('resume response lacks taskId; reconcile before retry')
            return ident
        return self.launch(name, prompt, timeout)

    def continuation_source(self, execution: dict[str, Any]) -> dict[str, Any]:
        if self.binding['runtime'] not in ('codex', 'codex-app-server'):
            return execution
        # read omits provider metadata in CLI 0.1.0. Use normalized public
        # agent events; goal-get is only supported by app-server session tasks.
        value = self.invoke('events', execution['taskId'], '--agent-only', '--compact', '--max-bytes', '2000000')
        if value.get('taskId') != execution['taskId']:
            raise WorkflowError('continuation metadata belongs to another execution')
        state = execution.get('goal') or {}
        if state.get('status') in ('paused', 'complete', 'completed', 'budget-limited', 'usage-limited'):
            raise NativeGoalHeld('native Goal is paused, complete or budget limited')
        thread_id = None
        for event in value.get('events', []):
            data = event.get('data') or {}
            if data.get('kind') == 'thread.started' and isinstance(data.get('threadId'), str):
                thread_id = data['threadId']
        if not thread_id and value.get('eventsTruncated'):
            raise WorkflowError('continuation metadata is truncated; do not assume a new session is safe')
        return {**execution, 'provider': {'provider': 'codex', 'threadId': thread_id} if thread_id else {}}

    def find(self, name: str) -> list[str]:
        view = self.invoke('ps', '--all')
        found: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get('name') == name and isinstance(value.get('taskId'), str):
                    found.add(value['taskId'])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(view)
        return sorted(found)
