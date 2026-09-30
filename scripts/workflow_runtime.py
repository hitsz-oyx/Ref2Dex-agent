"""Orchestrator CLI boundary. No provider protocol or execution-state mirror."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

TERMINAL = frozenset({'succeeded', 'failed', 'cancelled', 'timed_out'})


class WorkflowError(RuntimeError):
    pass


class Runtime:
    def __init__(self, config: dict[str, Any], binding: dict[str, Any]):
        self.command = config['command']
        self.binding = binding
        self.store = Path(binding['store']).resolve()
        self.workspace = Path(binding['workspace']).resolve()
        identity = {key: binding.get(key) for key in
                    ('provider', 'runtime', 'codex_home', 'workspace', 'env', 'runtime_config')}
        if binding.get('runtime_config'):
            identity['config_hash'] = hashlib.sha256(Path(binding['runtime_config']).read_bytes()).hexdigest()
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.store.mkdir(parents=True, exist_ok=True)
        marker = self.store / '.ref2dex-identity'
        if not marker.exists() and any(self.store.iterdir()):
            raise WorkflowError('unsealed nonempty store cannot be adopted; bind a new isolated store')
        try:
            with marker.open('x') as stream:
                stream.write(self.identity)
        except FileExistsError:
            if marker.read_text() != self.identity:
                raise WorkflowError('backend identity changed; bind a new isolated store')

    def invoke(self, *args: str) -> Any:
        # Do not inherit another role's model credentials. Binding env is local,
        # never included in status, prompts or error output.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('OPENAI_', 'ANTHROPIC_', 'AZURE_OPENAI_', 'GEMINI_', 'GOOGLE_API_'))
               and key not in {'CODEX_HOME', 'ORCHESTRATOR_HOME'}}
        env.update(self.binding.get('env', {}))
        if self.binding.get('codex_home'):
            env['CODEX_HOME'] = self.binding['codex_home']
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
                    'output', 'outputTruncated', 'exitCode', 'account', 'error') if key in value}

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
