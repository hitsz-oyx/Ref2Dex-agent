"""Explicit role bindings and provider alternatives."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.workflow_runtime import Runtime, WorkflowError


def validate_bindings(config: dict[str, Any], roles: dict[str, Any], budget: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if config['bindings'].get('root', {}).get('inherit_current_session'):
        raise WorkflowError('root configuration has not been captured; run bind-root first')
    bindings = {}
    for role, primary in config['bindings'].items():
        alternatives = primary.get('fallbacks', [])
        if not isinstance(alternatives, list) or any(not isinstance(item, dict) or item.get('verified') is not True for item in alternatives):
            raise WorkflowError('fallback bindings must be explicitly verified')
        candidates = [{key: value for key, value in primary.items() if key != 'fallbacks'}, *(dict(item) for item in alternatives)]
        if len({item.get('provider') for item in candidates}) != len(candidates):
            raise WorkflowError('a role requires distinct provider binding names')
        if any(item.get('workspace') != primary.get('workspace') for item in alternatives):
            raise WorkflowError('provider fallback must retain the role workspace')
        bindings[role] = candidates
    workspaces = [str(Path(items[0]['workspace']).resolve()) for items in bindings.values()]
    if len(set(workspaces)) != len(workspaces):
        raise WorkflowError('logical roles require separate workspaces')
    stores = []
    for role, candidates in bindings.items():
        account_homes = [str(Path(item['codex_home']).resolve()) for item in candidates if item.get('codex_home')]
        if len(set(account_homes)) != len(account_homes):
            raise WorkflowError('a role requires independent CODEX_HOME directories for alternatives')
        for binding in candidates:
            if role == 'root' and binding.get('runtime') != 'codex':
                raise WorkflowError('root bindings require Codex runtime')
            if role not in roles:
                raise WorkflowError('unknown logical role')
            for key in ('provider', 'runtime', 'workspace', 'store'):
                if not isinstance(binding.get(key), str) or not binding[key]:
                    raise WorkflowError('binding requires provider, runtime, workspace and isolated store')
            if binding['runtime'] not in ('codex', 'codex-app-server'):
                if binding['runtime'] in ('claude-code', 'copilot', 'grok', 'pi', 'shell') or binding.get('engineering_only') is not True or not binding.get('runtime_config'):
                    raise WorkflowError('non-Codex model account isolation is unverified; use an explicit custom engineering process')
                try:
                    definition = json.loads(Path(binding['runtime_config']).read_text())['agents'][binding['runtime']]
                    valid_process = definition['adapter'] == 'process' and isinstance(definition['command'], str) and bool(definition['command'])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise WorkflowError('engineering runtime requires an explicit process definition') from exc
                if not valid_process:
                    raise WorkflowError('engineering runtime requires an explicit process definition')
            if binding['runtime'] in ('codex', 'codex-app-server') and not binding.get('codex_home'):
                raise WorkflowError('Codex roles require an explicit CODEX_HOME')
            for key in ('workspace', 'store', 'codex_home', 'runtime_config'):
                if binding.get(key) and not Path(binding[key]).is_absolute():
                    raise WorkflowError('binding paths must be absolute')
            allowed = budget.get('allowed_workspace_roots')
            if not isinstance(allowed, list) or not allowed or any(
                    not isinstance(item, str) or not Path(item).is_absolute() for item in allowed):
                raise WorkflowError('campaign requires explicit absolute allowed_workspace_roots')
            workspace = Path(binding['workspace']).resolve()
            permitted = False
            for allowed_path in allowed:
                try:
                    workspace.relative_to(Path(allowed_path).resolve())
                    permitted = True
                except ValueError:
                    pass
            if not permitted:
                raise WorkflowError('role workspace is outside authorized campaign roots')
            if not Path(binding['workspace']).is_dir():
                raise WorkflowError('role workspace does not exist')
            binding['logical_role'] = role
            stores.append(str(Path(binding['store']).resolve()))
    if len(set(stores)) != len(stores):
        raise WorkflowError('roles must use independent backend stores')
    for candidates in bindings.values():
        for binding in candidates:
            Runtime(config, binding)
    return bindings


def protect_workspace(binding: dict[str, Any]) -> None:
    workspace = Path(binding['workspace'])
    result = subprocess.run(['git', '-C', str(workspace), 'rev-parse', '--show-toplevel'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise WorkspaceConflict('role workspace must be its own Git worktree')
    if result.returncode == 0:
        if Path(result.stdout.strip()).resolve() != workspace.resolve():
            raise WorkflowError('role workspace must be its own worktree root')
        status = subprocess.run(['git', '-C', str(workspace), 'status', '--porcelain'],
                                capture_output=True, text=True, timeout=10)
        if status.returncode or status.stdout.strip():
            raise WorkspaceConflict('independent or uncommitted work in ' + str(workspace))


class WorkspaceConflict(WorkflowError):
    pass
