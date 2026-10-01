"""Persist the foreground root's explicit account routing for detached execution."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from scripts.workflow_bindings import validate_bindings
from scripts.workflow_runtime import WorkflowError


def bind_root(config_path: Path, store: Path, codex_home: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    directory = Path(config['state_dir'])
    if not directory.is_absolute() or not store.is_absolute():
        raise WorkflowError('state_dir and root store must be absolute')
    home = codex_home or (Path(os.environ['CODEX_HOME']) if os.environ.get('CODEX_HOME') else None)
    if home is None or not home.is_absolute() or not home.is_dir():
        raise WorkflowError('pass the foreground CODEX_HOME explicitly; it must be an existing absolute directory')
    if store.exists() and any(store.iterdir()):
        raise WorkflowError('bind-root requires a new empty root store')
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for filename in ('bootstrap.lock', 'supervisor.lock', 'guardian.lock'):
            stream = stack.enter_context((directory / filename).open('a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkflowError('stop the background owner before binding root') from exc
        database = directory / 'research.sqlite'
        if database.exists():
            db = stack.enter_context(sqlite3.connect(database, timeout=40))
            db.execute('BEGIN IMMEDIATE')
            mode = db.execute('SELECT value FROM control WHERE key="mode"').fetchone()
            if not mode or json.loads(mode[0]) != 'paused':
                raise WorkflowError('pause the workflow before binding root')
            if db.execute('SELECT COUNT(*) FROM deliveries').fetchone()[0]:
                raise WorkflowError('existing execution history requires a new campaign; do not replace its root binding')
        # Capture only routing/configuration, never auth tokens or API keys. The
        # CODEX_HOME itself supplies credentials and persisted model settings.
        routing = ('PATH', 'HOME', 'XDG_CONFIG_HOME', 'http_proxy', 'https_proxy',
                   'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'no_proxy', 'SSL_CERT_FILE',
                   'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE')
        root = config['bindings']['root']
        root.update(runtime='codex', provider='foreground-root', codex_home=str(home.resolve()),
                    store=str(store.resolve()), configuration_source='foreground',
                    env={key: os.environ[key] for key in routing if key in os.environ})
        root.pop('inherit_current_session', None)
        roles = yaml.safe_load(Path(config['roles_file']).read_text())
        validate_bindings(config, {item['agent_key']: item for item in roles['roles']}, config['campaign'])
        temporary = config_path.with_name(config_path.name + '.bootstrap.tmp')
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(config, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(config_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {'state': 'root_bound', 'codex_home': root['codex_home'], 'store': root['store'],
            'configuration_source': 'foreground'}
