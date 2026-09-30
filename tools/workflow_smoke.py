#!/usr/bin/env python3
"""Real Orchestrator process smoke in an isolated temporary campaign; no model calls."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True, type=Path)
    parser.add_argument('--cli', type=Path, default=ROOT / 'tools/workflow_runtime/node_modules/@backnotprop/orchestrator-cli/dist/cli.js')
    args = parser.parse_args()
    (ROOT / '.runtime').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='workflow-smoke-', dir=ROOT / '.runtime') as temporary:
        area = Path(temporary)
        (area / 'worker.py').write_text('print("ENGINEERING_SMOKE_OK")\n')
        runtime_config = area / 'runtime.json'
        runtime_config.write_text(json.dumps({'agents': {'engineering-smoke': {
            'adapter': 'process', 'command': sys.executable,
            'args': [str(area / 'worker.py'), '{prompt}'], 'output': 'text'}}}))
        config = area / 'config.json'
        config.write_text(json.dumps({'command': [str(args.node.resolve()), str(args.cli.resolve())],
            'state_dir': str(area / 'control'), 'roles_file': str(ROOT / 'docs/AGENT_ROLES.yaml'),
            'bindings': {
                'root': {'provider': 'unused-codex', 'runtime': 'codex', 'codex_home': str(area / 'unused-account'),
                         'workspace': str(ROOT), 'store': str(area / 'unused-store')},
                'agent_infra': {'provider': 'engineering', 'runtime': 'engineering-smoke', 'workspace': str(ROOT),
                    'store': str(area / 'infra-store'), 'runtime_config': str(runtime_config),
                    'env': {'PATH': str(args.node.resolve().parent) + ':' + os.environ.get('PATH', '/usr/bin:/bin')}}},
            'campaign': {'max_dispatches': 1, 'max_root_turns': 1, 'wall_time_seconds': 60, 'max_gpu': 0}}))
        task = area / 'task.json'
        task.write_text(json.dumps({'task_id': 'engineering-smoke', 'role': 'agent_infra',
            'objective': 'Verify external process launch', 'decision_test': 'Observe known process output',
            'gpu': 0, 'timeout_seconds': 20, 'stop_conditions': ['20 seconds'],
            'deliverables': ['ENGINEERING_SMOKE_OK']}))

        def call(*arguments: str) -> dict:
            proc = subprocess.run([sys.executable, str(ROOT / 'scripts/researchctl.py'), '--config', str(config),
                                   *arguments], capture_output=True, text=True, cwd=ROOT, timeout=40)
            value = json.loads(proc.stdout)
            if proc.returncode:
                raise RuntimeError(value.get('error', 'workflow command failed'))
            return value

        call('supervisor', 'resume', '--legacy-dispatch-disabled')
        launched = call('dispatch', '--task', str(task))
        call('supervisor', 'pause')
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            delivery = call('supervisor', 'status')['deliveries'][0]
            execution = delivery['execution']
            if execution['status'] == 'succeeded':
                if execution.get('output', '').strip() != 'ENGINEERING_SMOKE_OK' or delivery['acceptance'] is not None:
                    raise RuntimeError('unexpected delivery result')
                print(json.dumps({'status': 'PASS', 'execution_id': launched['execution_id'],
                                  'output': 'ENGINEERING_SMOKE_OK', 'model_calls': 0}))
                return 0
            if execution['status'] in ('failed', 'cancelled', 'timed_out'):
                raise RuntimeError('engineering process failed')
            time.sleep(0.1)
        raise RuntimeError('engineering process did not complete before deadline')


if __name__ == '__main__':
    raise SystemExit(main())
