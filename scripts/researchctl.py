#!/usr/bin/env python3
"""Ref2Dex workflow command entry; legacy control requires explicit selection."""
from __future__ import annotations

import sys
from pathlib import Path


def write_lease(*args, **kwargs):
    """Compatibility API for old tasks; new supervision does not use leases."""
    try:
        from scripts.legacy_researchctl import write_lease as legacy_write
    except ModuleNotFoundError:
        from legacy_researchctl import write_lease as legacy_write
    return legacy_write(*args, **kwargs)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    legacy_flags = ('--lease', '--registry', '--broker-tasks-db', '--broker-state-db',
                    '--broker-roles', '--broker-bindings')
    if '--legacy' in args or any(arg.split('=')[0] in legacy_flags for arg in args):
        args = [arg for arg in args if arg != '--legacy']
        try:
            from scripts.legacy_researchctl import main as legacy_main
        except ModuleNotFoundError:
            from legacy_researchctl import main as legacy_main
        return legacy_main(args)
    try:
        from scripts.research_workflow import main as workflow_main
    except ModuleNotFoundError:
        from research_workflow import main as workflow_main
    if not any(arg.split('=')[0] == '--config' for arg in args):
        args = ['--config', str(Path('.runtime/workflow.json').resolve()), *args]
    return workflow_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
