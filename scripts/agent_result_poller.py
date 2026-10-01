#!/usr/bin/env python3
"""Compatibility entry point for the retired result poller.

Current worker and root daemons import :mod:`scripts.runtime_support` directly.
This module remains only so old commands and recorded references keep working;
it exposes the same module object and CLI without making legacy polling the
current runtime dependency.
"""

from __future__ import annotations

import sys

try:
    from scripts import runtime_support as _runtime
except ModuleNotFoundError:  # pragma: no cover - direct script path
    import runtime_support as _runtime

if __name__ == "__main__":  # pragma: no cover - compatibility CLI
    raise SystemExit(_runtime.main())

# Preserve monkeypatching and the historical import API: callers importing the
# compatibility name receive the support module itself, so its function globals
# and patched helpers stay coherent.
sys.modules[__name__] = _runtime
