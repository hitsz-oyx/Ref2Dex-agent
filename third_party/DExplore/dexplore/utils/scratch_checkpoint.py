"""Fail-closed authorization for resuming a Ref2Dex scratch policy."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def authorize_scratch_restore(checkpoint: str | None, resume: int,
                              environ: Mapping[str, str]) -> bool:
    """Authorize only an explicitly path- and SHA-pinned scratch restore."""
    if environ.get("REF2DEX_SCRATCH_POLICY") != "1":
        return False
    requested = checkpoint not in (None, "", "Base") or bool(resume)
    if not requested:
        return False
    allowed = environ.get("REF2DEX_SCRATCH_RESUME_CHECKPOINT")
    expected = environ.get("REF2DEX_SCRATCH_RESUME_SHA256")
    candidate = Path(checkpoint).resolve() if checkpoint not in (None, "", "Base") else None
    if (not allowed or not expected or candidate is None or
            candidate != Path(allowed).resolve() or len(expected) != 64 or
            not candidate.is_file() or
            hashlib.sha256(candidate.read_bytes()).hexdigest() != expected):
        raise ValueError("Scratch-policy training forbids unverified actor checkpoints")
    return True
