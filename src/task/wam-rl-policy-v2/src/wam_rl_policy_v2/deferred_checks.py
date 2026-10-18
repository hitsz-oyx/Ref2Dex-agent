"""Opt-in consolidation of WAM clean() host checks, without dropping validation.

Every masked finite predicate is still evaluated on device. The caller checks
all predicates once before exposing a prediction. Outside this context the
original synchronous behavior remains intact. Hooks are process-local.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import torch

_pending=ContextVar('wam_pending_finite_checks',default=None)


def install_clean_hooks(original):
    origin=getattr(original,'_deferred_origin',original)
    def clean(values,mask):
        pending=_pending.get()
        if pending is None:return origin(values,mask)
        values=torch.where(mask[...,None],values.float(),0.)
        pending.append(torch.isfinite(values).all())
        return values
    clean._deferred_origin=origin
    for name,module in list(sys.modules.items()):
        if name.startswith('ptv3_shared_wam') and module is not None:
            candidate=getattr(module,'clean',None)
            if candidate is origin or getattr(candidate,'_deferred_origin',None) is origin:
                module.clean=clean
    return clean


@contextmanager
def checked_prediction():
    checks=[];token=_pending.set(checks)
    try:
        yield checks
        if checks and not bool(torch.stack(checks).all()):
            raise ValueError('nonfinite valid observation')
    finally:
        _pending.reset(token)
