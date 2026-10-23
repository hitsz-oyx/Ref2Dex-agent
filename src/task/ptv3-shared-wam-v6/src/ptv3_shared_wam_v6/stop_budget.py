"""Separate explicitly authorized resume stop limits from frozen model config."""
import math


def resolve_stop_budget(config, last, resume, run_seconds=None, hard_stop_epoch=None):
    explicit = run_seconds is not None or hard_stop_epoch is not None
    if explicit and (not resume or run_seconds is None or hard_stop_epoch is None):
        raise ValueError('resume stop override requires resume and both finite limits')
    seconds = config['max_seconds'] if run_seconds is None else run_seconds
    epoch = config.get('hard_stop_epoch') if hard_stop_epoch is None else hard_stop_epoch
    if explicit and (not math.isfinite(seconds) or not 0 < seconds <= 43200
                     or not math.isfinite(epoch) or epoch <= 0):
        raise ValueError('resume stop override must have positive bounded limits')
    return dict(schema='ref2dex.resume-stop-budget.v1', explicit_resume_override=explicit,
                last_step=last, seconds=seconds, hard_stop_epoch=epoch,
                elapsed_basis='current-run' if explicit else 'cumulative')


def validation_budget_view(config, budget):
    # Only the control limits differ; never feed this view to model or resume.
    if not budget['explicit_resume_override']:
        return config
    return dict(config, updates=budget['last_step'], max_seconds=budget['seconds'])


def stop_budget_exhausted(budget, elapsed_s, previous_elapsed, epoch):
    elapsed = elapsed_s if budget['explicit_resume_override'] else previous_elapsed + elapsed_s
    return elapsed > budget['seconds'] or (budget['hard_stop_epoch'] is not None
                                         and epoch >= budget['hard_stop_epoch'])
