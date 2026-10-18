"""Best-score lineage across immutable run directories and resumptions."""
import math
from pathlib import Path
import os
import shutil

from .objectives import EVALUATION_SEMANTICS


def consider_selection(previous, score, step, checkpoint, run_id):
    if not math.isfinite(score):
        raise ValueError('finite validation score required')
    if previous is not None:
        if previous.get('evaluation_semantics') != EVALUATION_SEMANTICS:
            raise ValueError('historical best uses a different evaluation protocol')
        if not math.isfinite(previous['score']):
            raise ValueError('finite historical best score required')
        if score >= previous['score']:
            return dict(previous), False
    return dict(score=float(score), step=int(step), checkpoint=str(Path(checkpoint).resolve()),
                run_id=run_id, evaluation_semantics=EVALUATION_SEMANTICS), True


def inherit_checkpoint(output, source):
    """Own a retained checkpoint by hard link or bounded copy; never overwrite."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_file() or not source.name.startswith('step') or source.suffix != '.pt':
        raise ValueError('historical best checkpoint is unavailable: ' + str(source))
    int(source.stem[4:])
    dest = output / source.name
    if dest.exists():
        raise FileExistsError('immutable inherited checkpoint: ' + str(dest))
    try:
        os.link(source, dest)
    except OSError:
        with source.open('rb') as read, dest.open('xb') as write:
            shutil.copyfileobj(read, write, 1024*1024)
    return dest
