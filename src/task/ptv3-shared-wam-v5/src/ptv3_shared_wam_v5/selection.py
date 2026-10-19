"""Additional prospective tau selection; does not alter the training objective."""
import math
from pathlib import Path
from .storage import read_json

TAU_SELECTION = 'v5.tau-short-root-relative-contact.macro-mm.v1'


def trajectory_score(evaluation):
    terms = {'short': [], 'root_relative': [], 'contact_phase': []}
    coverage = {key: [] for key in terms}
    for sid, source in evaluation.get('sources', {}).items():
        metrics = source.get('metrics', {})
        def value(key):
            row = metrics.get(key, {})
            x = row.get('mean')
            return x if row.get('count', 0) > 0 and x is not None and math.isfinite(x) else None
        short = [value('hand/h%02d' % h) for h in range(1, 5)]
        values = dict(short=sum(short)/4 if all(x is not None for x in short) else None,
                      root_relative=value('hand/root_relative'),
                      contact_phase=value('hand/contact_phase'))
        for key, x in values.items():
            if x is not None:
                terms[key].append(x)
                coverage[key].append(str(sid))
    if not all(terms.values()):
        return None
    macro = {key: sum(xs)/len(xs) for key, xs in terms.items()}
    return dict(protocol=TAU_SELECTION, score=sum(macro.values())/3,
                unit='mm', terms=macro, source_coverage=coverage)


def bootstrap_trajectory(directory):
    """Consider only still-published checkpoints, never claim lost history."""
    best = None
    for path in sorted(Path(directory).glob('dev-step*.json')):
        step = int(path.stem.split('step')[-1])
        checkpoint = path.with_name('step%06d.pt' % step)
        if not checkpoint.is_file():
            continue
        result = trajectory_score(read_json(path))
        if result and (best is None or result['score'] < best['score']):
            best = dict(result, step=step, checkpoint=str(checkpoint.resolve()),
                        origin='available-parent-checkpoints')
    return best
