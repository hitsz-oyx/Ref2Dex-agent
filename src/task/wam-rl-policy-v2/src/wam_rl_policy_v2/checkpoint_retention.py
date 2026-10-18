"""Bounded retention for this run's official PPO checkpoints only."""
import re


def prune_checkpoints(directory, final_iteration=None):
    checkpoints = []
    for path in directory.glob('official-iteration-*.pt'):
        match = re.fullmatch(r'official-iteration-(\d+)\.pt', path.name)
        if match and path.is_file() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    recent = {n for n, _ in checkpoints[-3:]}
    removed = []
    for n, path in checkpoints:
        if n not in recent and n % 500 != 0 and n != final_iteration:
            path.unlink()
            removed.append(path.name)
    return removed
