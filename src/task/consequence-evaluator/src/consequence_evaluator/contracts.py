"""Lightweight contracts: safe before Isaac Gym initializes Torch."""
from pathlib import Path

K = 24
K_EXEC = 8
SCHEMA = 'ref2dex.consequence-evaluator.windows.v2'
EPISODE_SCHEMA = 'ref2dex.consequence-evaluator.episodes.v2'
ACTION_SEMANTICS = 'decision_known_requested_residual_plan'
HAND_LINKS = ('hand_base_link','thumb_proximal_base','thumb_tip',
              'index_proximal','index_tip','middle_proximal','middle_tip',
              'ring_proximal','ring_tip','pinky_proximal','pinky_tip')
FUTURE_DIM = 12 + 3 * len(HAND_LINKS)
ARMS = ('baseline','oracle_object','oracle_interaction')


def future_mode(arm):
    return {'baseline':False,'oracle_object':'object','oracle_interaction':'interaction'}[arm]


def is_within(path, root):
    """Path ownership check compatible with the native Python3.8 environment."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
