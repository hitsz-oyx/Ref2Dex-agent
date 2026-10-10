"""Four measured states for the independent trajectory actor."""
import numpy as np

from consequence_evaluator.proposal_history import condition

SCHEMA = 'ref2dex.trajectory-policy-measured-history.v1'
INPUT_DIM = 328


def measured_history(obj, hand, q, dq, velocity):
    """Batch of four past/current states; no reference, phase or force arguments.

    Retain current absolute object height and object-frame gravity. Native
    wrist Euler rates stay in their native joint convention; XYZ rates are
    rotated into the current object frame.
    """
    obj, hand, q, dq, velocity = map(lambda value: np.asarray(value, np.float32),
                                    (obj, hand, q, dq, velocity))
    n = obj.shape[1]
    if obj.shape != (4, n, 4, 4) or hand.shape != (4, n, 11, 3) or q.shape != (4, n, 18) or dq.shape != (4, n, 18) or velocity.shape != (4, n, 6):
        raise ValueError('four measured states per row required')
    features = []
    for row in range(n):
        base, _ = condition(obj[:, row], hand[:, row], q[:, row], dq[:, row], velocity[:, row])
        rotation = obj[-1, row, :3, :3]
        wrist_rate = np.concatenate((dq[:, row, :3] @ rotation, dq[:, row, 3:6]), -1)
        gravity = -rotation[2]
        features.append(np.concatenate((base, wrist_rate.reshape(-1),
                                         obj[-1, row, 2:3, 3], gravity)))
    value = np.asarray(features, np.float32)
    if value.shape != (n, INPUT_DIM) or not np.isfinite(value).all():
        raise ValueError('finite measured history required')
    return value
