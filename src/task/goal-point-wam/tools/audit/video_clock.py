"""Native H4/K24 windows extracted from cached 10Hz clip measurements."""
import numpy as np


def cached_window_clock(teacher_frame_ids, window_offset):
    frames = np.asarray(teacher_frame_ids)
    if (frames.ndim != 1 or frames.dtype.kind not in 'iu' or len(frames) != 11
            or frames[0] < 0 or not np.array_equal(np.diff(frames), np.full(10, 3))):
        raise ValueError('teacher requires eleven consecutive10Hz native samples')
    if not isinstance(window_offset, (int, np.integer)) or window_offset not in (0, 3, 6):
        raise ValueError('cached window offset must be0/3/6 native ticks')
    current = int(frames[0]) + int(window_offset)
    if current < 3:
        raise ValueError('current tick requires three past native frames')
    if current + 24 > frames[-1]:
        raise ValueError('cached clip does not cover all24 future ticks')
    return current, int(window_offset)//3, np.arange(current-3, current+1)
