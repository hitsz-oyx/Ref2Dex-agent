"""Action-buffer semantics independent of model and simulator runtimes."""
from collections import deque
import numpy as np


class ActionBuffer:
    """Query at buffer exhaustion; consume first K commands without replanning."""
    def __init__(self, execution_steps):
        if execution_steps not in (1, 4, 8):
            raise ValueError("K_exec must be 1,4,8")
        self.execution_steps = execution_steps
        self.queue = deque()
        self.last_query = None

    @property
    def needs_query(self):
        return not self.queue

    def refill(self, chunk, tick):
        chunk = np.asarray(chunk, np.float32)
        if not self.needs_query or chunk.shape != (8, 12) or not np.isfinite(chunk).all():
            raise ValueError("empty buffer and finite8x12 action chunk required")
        self.queue.extend((i, row.copy()) for i, row in enumerate(chunk[:self.execution_steps]))
        self.last_query = int(tick)

    def pop(self):
        if self.needs_query:
            raise ValueError("action buffer is empty")
        index, raw = self.queue.popleft()
        return raw, dict(query_tick=self.last_query, chunk_index=index)
