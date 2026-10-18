"""Optional CUDA event timing without per-stage synchronization."""
import time
import numpy as np
import torch


class StepTimings:
    STAGES = ('h2d', 'forward_loss', 'backward', 'optimizer')

    def __init__(self, limit=0):
        self.limit, self.count, self.pending = limit, 0, []
        self.active = False

    def before_batch(self):
        self.active = self.count < self.limit
        if self.active:
            self.wait_started = time.monotonic()

    def after_batch(self):
        if self.active:
            self.wait_seconds = time.monotonic() - self.wait_started
            self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            self.events[0].record()

    def mark(self, index):
        if self.active:
            self.events[index].record()
            if index == 4:
                self.pending.append((self.wait_seconds, self.events))
                self.count += 1

    def summary(self):
        """Caller has synchronized at the normal logging boundary."""
        if not self.pending:
            return None
        rows = {'input_wait': [wait for wait, _ in self.pending]}
        for i, name in enumerate(self.STAGES):
            rows[name] = [events[i].elapsed_time(events[i+1])/1000 for _, events in self.pending]
        result = dict(steps=len(self.pending), seconds={k:dict(mean=float(np.mean(v)),
            median=float(np.median(v)), p95=float(np.percentile(v,95))) for k,v in rows.items()})
        self.pending.clear()
        return result
