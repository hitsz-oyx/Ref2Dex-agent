"""Observed scene/hand shared forecasting and robot action retargeting."""

__all__ = ["SharedWAM"]


def __getattr__(name):
    # paths/execution are also imported by IsaacGym BEFORE importing torch.
    if name == "SharedWAM":
        from .model import SharedWAM
        return SharedWAM
    raise AttributeError(name)
