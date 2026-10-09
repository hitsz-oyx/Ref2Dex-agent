"""Explicit execution contracts for fresh native Gate1 replay.

Isaac Gym has two independent device choices here: the device on which PhysX
steps and the device used for tensor exchange.  Keeping them in one
``physics_device`` flag made it impossible to reproduce ref13's verified
GPU-PhysX/CPU-pipeline run.  This module is deliberately dependency-free so
the contract can be tested without importing Isaac Gym or Torch.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class NativeBackend:
    """Resolved command-line and runtime contract for one native worker."""

    name: str
    sim_device: str
    pipeline: str
    physx_use_gpu: bool
    physx_num_threads: Optional[int]
    tensor_device: str
    actor_device: str

    @property
    def is_gpu_physx(self) -> bool:
        return self.physx_use_gpu

    def argv(self) -> Tuple[str, ...]:
        """Return the native evaluate.py device arguments."""
        values = [
            '--sim_device', self.sim_device,
            '--rl_device', self.actor_device,
            '--pipeline', self.pipeline,
        ]
        if self.physx_num_threads is not None:
            values.extend(('--num_threads', str(self.physx_num_threads)))
        return tuple(values)

    def as_dict(self) -> Dict[str, object]:
        """Stable JSON representation included in manifests and packets."""
        return {
            'name': self.name,
            'sim_device': self.sim_device,
            'pipeline': self.pipeline,
            'physx_use_gpu': self.physx_use_gpu,
            'physx_num_threads': self.physx_num_threads,
            'tensor_device': self.tensor_device,
            'actor_device': self.actor_device,
        }


def canonical_device(value: str) -> str:
    """Normalize Isaac/rl-games' shorthand ``cuda`` to the explicit card."""
    value = str(value)
    return 'cuda:0' if value == 'cuda' else value


_GPU = NativeBackend(
    name='gpu_physx_gpu_pipeline', sim_device='cuda:0', pipeline='gpu',
    physx_use_gpu=True, physx_num_threads=8,
    tensor_device='cuda:0', actor_device='cuda:0')
_GPU_THREADS1 = NativeBackend(
    name='gpu_physx_gpu_pipeline_threads1', sim_device='cuda:0', pipeline='gpu',
    physx_use_gpu=True, physx_num_threads=1,
    tensor_device='cuda:0', actor_device='cuda:0')
_CPU = NativeBackend(
    name='cpu_physx_cpu_pipeline', sim_device='cpu', pipeline='cpu',
    physx_use_gpu=False, physx_num_threads=1,
    tensor_device='cpu', actor_device='cuda:0')
_HOST = NativeBackend(
    name='gpu_physx_cpu_pipeline', sim_device='cuda:0', pipeline='cpu',
    physx_use_gpu=True, physx_num_threads=1,
    tensor_device='cpu', actor_device='cuda:0')


BACKENDS = {
    'gpu': _GPU,
    'gpu_physx_gpu_pipeline': _GPU,
    'gpu_physx_gpu_pipeline_threads1': _GPU_THREADS1,
    'cpu': _CPU,
    'cpu_physx_cpu_pipeline': _CPU,
    'host': _HOST,
    'gpu_physx_cpu_pipeline': _HOST,
}


def resolve_backend(name: str) -> NativeBackend:
    """Resolve a CLI backend name, rejecting accidental cross-device drift."""
    try:
        return BACKENDS[name]
    except KeyError:
        choices = ', '.join(sorted(BACKENDS))
        raise ValueError('unknown native backend %r (choose %s)' % (name, choices))


def resolve_legacy_backend(backend: Optional[str], physics_device: Optional[str]) -> NativeBackend:
    """Resolve the new option while accepting the previous two-value flag.

    ``--physics-device gpu|cpu`` remains an alias for old manifests and shell
    snippets.  Supplying both options is allowed only when they resolve to the
    same contract.
    """
    selected = backend or physics_device or 'gpu'
    resolved = resolve_backend(selected)
    if backend is not None and physics_device is not None:
        legacy = resolve_backend(physics_device)
        if legacy != resolved:
            raise ValueError('--backend and --physics-device select different contracts')
    return resolved
