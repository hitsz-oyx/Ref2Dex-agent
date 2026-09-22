"""Small observation policy used by the no-checkpoint DExplore teacher path."""
from __future__ import annotations

import torch


class DExploreBcPolicy(torch.nn.Module):
    """The compact MLP used by the local DAgger policy artifact."""

    def __init__(self, observation_dim: int, action_dim: int = 18,
                 hidden_dims: tuple[int, ...] = (512, 256, 128)) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim != 18:
            raise ValueError("DExplore BC policy dimensions are invalid")
        dims = (int(observation_dim),) + tuple(int(v) for v in hidden_dims)
        if any(v < 1 for v in dims):
            raise ValueError("hidden dimensions must be positive")
        layers: list[torch.nn.Module] = []
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            layers.extend((torch.nn.Linear(input_dim, output_dim), torch.nn.ReLU()))
        layers.extend((torch.nn.Linear(dims[-1], action_dim), torch.nn.Tanh()))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.network[0].in_features:
            raise ValueError("observation must be [B, observation_dim]")
        if not torch.isfinite(observation).all():
            raise FloatingPointError("observation contains non-finite values")
        return self.network(observation)


def normalized_action(model: DExploreBcPolicy, observation: torch.Tensor,
                      mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Evaluate a policy checkpoint with the normalization saved beside it."""
    if mean.ndim != 1 or std.shape != mean.shape or observation.shape[-1] != mean.numel():
        raise ValueError("normalization statistics do not match observation")
    if (std <= 0).any() or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("normalization statistics must be finite and positive")
    return model(((observation - mean) / std).clamp(-10, 10))
