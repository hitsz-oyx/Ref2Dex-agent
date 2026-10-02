"""Causal relative-task-value ranking on frozen Cm physical predictions."""
from __future__ import annotations

from pathlib import Path

import torch


class FrozenRelativeTaskValueSelector:
    """Use a frozen ridge head to rank safe candidates against fixed Cup."""

    def __init__(self, checkpoint: Path | str, device: torch.device | str):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        schema = payload.get("schema")
        if schema not in (
            "ref2dex.relative_task_value_calibration.v1",
            "ref2dex.current_decision_task_value.v1",
        ):
            raise ValueError("relative task-value calibration bundle required")
        self.mode = "current" if schema == "ref2dex.current_decision_task_value.v1" else "prefix"
        self.device = torch.device(device)
        bundle = payload["bundle"]
        self.mean = bundle["mean"].to(self.device)
        self.std = bundle["std"].to(self.device)
        self.beta = bundle["beta"].to(self.device)
        self.y_mean = bundle["y_mean"].to(self.device)

    @torch.no_grad()
    def choose(
        self,
        diagnostics: dict[str, torch.Tensor],
        state: torch.Tensor,
        rest_z: torch.Tensor,
        initial_clearance: torch.Tensor,
        contact: torch.Tensor,
        physical: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        score = diagnostics["score_mm"]
        if self.mode == "current":
            if score.ndim == 3:
                score = score[:, -1]
            fixed = score[:, 7:8]
            relative = score - fixed
            selected_std = diagnostics["relative_std_mm"]
            if selected_std.ndim == 3:
                selected_std = selected_std[:, -1]
            fixed_std = selected_std[:, 7:8]
            relative_std = selected_std - fixed_std
            retention = diagnostics["retention"]
            if retention.ndim == 3:
                retention = retention[:, -1]
            relative_retention = retention - retention[:, 7:8]
            release = diagnostics["release"]
            if release.ndim == 3:
                release = release[:, -1]
            relative_release = release - release[:, 7:8]
            values = [
                relative, relative, relative, relative, selected_std, relative_std,
                relative_retention, relative_release, fixed.expand(-1, 8),
            ]
        else:
            fixed = score[:, :, 7:8]
            relative = score - fixed
            selected_std = diagnostics["relative_std_mm"]
            relative_std = selected_std - selected_std[:, :, 7:8]
            retention = diagnostics["retention"]
            relative_retention = retention - retention[:, :, 7:8]
            release = diagnostics["release"]
            relative_release = release - release[:, :, 7:8]
            values = [
                relative.mean(1), relative[:, -1], relative.amin(1), relative.amax(1),
                selected_std.mean(1), relative_std.mean(1), relative_retention.mean(1),
                relative_release.mean(1), fixed.mean(1).expand(-1, 8),
            ]
        values.extend([
            (state[:, 38] - rest_z)[:, None].expand(-1, 8),
            initial_clearance[:, None].expand(-1, 8),
            contact.bool().all(-1).float()[:, None].expand(-1, 8),
        ])
        features = torch.stack(values, -1)
        normalized = (features - self.mean) / self.std
        calibrated = normalized @ self.beta + self.y_mean

        # Cm remains the safety authority; calibration only changes ranking.
        safe = physical["risk"] & physical["contact"]
        safe = safe & ~physical["ood"][:, None] & ~physical["candidate_ood"]
        safe[:, 7] = True
        choice = calibrated.masked_fill(~safe, -torch.inf).argmax(-1)
        return choice, dict(calibrated=calibrated, safe=safe, features=features)
