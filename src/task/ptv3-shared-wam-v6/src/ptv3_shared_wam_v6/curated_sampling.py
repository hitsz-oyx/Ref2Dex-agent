"""Identity-agnostic sampler for the audited tau window manifest.

This module is separate from ``V5Windows`` and is not wired into training
until the repaired goal contract is frozen. It never reads object names.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping
import json

import numpy as np


PHASES = ("approach", "close", "contact", "hold")


def load_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema") not in {
        "ref2dex.tau-curated-manifest.v1",
        "ref2dex.tau-curated-manifest.v2",
    }:
        raise ValueError("curated tau manifest schema required")
    if data.get("status") not in {
        "AUDITED_NO_TRAINING",
        "CURATED_LOCAL_WINDOWS_NO_TRAINING",
    }:
        raise ValueError("sampler only accepts an audited, non-training manifest")
    if not isinstance(data.get("windows", data.get("selection", {}).get("train_window_rows")), list):
        raise ValueError("curated tau manifest must contain window rows")
    # v2 stores the selected TRAIN rows under selection while retaining the
    # complete audit rows separately.  Normalize the public sampler view to
    # the same admitted-window contract used by v1.
    if "windows" not in data and isinstance(data.get("selection", {}).get("train_window_rows"), list):
        data = dict(data)
        data["windows"] = list(data["selection"]["train_window_rows"])
    return data


class CuratedTauSampler:
    """Draw source/sequence/phase-balanced rows with a per-sequence cap."""

    def __init__(self, manifest: Mapping[str, object], split: str = "train",
                 *, seed: int = 170, max_per_sequence: int = 8):
        if split not in ("train", "dev"):
            raise ValueError("split must be train or dev")
        if max_per_sequence < 1:
            raise ValueError("max_per_sequence must be positive")
        rows = [r for r in manifest.get("windows", [])
                if r.get("split") == split and bool(r.get("admitted"))]
        if not rows:
            raise ValueError("curated manifest has no admitted rows for split")
        self.split = split
        self.seed = int(seed)
        self.max_per_sequence = int(max_per_sequence)
        by_source: dict[int, dict[str, dict[str, list[dict]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        for row in rows:
            source = int(row["source_id"])
            sequence = str(row["sequence_key"])
            phase = str(row["phase"])
            if phase not in PHASES and phase != "unknown":
                raise ValueError(f"unknown phase {phase!r}")
            by_source[source][sequence][phase].append(row)
        self.by_source = by_source
        self.sources = tuple(sorted(by_source))
        self.source_capacities = {
            source: len(by_source[source]) * self.max_per_sequence
            for source in self.sources
        }

    def _draw_source_counts(self, count: int, rng) -> dict[int, int]:
        if count > sum(self.source_capacities.values()):
            raise ValueError(f"batch {count} exceeds sequence cap across all sources")
        chosen = rng.choice(self.sources, size=count, replace=True)
        per_source = {int(s): int((chosen == s).sum()) for s in self.sources}
        # Preserve every previously feasible draw and its RNG stream exactly.
        # Only impossible overflow is reassigned, uniformly among sources with
        # remaining sequence capacity. No source/seed/step-specific exception.
        excess = 0
        for source, capacity in self.source_capacities.items():
            excess += max(0, per_source[source] - capacity)
            per_source[source] = min(per_source[source], capacity)
        for _ in range(excess):
            available = [s for s in self.sources
                         if per_source[s] < self.source_capacities[s]]
            per_source[int(rng.choice(available))] += 1
        return per_source

    def draw(self, count: int, *, step: int = 0, rank: int = 0) -> list[dict]:
        """Draw one deterministic batch and enforce the sequence cap."""
        if count < 1:
            raise ValueError("count must be positive")
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, int(step), int(rank), 20261023]))
        per_source = self._draw_source_counts(count, rng)
        out: list[dict] = []
        for source in self.sources:
            need = per_source[source]
            sequences = tuple(sorted(self.by_source[source]))
            counts = defaultdict(int)
            for _ in range(need):
                available = tuple(s for s in sequences
                                  if counts[s] < self.max_per_sequence)
                sequence = str(rng.choice(available))
                counts[sequence] += 1
                phases = [p for p in PHASES if self.by_source[source][sequence].get(p)]
                if not phases:
                    phases = ["unknown"]
                phase = str(rng.choice(phases))
                pool = self.by_source[source][sequence][phase]
                out.append(pool[int(rng.integers(len(pool)))])
        rng.shuffle(out)
        return out
