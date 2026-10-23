"""Regression for the real rank2 source-overflow that ended update1570."""
from collections import Counter

import pytest

from ptv3_shared_wam_v6.curated_sampling import CuratedTauSampler


def capacity_fixture():
    rows = []
    for source in (0, 1, 2, 4, 5, 6, 8, 9, 10):
        for seq in range(3 if source == 8 else 12):
            for phase in ("approach", "hold"):
                rows.append(dict(split="train", admitted=True, source_id=source,
                                 sequence_key=f"{source}:{seq}", phase=phase,
                                 tick=len(rows)))
    return {"windows": rows}


def assert_admitted_capped_batch(sampler, batch, count):
    assert len(batch) == count
    assert max(Counter(row["sequence_key"] for row in batch).values()) <= sampler.max_per_sequence
    assert all(row in sampler.by_source[row["source_id"]][row["sequence_key"]][row["phase"]]
               for row in batch)


def test_real_prefetched_source_overflow_is_feasible_and_deterministic():
    sampler = CuratedTauSampler(capacity_fixture(), seed=170, max_per_sequence=8)
    # Original iid source draw requests 25 from a capacity24 source at this call.
    batch = sampler.draw(96, step=1579, rank=2)
    assert_admitted_capped_batch(sampler, batch, 96)
    assert batch == sampler.draw(96, step=1579, rank=2)


def test_previously_feasible_draw_is_bit_exact():
    sampler = CuratedTauSampler(capacity_fixture(), seed=170, max_per_sequence=8)
    assert [row["tick"] for row in sampler.draw(12, step=4, rank=1)] == [
        136, 81, 193, 107, 7, 43, 13, 72, 127, 133, 85, 58]


@pytest.mark.parametrize("seed", [1, 170, 901])
def test_capacity_repair_is_general_and_terminates_at_full_capacity(seed):
    rows = [dict(split="train", admitted=True, source_id=sid,
                 sequence_key=f"{sid}:{seq}", phase="unknown", tick=seq)
            for sid, n in [(17, 1), (29, 2), (53, 3)] for seq in range(n)]
    sampler = CuratedTauSampler({"windows": rows}, seed=seed, max_per_sequence=2)
    for step in range(20):
        batch = sampler.draw(12, step=step, rank=1)
        assert_admitted_capped_batch(sampler, batch, 12)
        assert Counter(row["source_id"] for row in batch) == {17: 2, 29: 4, 53: 6}
    with pytest.raises(ValueError, match="exceeds sequence cap"):
        sampler.draw(13)
