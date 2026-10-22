import pytest

from ptv3_shared_wam_v6.curated_sampling import CuratedTauSampler, load_manifest


def manifest():
    rows = []
    for source in (0, 1):
        for seq in (f"s{source}-a", f"s{source}-b"):
            for phase in ("approach", "hold"):
                rows.append(dict(split="train", admitted=True, source_id=source,
                                 sequence_key=seq, phase=phase, tick=len(rows)))
    rows.append(dict(split="dev", admitted=True, source_id=0,
                     sequence_key="dev", phase="approach", tick=0))
    return {"schema": "ref2dex.tau-curated-manifest.v1",
            "status": "AUDITED_NO_TRAINING", "windows": rows}


def test_sampler_is_deterministic_and_identity_agnostic():
    sampler = CuratedTauSampler(manifest(), seed=170, max_per_sequence=16)
    a = sampler.draw(32, step=4, rank=1)
    b = sampler.draw(32, step=4, rank=1)
    assert a == b
    assert all("object" not in row for row in a)
    assert {row["source_id"] for row in a} == {0, 1}


def test_sampler_enforces_sequence_cap():
    sampler = CuratedTauSampler(manifest(), max_per_sequence=1)
    with pytest.raises(ValueError, match="exceeds sequence cap"):
        sampler.draw(128)


def test_v2_local_curated_manifest_is_loadable(tmp_path):
    value = manifest()
    value["schema"] = "ref2dex.tau-curated-manifest.v2"
    value["status"] = "CURATED_LOCAL_WINDOWS_NO_TRAINING"
    path = tmp_path / "curated.json"
    path.write_text(__import__("json").dumps(value))
    loaded = load_manifest(path)
    assert loaded["schema"].endswith(".v2")
    assert CuratedTauSampler(loaded).draw(4)
